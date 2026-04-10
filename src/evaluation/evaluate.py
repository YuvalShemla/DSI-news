"""Evaluate a trained DSI model using constrained beam search."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from src.evaluation.metrics import evaluate_all
from src.model.backbone import load_trained_model
from src.model.constrained_decoding import Prefixer, build_smtid_to_docids

logger = logging.getLogger(__name__)


class QueryDataset(Dataset):
    """Simple dataset for evaluation queries.

    Loads from a TSV file: row_id \\t query_text
    Or from a JSON/JSONL file with query_id and query_text fields.
    """

    def __init__(self, query_path: str | Path):
        self.queries: list[tuple[str, str]] = []  # (query_id, query_text)
        query_path = Path(query_path)

        if query_path.suffix == ".tsv":
            with open(query_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", maxsplit=1)
                    if len(parts) == 2:
                        self.queries.append((parts[0], parts[1]))
        elif query_path.suffix == ".jsonl":
            with open(query_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.queries.append((str(obj["query_id"]), obj["query_text"]))
        elif query_path.suffix == ".json":
            with open(query_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    self.queries.append((str(obj["query_id"]), obj["query_text"]))
            elif isinstance(data, dict):
                for qid, text in data.items():
                    self.queries.append((str(qid), text))
        else:
            raise ValueError(f"Unsupported query file format: {query_path.suffix}")

        logger.info("Loaded %d queries from %s", len(self.queries), query_path)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """Returns (query_id, query_text)."""
        return self.queries[idx]


class QueryCollator:
    """Tokenize queries for generation."""

    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, str]]) -> dict:
        """Returns dict with input_ids, attention_mask, decoder_input_ids, id."""
        qids, texts = zip(*batch)
        encoded = self.tokenizer(
            list(texts),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # decoder_input_ids: start with pad token (standard for T5 generation)
        decoder_start = torch.full(
            (len(qids), 1), self.tokenizer.pad_token_id, dtype=torch.long
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "decoder_input_ids": decoder_start,
            "id": list(qids),
        }


def convert_sequences_to_str_smtids(
    sequences: torch.Tensor, topk: int, max_new_tokens: int
) -> list[list[str]]:
    """Convert generated token sequences to string-joined DocID identifiers.

    sequences: [batch_size * topk, max_new_tokens + 1] tensor
    Returns: list of list of str (batch_size x topk string IDs)
    """
    # Reshape to [batch_size, topk, seq_len]
    batch_size = sequences.size(0) // topk
    reshaped = sequences.view(batch_size, topk, -1).cpu().tolist()

    result = []
    for beam_seqs in reshaped:
        beam_strs = []
        for seq in beam_seqs:
            # Skip the first token (decoder start / pad token)
            beam_strs.append("_".join(str(t) for t in seq[1:]))
        result.append(beam_strs)
    return result


def constrained_beam_search(
    model,
    dataloader: DataLoader,
    prefixer: Prefixer,
    smtid_to_docids: dict[str, list[str]],
    max_new_tokens: int = 8,
    device: str | torch.device = "cuda",
    topk: int = 100,
) -> dict[str, dict[str, float]]:
    """Run constrained beam search over all queries.

    Returns:
        qid_to_rankdata: {query_id: {doc_id: score}}
    """
    model.eval()
    qid_to_rankdata: dict[str, dict[str, float]] = {}
    generation_config = GenerationConfig.from_model_config(model.config)

    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device).long(),
                attention_mask=batch["attention_mask"].to(device).long(),
                generation_config=generation_config,
                prefix_allowed_tokens_fn=prefixer,
                max_new_tokens=max_new_tokens,
                num_beams=topk,
                num_return_sequences=topk,
                output_scores=True,
                return_dict_in_generate=True,
            )

        batch_qids = batch["id"]
        str_smtids = convert_sequences_to_str_smtids(
            outputs.sequences, topk, max_new_tokens
        )
        scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()

        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, scores):
            qid = str(qid)
            qid_to_rankdata[qid] = {}
            for smtid, score in zip(ranked_smtids, rel_scores):
                if smtid in smtid_to_docids:
                    for docid in smtid_to_docids[smtid]:
                        qid_to_rankdata[qid][docid] = score * max_new_tokens

    return qid_to_rankdata


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    query_path: str,
    docid_to_tokenids_path: str,
    qrel_path: str | None = None,
    output_dir: str | None = None,
    topk: int | None = None,
    batch_size: int | None = None,
) -> dict:
    """Full evaluation pipeline: load model -> beam search -> metrics."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})
    max_new_tokens = eval_cfg.get("max_new_tokens", 8)
    topk = topk or eval_cfg.get("topk", 100)
    batch_size = batch_size or config.get("training", {}).get("d0", {}).get("batch_size", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model, tokenizer = load_trained_model(
        checkpoint_path, model_cfg.get("backbone", "google/t5gemma-2-270m-270m"), dtype
    )
    model.to(device)
    model.eval()

    # Build prefixer and smtid mapping
    prefixer = Prefixer(
        docid_to_tokenids_path=docid_to_tokenids_path,
        tokenizer=tokenizer,
    )
    smtid_to_docids = build_smtid_to_docids(
        docid_to_tokenids_path, max_new_tokens=max_new_tokens
    )

    # Build dataloader
    dataset = QueryDataset(query_path)
    collator = QueryCollator(
        tokenizer, max_length=config.get("training", {}).get("d0", {}).get("max_query_length", 64)
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    # Run beam search
    qid_to_rankdata = constrained_beam_search(
        model, dataloader, prefixer, smtid_to_docids,
        max_new_tokens=max_new_tokens, device=device, topk=topk,
    )

    # Save run file
    output_dir = Path(output_dir or "outputs/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    with open(run_path, "w") as f:
        json.dump(qid_to_rankdata, f)
    logger.info("Saved run file to %s (%d queries)", run_path, len(qid_to_rankdata))

    # Compute metrics if qrels available
    results = {}
    if qrel_path is not None:
        with open(qrel_path) as f:
            qrel = json.load(f)
        metric_names = eval_cfg.get("metrics", None)
        results = evaluate_all(qid_to_rankdata, qrel, metrics=metric_names)
        for name, value in results.items():
            logger.info("%s: %.4f", name, value)

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved metrics to %s", metrics_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Evaluate a trained DSI model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--docid-to-tokenids", type=str, required=True)
    parser.add_argument("--qrels", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    run_evaluation(
        args.config, args.checkpoint, args.queries,
        args.docid_to_tokenids, args.qrels, args.output_dir,
        topk=args.topk, batch_size=args.batch_size,
    )
