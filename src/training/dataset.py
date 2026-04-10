"""Training dataset and collator for chrono-semantic DocID generation."""

import json
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ChronoDocIDDataset(Dataset):
    """Dataset for (query, docid_tokens) pairs.

    Loads examples from JSONL and maps doc_ids to token ID sequences.
    For filter-aware training: a fraction of samples get [FILTER:YYYY] or
    [FILTER:YYYY-MM] prepended to teach the model temporal filtering.
    """

    def __init__(
        self,
        examples_path: str | Path,
        docid_to_tokenids_path: str | Path,
        filter_ratio: float = 0.3,
    ):
        """
        Args:
            examples_path: Path to JSONL file with {"doc_id", "query", "date"} per line.
            docid_to_tokenids_path: Path to docid_to_tokenids.json mapping doc_id
                to a list of 8 absolute vocab token IDs.
            filter_ratio: Fraction of examples to augment with a time filter prefix.
        """
        self.filter_ratio = filter_ratio

        with open(docid_to_tokenids_path) as f:
            docid_to_tokenids = json.load(f)

        self.examples: list[tuple[str, list[int], str]] = []
        with open(examples_path) as f:
            for line in f:
                ex = json.loads(line)
                doc_id = str(ex["doc_id"])
                if doc_id not in docid_to_tokenids:
                    continue
                token_ids = docid_to_tokenids[doc_id]
                self.examples.append((ex["query"], token_ids, ex.get("date", "")))

        logger.info(
            "Loaded %d training examples from %s", len(self.examples), examples_path
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, list[int]]:
        """Returns (query_str, docid_token_ids_list)."""
        query, token_ids, date = self.examples[idx]

        if date and random.random() < self.filter_ratio:
            # Prepend a temporal filter: 50/50 year-only vs year-month
            year = date[:4]
            month = date[5:7] if len(date) >= 7 else None
            if month and random.random() < 0.5:
                query = f"[FILTER:{year}-{month}] {query}"
            else:
                query = f"[FILTER:{year}] {query}"

        return query, token_ids


class ChronoDocIDCollator:
    """Tokenizes queries and builds decoder inputs for Seq2Seq training."""

    def __init__(self, tokenizer_path_or_name: str, max_query_length: int = 64):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        self.max_length = max_query_length

    def __call__(self, batch: list[tuple[str, list[int]]]) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - input_ids: tokenized queries [B, seq_len]
                - attention_mask: [B, seq_len]
                - decoder_input_ids: [pad, tok0, ..., tok6] shifted right [B, 8]
                - labels: [tok0, tok1, ..., tok7] target DocID tokens [B, 8]
        """
        queries, docid_tokenids = zip(*batch)

        tokenized = self.tokenizer(
            list(queries),
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = torch.LongTensor(list(docid_tokenids))
        batch_size = labels.size(0)
        prefix = torch.full(
            (batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long
        )
        decoder_input_ids = torch.hstack((prefix, labels[:, :-1]))

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }


class ReplayDataset(Dataset):
    """Wraps a ChronoDocIDDataset with a replay buffer for continual learning.

    Interleaves current-period examples with replayed past examples according
    to the replay_ratio.
    """

    def __init__(
        self,
        current_dataset: ChronoDocIDDataset,
        replay_buffer,
        replay_ratio: float = 0.3,
    ):
        """
        Args:
            current_dataset: ChronoDocIDDataset for the current time period.
            replay_buffer: ReplayBuffer instance with past examples.
            replay_ratio: Fraction of each effective dataset that comes from replay.
        """
        self.current_dataset = current_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio

        # Pre-sample replay indices so __len__ is deterministic within an epoch.
        num_current = len(current_dataset)
        num_replay = int(num_current * replay_ratio / (1 - replay_ratio)) if replay_ratio < 1.0 else num_current
        num_replay = min(num_replay, len(replay_buffer))
        self._replay_samples = replay_buffer.sample(num_replay) if num_replay > 0 else []
        self._total = num_current + len(self._replay_samples)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> tuple[str, list[int]]:
        num_current = len(self.current_dataset)
        if idx < num_current:
            return self.current_dataset[idx]
        replay_idx = idx - num_current
        return self._replay_samples[replay_idx]


def create_training_examples(
    parquet_path: str | Path,
    queries_path: str | Path,
    docid_to_tokenids_path: str | Path,
    output_path: str | Path,
    text_col: str = "text",
    id_col: str | None = None,
    date_col: str = "date",
) -> int:
    """Create JSONL training examples from parquet data + generated queries.

    Reads the parquet for document metadata and a queries JSONL file that
    contains {"doc_id": ..., "query": ...} lines. Joins them on doc_id and
    writes (query, doc_id, date) triples to output_path.

    Returns the number of examples written.
    """
    df = pd.read_parquet(parquet_path)

    with open(docid_to_tokenids_path) as f:
        docid_to_tokenids = json.load(f)
    valid_ids = set(docid_to_tokenids.keys())

    # Build a lookup from doc_id to date
    if id_col and id_col in df.columns:
        id_to_date = dict(zip(df[id_col].astype(str), df[date_col].astype(str)))
    else:
        id_to_date = dict(zip(df.index.astype(str), df[date_col].astype(str)))

    count = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(queries_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            ex = json.loads(line)
            doc_id = str(ex["doc_id"])
            if doc_id not in valid_ids:
                continue
            date = id_to_date.get(doc_id, "")
            out = {"doc_id": doc_id, "query": ex["query"], "date": date}
            fout.write(json.dumps(out) + "\n")
            count += 1

    logger.info("Wrote %d training examples to %s", count, output_path)
    return count
