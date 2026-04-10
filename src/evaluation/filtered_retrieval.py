"""Evaluate filtered retrieval: time-constrained queries using chrono-semantic DocIDs.

This evaluates the unique capability of chrono-semantic DocIDs: handling queries like
"[FILTER:2023] climate change policy" that should only return documents from 2023.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import GenerationConfig

from src.evaluation.evaluate import (
    QueryCollator,
    QueryDataset,
    constrained_beam_search,
    convert_sequences_to_str_smtids,
)
from src.evaluation.metrics import evaluate_all, ndcg_k, truncate_run
from src.model.backbone import load_trained_model
from src.model.constrained_decoding import (
    FilteredPrefixer,
    build_smtid_to_docids,
    parse_filter_from_query,
)

logger = logging.getLogger(__name__)


def _parse_date(date_str: str) -> tuple[int | None, int | None]:
    """Extract year and month from a date string."""
    date_str = date_str.strip().replace("/", "-")
    parts = date_str.split("-")
    year = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else None
    month = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
    return year, month


def _doc_matches_filter(
    docid: str,
    doc_dates: dict[str, str],
    filter_year: int | None,
    filter_month: int | None,
) -> bool:
    """Check if a document's date matches the given filter."""
    date_str = doc_dates.get(docid)
    if date_str is None:
        return False
    year, month = _parse_date(date_str)
    if filter_year is not None and year != filter_year:
        return False
    if filter_month is not None and month != filter_month:
        return False
    return True


def compute_filter_precision(
    run: dict[str, dict[str, float]],
    doc_dates: dict[str, str],
    query_filters: dict[str, dict],
    k: int = 10,
) -> dict:
    """Compute filter precision: % of top-k results in the correct time range.

    Args:
        run: {qid: {docid: score}}
        doc_dates: {docid: "YYYY-MM-DD"}
        query_filters: {qid: {"year": int, "month": int|None}}
        k: Top-k

    Returns:
        dict with:
            - filter_precision: overall precision
            - per_query: {qid: precision}
            - year_precision: precision when filtering by year only
            - month_precision: precision when filtering by year+month
    """
    per_query: dict[str, float] = {}
    year_only_precisions = []
    month_precisions = []

    for qid, filt in query_filters.items():
        if qid not in run:
            continue

        filter_year = filt.get("year")
        filter_month = filt.get("month")

        # Get top-k docs
        sorted_docs = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        if not sorted_docs:
            continue

        matches = sum(
            1 for docid, _ in sorted_docs
            if _doc_matches_filter(docid, doc_dates, filter_year, filter_month)
        )
        precision = matches / len(sorted_docs)
        per_query[qid] = precision

        if filter_month is not None:
            month_precisions.append(precision)
        else:
            year_only_precisions.append(precision)

    all_precisions = list(per_query.values())
    filter_precision = sum(all_precisions) / max(1, len(all_precisions))
    year_precision = (
        sum(year_only_precisions) / max(1, len(year_only_precisions))
        if year_only_precisions else None
    )
    month_precision = (
        sum(month_precisions) / max(1, len(month_precisions))
        if month_precisions else None
    )

    return {
        "filter_precision": filter_precision,
        "per_query": per_query,
        "year_precision": year_precision,
        "month_precision": month_precision,
        "num_queries": len(per_query),
    }


def compute_filtered_ndcg(
    run: dict[str, dict[str, float]],
    qrel: dict[str, dict[str, int]],
    doc_dates: dict[str, str],
    query_filters: dict[str, dict],
    k: int = 10,
) -> dict:
    """nDCG computed only over documents in the correct time range.

    For each query, we remove out-of-range documents from both the run and qrel
    before computing nDCG, measuring retrieval quality WITHIN the filtered range.
    """
    filtered_run: dict[str, dict[str, float]] = {}
    filtered_qrel: dict[str, dict[str, int]] = {}

    for qid, filt in query_filters.items():
        if qid not in run:
            continue

        filter_year = filt.get("year")
        filter_month = filt.get("month")

        # Filter run to only in-range docs
        filtered_run[qid] = {
            docid: score
            for docid, score in run[qid].items()
            if _doc_matches_filter(docid, doc_dates, filter_year, filter_month)
        }

        # Filter qrels similarly
        if qid in qrel:
            filtered_qrel[qid] = {
                docid: rel
                for docid, rel in qrel[qid].items()
                if _doc_matches_filter(docid, doc_dates, filter_year, filter_month)
            }

    if not filtered_run or not filtered_qrel:
        return {"filtered_ndcg": 0.0, "num_queries": 0}

    score = ndcg_k(filtered_run, filtered_qrel, k=k)
    return {
        "filtered_ndcg": score,
        "num_queries": len(filtered_run),
    }


def evaluate_filtered_vs_unfiltered(
    filtered_run: dict[str, dict[str, float]],
    unfiltered_run: dict[str, dict[str, float]],
    qrel: dict[str, dict[str, int]],
    doc_dates: dict[str, str],
    query_filters: dict[str, dict],
    k: int = 10,
) -> dict:
    """Compare filtered vs unfiltered retrieval on the same queries.

    Shows the benefit of time filtering by computing metrics for both and
    reporting the delta.
    """
    # Compute filter precision for both
    filtered_fp = compute_filter_precision(filtered_run, doc_dates, query_filters, k)
    unfiltered_fp = compute_filter_precision(unfiltered_run, doc_dates, query_filters, k)

    # Compute filtered nDCG for both
    filtered_ndcg = compute_filtered_ndcg(filtered_run, qrel, doc_dates, query_filters, k)
    unfiltered_ndcg = compute_filtered_ndcg(unfiltered_run, qrel, doc_dates, query_filters, k)

    # Standard metrics on the filtered queries subset
    common_qids = set(query_filters.keys()) & set(qrel.keys())
    sub_qrel = {q: qrel[q] for q in common_qids if q in qrel}

    filtered_metrics = {}
    unfiltered_metrics = {}
    if sub_qrel:
        sub_filtered = {q: filtered_run[q] for q in common_qids if q in filtered_run}
        sub_unfiltered = {q: unfiltered_run[q] for q in common_qids if q in unfiltered_run}
        if sub_filtered:
            filtered_metrics = evaluate_all(sub_filtered, sub_qrel)
        if sub_unfiltered:
            unfiltered_metrics = evaluate_all(sub_unfiltered, sub_qrel)

    return {
        "filtered": {
            "filter_precision": filtered_fp["filter_precision"],
            "filtered_ndcg": filtered_ndcg["filtered_ndcg"],
            **filtered_metrics,
        },
        "unfiltered": {
            "filter_precision": unfiltered_fp["filter_precision"],
            "filtered_ndcg": unfiltered_ndcg["filtered_ndcg"],
            **unfiltered_metrics,
        },
        "delta": {
            "filter_precision": (
                filtered_fp["filter_precision"] - unfiltered_fp["filter_precision"]
            ),
            "filtered_ndcg": (
                filtered_ndcg["filtered_ndcg"] - unfiltered_ndcg["filtered_ndcg"]
            ),
        },
        "num_queries": len(common_qids),
    }


def run_filtered_evaluation(
    config_path: str,
    checkpoint_path: str,
    query_path: str,
    docid_to_tokenids_path: str,
    qrel_path: str,
    doc_dates_path: str,
    output_dir: str,
) -> dict:
    """Full filtered retrieval evaluation pipeline.

    1. Load model with FilteredPrefixer
    2. For each query with a filter:
       a. Parse filter from query
       b. Set filter on prefixer
       c. Run constrained beam search
       d. Clear filter
    3. Compute filter precision and filtered nDCG
    4. Compare with unfiltered results
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})
    docid_cfg = config.get("docid", {})
    max_new_tokens = eval_cfg.get("max_new_tokens", 8)
    topk = eval_cfg.get("topk", 100)
    batch_size = config.get("training", {}).get("d0", {}).get("batch_size", 32)
    max_query_length = config.get("training", {}).get("d0", {}).get("max_query_length", 64)
    year_range_start = docid_cfg.get("year_range", [2017, 2025])[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model, tokenizer = load_trained_model(
        checkpoint_path, model_cfg.get("backbone", "google/t5gemma-2-270m-270m"), dtype
    )
    model.to(device)
    model.eval()

    # Load doc dates
    with open(doc_dates_path) as f:
        doc_dates = json.load(f)

    # Build smtid mapping
    smtid_to_docids = build_smtid_to_docids(
        docid_to_tokenids_path, max_new_tokens=max_new_tokens
    )

    # Load qrels
    with open(qrel_path) as f:
        qrel = json.load(f)

    # We need token_id_ranges for FilteredPrefixer. Load from docid_tokenizer module.
    from src.model.docid_tokenizer import get_token_id_ranges
    token_id_ranges = get_token_id_ranges(
        tokenizer,
        year_range=(year_range_start, docid_cfg.get("year_range", [2017, 2025])[1]),
        num_codebooks=docid_cfg.get("rq_codebooks", 6),
        codebook_size=2 ** docid_cfg.get("rq_bits", 8),
    )

    # Build FilteredPrefixer
    prefixer = FilteredPrefixer(
        docid_to_tokenids_path=docid_to_tokenids_path,
        tokenizer=tokenizer,
        token_id_ranges=token_id_ranges,
    )

    # Load and parse queries
    dataset = QueryDataset(query_path)
    query_filters: dict[str, dict] = {}  # qid -> {"year": ..., "month": ...}
    clean_queries: list[tuple[str, str]] = []  # (qid, clean_text)

    for qid, query_text in dataset.queries:
        clean_text, filter_year, filter_month = parse_filter_from_query(query_text)
        clean_queries.append((qid, clean_text))
        if filter_year is not None:
            query_filters[qid] = {"year": filter_year, "month": filter_month}

    logger.info(
        "Parsed %d queries: %d with filters, %d without",
        len(clean_queries), len(query_filters), len(clean_queries) - len(query_filters),
    )

    # Run filtered evaluation: process queries one at a time for filter control
    # For efficiency, batch unfiltered queries together and filtered queries per-filter
    collator = QueryCollator(tokenizer, max_length=max_query_length)
    generation_config = GenerationConfig.from_model_config(model.config)

    filtered_run: dict[str, dict[str, float]] = {}
    unfiltered_run: dict[str, dict[str, float]] = {}

    # First pass: unfiltered (all queries, no filter)
    prefixer.clear_filter()
    unfiltered_dataset = QueryDataset.__new__(QueryDataset)
    unfiltered_dataset.queries = clean_queries
    unfiltered_loader = DataLoader(
        unfiltered_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    unfiltered_run = constrained_beam_search(
        model, unfiltered_loader, prefixer, smtid_to_docids,
        max_new_tokens=max_new_tokens, device=device, topk=topk,
    )

    # Second pass: filtered queries only, with per-query filters
    # Group by (year, month) to batch queries with the same filter
    filter_groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for qid, clean_text in clean_queries:
        if qid in query_filters:
            filt = query_filters[qid]
            key = (filt["year"], filt.get("month"))
            filter_groups[key].append((qid, clean_text))

    for (year, month), group_queries in filter_groups.items():
        prefixer.set_filter(filter_year=year, filter_month=month,
                            year_range_start=year_range_start)
        group_dataset = QueryDataset.__new__(QueryDataset)
        group_dataset.queries = group_queries
        group_loader = DataLoader(
            group_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
        )
        group_run = constrained_beam_search(
            model, group_loader, prefixer, smtid_to_docids,
            max_new_tokens=max_new_tokens, device=device, topk=topk,
        )
        filtered_run.update(group_run)
        prefixer.clear_filter()

    # For unfiltered queries (no filter tag), copy from unfiltered_run
    for qid, _ in clean_queries:
        if qid not in query_filters and qid not in filtered_run:
            filtered_run[qid] = unfiltered_run.get(qid, {})

    # Compute metrics
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save runs
    with open(output_dir / "filtered_run.json", "w") as f:
        json.dump(filtered_run, f)
    with open(output_dir / "unfiltered_run.json", "w") as f:
        json.dump(unfiltered_run, f)

    # Filter precision
    fp_result = compute_filter_precision(filtered_run, doc_dates, query_filters)
    logger.info("Filter precision@%d: %.4f", 10, fp_result["filter_precision"])

    # Filtered nDCG
    fndcg = compute_filtered_ndcg(filtered_run, qrel, doc_dates, query_filters)
    logger.info("Filtered nDCG@10: %.4f", fndcg["filtered_ndcg"])

    # Comparison
    comparison = evaluate_filtered_vs_unfiltered(
        filtered_run, unfiltered_run, qrel, doc_dates, query_filters
    )

    results = {
        "filter_precision": fp_result,
        "filtered_ndcg": fndcg,
        "comparison": comparison,
    }

    with open(output_dir / "filtered_eval_results.json", "w") as f:
        # Remove per-query data for the summary
        summary = {
            "filter_precision": fp_result["filter_precision"],
            "year_precision": fp_result["year_precision"],
            "month_precision": fp_result["month_precision"],
            "filtered_ndcg": fndcg["filtered_ndcg"],
            "comparison": comparison,
        }
        json.dump(summary, f, indent=2)
    logger.info("Saved filtered evaluation results to %s", output_dir)

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Evaluate filtered retrieval")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--docid-to-tokenids", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--doc-dates", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/filtered_eval")
    args = parser.parse_args()

    run_filtered_evaluation(
        args.config, args.checkpoint, args.queries,
        args.docid_to_tokenids, args.qrels, args.doc_dates, args.output_dir,
    )
