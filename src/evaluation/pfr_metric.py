"""Positive Forgetting Rate (PFR) -- a novel metric for beneficial forgetting in DSI.

PFR measures the rate at which a model replaces outdated documents with newer,
equally-or-more relevant documents in its rankings. High PFR indicates the model
is correctly "forgetting" old documents as fresher alternatives emerge.

Formal definition (from proposal.ltx Section 3.2):
For a query q, let R_t be the ranked list at time t and R_{t+1} at time t+1.
A "positive forget" occurs when:
  1. A document d_old in R_t is NOT in R_{t+1}
  2. A document d_new in R_{t+1} but NOT in R_t replaces it
  3. d_new is at least as relevant as d_old (per qrels)
  4. d_new is more recent than d_old

PFR = |positive forgets| / |all forgets|
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_topk_docids(run_for_query: dict[str, float], k: int) -> list[str]:
    """Return top-k doc IDs sorted by descending score."""
    sorted_docs = sorted(run_for_query.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]


def _get_relevance(qrel_for_query: dict[str, int], docid: str) -> int:
    """Get relevance grade for a document (0 if unjudged)."""
    return qrel_for_query.get(docid, 0)


def _parse_date(date_str: str) -> str:
    """Normalize date string for comparison. Returns YYYY-MM-DD."""
    # Handle common formats: YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD
    date_str = date_str.strip().replace("/", "-")
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str[:10]  # Truncate to YYYY-MM-DD


def compute_pfr(
    run_before: dict[str, dict[str, float]],
    run_after: dict[str, dict[str, float]],
    qrel: dict[str, dict[str, int]],
    doc_dates: dict[str, str],
    k: int = 10,
) -> dict:
    """Compute Positive Forgetting Rate between two time steps.

    Args:
        run_before: Run file at time t: {qid: {docid: score}}
        run_after: Run file at time t+1: {qid: {docid: score}}
        qrel: Relevance judgments: {qid: {docid: relevance_int}}
        doc_dates: Document publication dates: {docid: "YYYY-MM-DD"}
        k: Evaluate over top-k results

    Returns:
        dict with:
            - pfr: overall PFR score
            - per_query_pfr: {qid: pfr_score}
            - total_forgets: total number of documents forgotten
            - positive_forgets: number of positive forgets
            - negative_forgets: number of negative forgets
    """
    # Operate over queries present in both runs
    common_qids = set(run_before.keys()) & set(run_after.keys())

    total_forgets = 0
    positive_forgets = 0
    negative_forgets = 0
    per_query_pfr: dict[str, float] = {}

    for qid in common_qids:
        topk_before = set(_get_topk_docids(run_before[qid], k))
        topk_after = set(_get_topk_docids(run_after[qid], k))

        # Forgotten docs: in top-k before but not after
        forgotten = topk_before - topk_after
        # New docs: in top-k after but not before
        newcomers = topk_after - topk_before

        if not forgotten:
            per_query_pfr[qid] = 1.0  # Nothing forgotten => no negative forgetting
            continue

        qrel_q = qrel.get(qid, {})
        q_positive = 0
        q_negative = 0

        for d_old in forgotten:
            old_rel = _get_relevance(qrel_q, d_old)
            old_date = _parse_date(doc_dates.get(d_old, "0000-00-00"))

            # Check if any newcomer is a valid positive replacement
            replaced = False
            for d_new in newcomers:
                new_rel = _get_relevance(qrel_q, d_new)
                new_date = _parse_date(doc_dates.get(d_new, "0000-00-00"))

                if new_rel >= old_rel and new_date > old_date:
                    replaced = True
                    break

            if replaced:
                q_positive += 1
            else:
                q_negative += 1

        q_total = q_positive + q_negative
        per_query_pfr[qid] = q_positive / q_total if q_total > 0 else 1.0
        total_forgets += q_total
        positive_forgets += q_positive
        negative_forgets += q_negative

    pfr = positive_forgets / total_forgets if total_forgets > 0 else 1.0

    return {
        "pfr": pfr,
        "per_query_pfr": per_query_pfr,
        "total_forgets": total_forgets,
        "positive_forgets": positive_forgets,
        "negative_forgets": negative_forgets,
        "num_queries": len(common_qids),
    }


def compute_weighted_pfr(
    run_before: dict[str, dict[str, float]],
    run_after: dict[str, dict[str, float]],
    qrel: dict[str, dict[str, int]],
    doc_dates: dict[str, str],
    k: int = 10,
) -> dict:
    """Weighted PFR -- weights positive forgets by the relevance improvement.

    A replacement where d_new has rel=3 replacing d_old with rel=1 counts more
    than a replacement where both have rel=1.
    """
    common_qids = set(run_before.keys()) & set(run_after.keys())

    total_weight = 0.0
    positive_weight = 0.0
    per_query_wpfr: dict[str, float] = {}

    for qid in common_qids:
        topk_before = set(_get_topk_docids(run_before[qid], k))
        topk_after = set(_get_topk_docids(run_after[qid], k))

        forgotten = topk_before - topk_after
        newcomers = topk_after - topk_before

        if not forgotten:
            per_query_wpfr[qid] = 1.0
            continue

        qrel_q = qrel.get(qid, {})
        q_total_w = 0.0
        q_pos_w = 0.0

        for d_old in forgotten:
            old_rel = _get_relevance(qrel_q, d_old)
            old_date = _parse_date(doc_dates.get(d_old, "0000-00-00"))
            # Base weight of 1 for each forget event
            q_total_w += 1.0

            best_improvement = 0.0
            best_replaced = False
            for d_new in newcomers:
                new_rel = _get_relevance(qrel_q, d_new)
                new_date = _parse_date(doc_dates.get(d_new, "0000-00-00"))
                if new_rel >= old_rel and new_date > old_date:
                    improvement = 1.0 + max(0, new_rel - old_rel)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_replaced = True

            if best_replaced:
                q_pos_w += best_improvement

        per_query_wpfr[qid] = q_pos_w / q_total_w if q_total_w > 0 else 1.0
        total_weight += q_total_w
        positive_weight += q_pos_w

    wpfr = positive_weight / total_weight if total_weight > 0 else 1.0

    return {
        "weighted_pfr": wpfr,
        "per_query_wpfr": per_query_wpfr,
        "total_weight": total_weight,
        "positive_weight": positive_weight,
        "num_queries": len(common_qids),
    }


def compute_temporal_pfr(
    runs_over_time: list[dict[str, dict[str, float]]],
    qrels_over_time: list[dict[str, dict[str, int]]],
    doc_dates: dict[str, str],
    k: int = 10,
) -> list[dict]:
    """Compute PFR across all time step pairs.

    Args:
        runs_over_time: List of run dicts, one per time step [R_0, R_1, ..., R_T]
        qrels_over_time: List of qrel dicts, one per time step
        doc_dates: Document dates
        k: Top-k

    Returns:
        List of PFR results, one for each (t, t+1) pair
    """
    if len(runs_over_time) != len(qrels_over_time):
        raise ValueError(
            f"Mismatched lengths: {len(runs_over_time)} runs vs "
            f"{len(qrels_over_time)} qrels"
        )

    results = []
    for t in range(len(runs_over_time) - 1):
        # Use the qrels from the later time step (t+1) as the ground truth,
        # since it reflects the updated relevance landscape
        pfr_result = compute_pfr(
            runs_over_time[t], runs_over_time[t + 1],
            qrels_over_time[t + 1], doc_dates, k=k,
        )
        pfr_result["time_step"] = (t, t + 1)
        results.append(pfr_result)

    return results


def compute_per_topic_pfr(
    run_before: dict[str, dict[str, float]],
    run_after: dict[str, dict[str, float]],
    qrel: dict[str, dict[str, int]],
    doc_dates: dict[str, str],
    query_topics: dict[str, str],
    k: int = 10,
) -> dict[str, dict]:
    """PFR broken down by topic/category.

    Args:
        query_topics: {qid: topic_label}

    Returns:
        {topic_label: pfr_result_dict}
    """
    # Group queries by topic
    topic_qids: dict[str, list[str]] = defaultdict(list)
    for qid, topic in query_topics.items():
        topic_qids[topic].append(qid)

    topic_results = {}
    for topic, qids in topic_qids.items():
        # Build sub-runs with only this topic's queries
        qid_set = set(qids)
        sub_before = {q: run_before[q] for q in qid_set if q in run_before}
        sub_after = {q: run_after[q] for q in qid_set if q in run_after}
        sub_qrel = {q: qrel[q] for q in qid_set if q in qrel}

        if sub_before and sub_after:
            topic_results[topic] = compute_pfr(
                sub_before, sub_after, sub_qrel, doc_dates, k=k
            )
            topic_results[topic]["num_topic_queries"] = len(qid_set)

    return topic_results


def load_doc_dates(manifest_path: str | Path) -> dict[str, str]:
    """Load document dates from a JSON manifest.

    Expected format: {"doc_id": "YYYY-MM-DD", ...}
    Or a list of dicts with "id" and "date" fields.
    """
    with open(manifest_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {str(item["id"]): item["date"] for item in data}
    else:
        raise ValueError(f"Unexpected format in {manifest_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Compute Positive Forgetting Rate")
    parser.add_argument("--run-before", required=True, help="Run file at time t")
    parser.add_argument("--run-after", required=True, help="Run file at time t+1")
    parser.add_argument("--qrels", required=True, help="Qrel file")
    parser.add_argument("--doc-dates", required=True, help="Document dates JSON")
    parser.add_argument("--k", type=int, default=10, help="Top-k for evaluation")
    parser.add_argument("--output", type=str, default=None, help="Output path for results")
    args = parser.parse_args()

    with open(args.run_before) as f:
        run_before = json.load(f)
    with open(args.run_after) as f:
        run_after = json.load(f)
    with open(args.qrels) as f:
        qrel = json.load(f)
    doc_dates = load_doc_dates(args.doc_dates)

    result = compute_pfr(run_before, run_after, qrel, doc_dates, k=args.k)
    weighted = compute_weighted_pfr(run_before, run_after, qrel, doc_dates, k=args.k)
    result["weighted_pfr"] = weighted["weighted_pfr"]

    logger.info("PFR@%d: %.4f", args.k, result["pfr"])
    logger.info("Weighted PFR@%d: %.4f", args.k, result["weighted_pfr"])
    logger.info("Positive forgets: %d / %d total", result["positive_forgets"], result["total_forgets"])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove per-query data for the summary file
        summary = {k: v for k, v in result.items() if k != "per_query_pfr"}
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved results to %s", output_path)
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "per_query_pfr"}, indent=2))
