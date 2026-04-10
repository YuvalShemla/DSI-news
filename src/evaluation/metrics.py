"""IR evaluation metrics using pytrec_eval."""

from __future__ import annotations

import json
import logging
from collections import Counter

import pytrec_eval
from pytrec_eval import RelevanceEvaluator

logger = logging.getLogger(__name__)


def truncate_run(run: dict, k: int) -> dict:
    """Truncate run file to top-k results per query."""
    truncated = {}
    for qid in run:
        sorted_docs = sorted(run[qid].items(), key=lambda item: item[1], reverse=True)
        truncated[qid] = {doc: score for doc, score in sorted_docs[:k]}
    return truncated


def mrr_k(run: dict, qrel: dict, k: int, agg: bool = True):
    """Mean Reciprocal Rank at k."""
    evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)
    result = evaluator.evaluate(truncated)
    if agg:
        return sum(d["recip_rank"] for d in result.values()) / max(1, len(result))
    return result


def recall_k(run: dict, qrel: dict, k: int, agg: bool = True):
    """Recall at k."""
    evaluator = RelevanceEvaluator(qrel, {"recall"})
    out_eval = evaluator.evaluate(run)
    if agg:
        key = f"recall_{k}"
        total = 0.0
        for v_dict in out_eval.values():
            total += v_dict.get(key, 0.0)
        return total / max(1, len(out_eval))
    return out_eval


def ndcg_k(run: dict, qrel: dict, k: int, agg: bool = True):
    """nDCG at k."""
    metric = "ndcg_cut"
    evaluator = RelevanceEvaluator(qrel, {metric})
    out_eval = evaluator.evaluate(run)
    if agg:
        key = f"ndcg_cut_{k}"
        total = 0.0
        for v_dict in out_eval.values():
            total += v_dict.get(key, 0.0)
        return total / max(1, len(out_eval))
    return out_eval


def map_score(run: dict, qrel: dict, agg: bool = True):
    """Mean Average Precision."""
    evaluator = RelevanceEvaluator(qrel, {"map"})
    out_eval = evaluator.evaluate(run)
    if agg:
        total = 0.0
        for v_dict in out_eval.values():
            total += v_dict.get("map", 0.0)
        return total / max(1, len(out_eval))
    return out_eval


def evaluate(run: dict, qrel: dict, metric: str, agg: bool = True,
             select: int | None = None):
    """General evaluation using any pytrec_eval metric."""
    assert metric in pytrec_eval.supported_measures, (
        f"Unsupported metric: {metric}. Use one of pytrec_eval.supported_measures."
    )
    evaluator = RelevanceEvaluator(qrel, {metric})
    out_eval = evaluator.evaluate(run)
    if agg:
        res = Counter({})
        for d in out_eval.values():
            res += Counter(d)
        res = {k: v / max(1, len(out_eval)) for k, v in res.items()}
        if select is not None:
            key = f"{metric}_{select}"
            return res.get(key, 0.0)
        return dict(res)
    return out_eval


def evaluate_all(run: dict, qrel: dict, metrics: list[str] | None = None) -> dict:
    """Evaluate with multiple metrics at once. Returns {metric_name: value}.

    Default metrics: MRR@10, nDCG@5, nDCG@10, Recall@10, Recall@100, MAP.
    """
    results = {}
    if metrics is None:
        metrics = ["mrr_10", "ndcg_cut_5", "ndcg_cut_10", "recall_10", "recall_100", "map"]

    for metric in metrics:
        if metric == "mrr_10":
            results["mrr_10"] = mrr_k(run, qrel, k=10)
        elif metric.startswith("ndcg_cut_"):
            k = int(metric.split("_")[-1])
            results[metric] = ndcg_k(run, qrel, k=k)
        elif metric.startswith("recall_"):
            k = int(metric.split("_")[-1])
            results[metric] = recall_k(run, qrel, k=k)
        elif metric == "map":
            results["map"] = map_score(run, qrel)
        else:
            result = evaluate(run, qrel, metric=metric)
            results.update(result if isinstance(result, dict) else {metric: result})

    return results


def load_and_evaluate(qrel_path: str, run_path: str,
                      metrics: list[str] | None = None) -> dict:
    """Load files and evaluate."""
    with open(qrel_path) as f:
        qrel = json.load(f)
    with open(run_path) as f:
        run = json.load(f)

    results = evaluate_all(run, qrel, metrics=metrics)
    for name, value in results.items():
        logger.info("%s: %.4f", name, value)
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Evaluate IR metrics from run/qrel files")
    parser.add_argument("--qrel", type=str, required=True, help="Path to qrel.json")
    parser.add_argument("--run", type=str, required=True, help="Path to run.json")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to compute")
    args = parser.parse_args()

    results = load_and_evaluate(args.qrel, args.run, metrics=args.metrics)
    print(json.dumps(results, indent=2))
