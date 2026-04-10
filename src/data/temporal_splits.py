"""
Split datasets into temporal periods for continual learning.

BBC News  -> monthly splits (e.g. "2017-01", "2017-02", ...)
CC-News   -> quarterly splits (e.g. "2017-Q1", "2017-Q2", ...)
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# BBC News -- monthly splits
# ---------------------------------------------------------------------------

def split_bbc_news(parquet_path: str | Path, output_dir: str | Path,
                   date_col: str = "published_date") -> dict:
    """Split BBC News by month. Returns manifest dict."""
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading BBC News from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[date_col])
    logger.info("Dropped %d rows with invalid dates (%d remaining)", n_before - len(df), len(df))

    df["_period"] = df[date_col].dt.to_period("M").astype(str)  # "2017-01"

    manifest = {}
    for period, group in sorted(df.groupby("_period")):
        split_name = period  # e.g. "2017-01"
        split_path = output_dir / f"{split_name}.parquet"
        group.drop(columns=["_period"]).to_parquet(split_path, index=False)
        manifest[split_name] = {
            "parquet_path": str(split_path),
            "start_date": str(group[date_col].min().date()),
            "end_date": str(group[date_col].max().date()),
            "num_docs": len(group),
        }
        logger.info("  %s: %d docs (%s to %s)", split_name, len(group),
                     manifest[split_name]["start_date"], manifest[split_name]["end_date"])

    return manifest


# ---------------------------------------------------------------------------
# CC-News -- quarterly splits
# ---------------------------------------------------------------------------

def split_cc_news(parquet_path: str | Path, output_dir: str | Path,
                  date_col: str = "date") -> dict:
    """Split CC-News by quarter. Returns manifest dict."""
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CC-News from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[date_col])
    logger.info("Dropped %d rows with invalid dates (%d remaining)", n_before - len(df), len(df))

    df["_quarter"] = df[date_col].dt.to_period("Q").astype(str)  # "2017Q1"
    # Normalize format to "2017-Q1"
    df["_quarter"] = df["_quarter"].str.replace(r"(\d{4})Q(\d)", r"\1-Q\2", regex=True)

    manifest = {}
    for period, group in sorted(df.groupby("_quarter")):
        split_name = period  # e.g. "2017-Q1"
        split_path = output_dir / f"{split_name}.parquet"
        group.drop(columns=["_quarter"]).to_parquet(split_path, index=False)
        manifest[split_name] = {
            "parquet_path": str(split_path),
            "start_date": str(group[date_col].min().date()),
            "end_date": str(group[date_col].max().date()),
            "num_docs": len(group),
        }
        logger.info("  %s: %d docs (%s to %s)", split_name, len(group),
                     manifest[split_name]["start_date"], manifest[split_name]["end_date"])

    return manifest


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_temporal_splits(config_path: str | Path) -> dict:
    """Read config, create splits for all configured datasets, save manifests."""
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    all_manifests = {}

    for dataset_name, ds_cfg in data_cfg.items():
        parquet_path = PROJECT_ROOT / ds_cfg["parquet_path"]
        if not parquet_path.exists():
            logger.warning("Skipping %s: %s not found", dataset_name, parquet_path)
            continue

        granularity = ds_cfg.get("split_granularity", "month")
        date_col = ds_cfg.get("date_col", "date")
        output_dir = PROJECT_ROOT / "data" / "splits" / dataset_name

        logger.info("=== Splitting %s (%s) ===", dataset_name, granularity)
        if granularity == "month":
            manifest = split_bbc_news(parquet_path, output_dir, date_col=date_col)
        elif granularity == "quarter":
            manifest = split_cc_news(parquet_path, output_dir, date_col=date_col)
        else:
            raise ValueError(f"Unknown split_granularity: {granularity}")

        # Save manifest JSON alongside splits
        manifest_path = output_dir / "splits_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved manifest to %s (%d splits)", manifest_path, len(manifest))

        all_manifests[dataset_name] = {
            "manifest_path": str(manifest_path),
            "splits": manifest,
        }

        # Print summary
        total_docs = sum(s["num_docs"] for s in manifest.values())
        print(f"\n{'='*60}")
        print(f"  {dataset_name} -- {len(manifest)} temporal splits, {total_docs:,} docs total")
        print(f"{'='*60}")
        for name, info in sorted(manifest.items()):
            print(f"  {name}: {info['num_docs']:>8,} docs  ({info['start_date']} to {info['end_date']})")
        print()

    return all_manifests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Split datasets into temporal periods")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    create_temporal_splits(args.config)


if __name__ == "__main__":
    main()
