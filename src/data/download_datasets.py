"""Download CC-News and BBC News AllTime datasets from HuggingFace."""

import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def download_cc_news():
    """Download CC-News dataset (708K articles, 2017-2019) and save as Parquet."""
    out_dir = DATA_DIR / "cc-news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cc_news.parquet"

    if out_path.exists():
        logger.info("CC-News already downloaded at %s, skipping.", out_path)
        return out_path

    logger.info("Downloading CC-News from HuggingFace (vblagoje/cc_news)...")
    ds = load_dataset("vblagoje/cc_news", split="train")

    logger.info("Converting to DataFrame and saving as Parquet...")
    df = ds.to_pandas()
    df.to_parquet(out_path, index=False)

    logger.info("Saved CC-News to %s", out_path)
    print_summary("CC-News", df, date_col="date")
    return out_path


def download_bbc_news():
    """Download BBC News AllTime dataset (all monthly configs) and save as Parquet."""
    out_dir = DATA_DIR / "bbc-news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bbc_news_alltime.parquet"

    if out_path.exists():
        logger.info("BBC News already downloaded at %s, skipping.", out_path)
        return out_path

    logger.info("Downloading BBC News AllTime from HuggingFace (RealTimeData/bbc_news_alltime)...")
    repo = "RealTimeData/bbc_news_alltime"
    configs = sorted(get_dataset_config_names(repo))
    logger.info("Found %d monthly configs: %s ... %s", len(configs), configs[0], configs[-1])

    dfs = []
    for i, config in enumerate(configs, 1):
        logger.info("  [%d/%d] Downloading config %s...", i, len(configs), config)
        ds = load_dataset(repo, config, split="train")
        dfs.append(ds.to_pandas())

    logger.info("Concatenating all months...")
    df = pd.concat(dfs, ignore_index=True)

    # Normalize authors column -- mixed types (lists, strings, None) cause Arrow errors
    if "authors" in df.columns:
        df["authors"] = df["authors"].apply(
            lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else (str(x) if pd.notna(x) else "")
        )

    logger.info("Saving as Parquet...")
    df.to_parquet(out_path, index=False)

    logger.info("Saved BBC News to %s", out_path)
    print_summary("BBC News AllTime", df, date_col="published_date")
    return out_path


def print_summary(name: str, df: pd.DataFrame, date_col: str):
    """Print summary statistics for a downloaded dataset."""
    print(f"\n{'='*60}")
    print(f"  {name} Summary")
    print(f"{'='*60}")
    print(f"  Total articles: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        valid = dates.dropna()
        if len(valid) > 0:
            print(f"  Date range: {valid.min()} to {valid.max()}")
            print(f"  Articles with valid dates: {len(valid):,} / {len(df):,}")
    print(f"{'='*60}\n")


def main():
    logger.info("Data directory: %s", DATA_DIR)

    try:
        download_cc_news()
    except Exception:
        logger.exception("Failed to download CC-News")
        sys.exit(1)

    try:
        download_bbc_news()
    except Exception:
        logger.exception("Failed to download BBC News AllTime")
        sys.exit(1)

    logger.info("All datasets downloaded successfully.")


if __name__ == "__main__":
    main()
