"""
Prepare the BBC News 25K dataset for DSI-CL experiments.

Loads BBC News AllTime parquet, deduplicates, filters, selects
Jan 2023 - Jul 2024 (19 months), assigns doc_ids and task_ids.

Usage:
    python src/data/prepare_bbc_25k.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BBC_PARQUET = DATA_DIR / "bbc-news" / "bbc_news_alltime.parquet"
OUTPUT_PATH = DATA_DIR / "queries" / "bbc_25k_metadata.csv"

# Training months: Jan 2023 - Apr 2024 (T0-T15)
# CL months: May 2024 - Jul 2024 (T16-T18)
TRAIN_MONTHS = pd.period_range("2023-01", "2024-04", freq="M")
CL_MONTHS = pd.period_range("2024-05", "2024-07", freq="M")
ALL_MONTHS = TRAIN_MONTHS.append(CL_MONTHS)

MIN_CONTENT_LEN = 200
MAX_CONTENT_WORDS = 2000


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load BBC parquet, deduplicate, filter."""
    logger.info("Loading BBC dataset from %s", path)
    df = pd.read_parquet(path)
    logger.info("Raw articles: %d", len(df))

    # Parse dates
    df["date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Dedup by link (URL)
    before = len(df)
    df = df.drop_duplicates(subset="link")
    logger.info("After dedup by link: %d (removed %d)", len(df), before - len(df))

    # Filter short/empty content
    df = df[df["content"].str.len() >= MIN_CONTENT_LEN]
    logger.info("After filtering short content: %d", len(df))

    # Truncate long articles
    df["content"] = df["content"].apply(
        lambda x: " ".join(str(x).split()[:MAX_CONTENT_WORDS])
    )

    # Additional dedup: same title within same month (near-duplicates)
    df["year_month"] = df["date"].dt.to_period("M")
    before = len(df)
    df = df.drop_duplicates(subset=["title", "year_month"])
    logger.info("After title+month dedup: %d (removed %d)", len(df), before - len(df))

    return df


def select_months(df: pd.DataFrame) -> pd.DataFrame:
    """Select target months and assign task IDs."""
    df = df[df["year_month"].isin(ALL_MONTHS)].copy()
    logger.info("After month selection (%d months): %d docs", len(ALL_MONTHS), len(df))

    # Sort by date within each month for stable ordering
    df = df.sort_values(["year_month", "date", "title"]).reset_index(drop=True)

    # Build task_id mapping
    task_map = {ym: i for i, ym in enumerate(ALL_MONTHS)}
    df["task_id"] = df["year_month"].map(task_map)

    # Assign doc_id: doc_T{task_id}_{seq:04d}
    doc_ids = []
    for task_id in sorted(df["task_id"].unique()):
        mask = df["task_id"] == task_id
        count = mask.sum()
        doc_ids.extend([f"doc_T{task_id}_{i:04d}" for i in range(count)])
    df["doc_id"] = doc_ids

    # Mark train vs CL
    train_task_ids = set(range(len(TRAIN_MONTHS)))
    df["phase"] = df["task_id"].apply(lambda t: "train" if t in train_task_ids else "cl")

    return df


def main():
    df = load_and_clean(BBC_PARQUET)
    df = select_months(df)

    # Print summary
    logger.info("\n=== Dataset Summary ===")
    for task_id in sorted(df["task_id"].unique()):
        subset = df[df["task_id"] == task_id]
        ym = subset["year_month"].iloc[0]
        phase = subset["phase"].iloc[0]
        logger.info("  T%d (%s) [%s]: %d docs", task_id, ym, phase, len(subset))

    train_count = len(df[df["phase"] == "train"])
    cl_count = len(df[df["phase"] == "cl"])
    logger.info("Training docs: %d", train_count)
    logger.info("CL docs: %d", cl_count)
    logger.info("Total: %d", len(df))

    # Save metadata
    out_cols = ["doc_id", "task_id", "year_month", "phase", "title", "content", "section", "link", "date"]
    df["year_month"] = df["year_month"].astype(str)
    df["date"] = df["date"].astype(str)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUTPUT_PATH, index=False)
    logger.info("Metadata saved to %s (%d rows)", OUTPUT_PATH, len(df))


if __name__ == "__main__":
    main()
