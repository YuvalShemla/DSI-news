"""
Batch query generation for BBC 25K dataset using Gemini 2.5 Flash.

Generates 10 training queries + 1 evaluation query per document.
Uses async concurrency for fast throughput. Supports checkpointing.

Usage:
    python src/data/generate_queries_batch.py --dry-run          # test 5 docs
    python src/data/generate_queries_batch.py                     # full run
    python src/data/generate_queries_batch.py --resume            # resume
    python src/data/generate_queries_batch.py --merge-only        # merge checkpoints
    python src/data/generate_queries_batch.py --build-final       # build final CSV
"""

import argparse
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
METADATA_PATH = DATA_DIR / "queries" / "bbc_25k_metadata.csv"
CHECKPOINT_DIR = DATA_DIR / "queries" / "checkpoints"
OUTPUT_PATH = DATA_DIR / "queries" / "bbc_25k_queries.csv"
FINAL_DATASET_PATH = DATA_DIR / "queries" / "bbc_25k_dataset.csv"

GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_CONCURRENT = 40  # concurrent API requests (80 hits rate limits)
MAX_RETRIES = 3
RETRY_DELAY = 2
CHECKPOINT_EVERY = 500  # docs between checkpoints
MAX_CONTENT_CHARS = 3000

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRAINING_QUERIES_PROMPT = """\
Given this news article, generate exactly 10 diverse search queries that would lead \
a user to find this article. Follow this EXACT format (one query per line, with the label prefix):

KEYWORD_1: [3-8 word keyword query]
KEYWORD_2: [3-8 word keyword query]
QUESTION_1: [factual question this article answers, 8-20 words]
QUESTION_2: [factual question this article answers, 8-20 words]
QUESTION_3: [factual question this article answers, 8-20 words]
TOPICAL_1: [broader topic query, 5-12 words]
TOPICAL_2: [broader topic query, 5-12 words]
BACKGROUND_1: [query a journalist would use for background research, 6-15 words]
BACKGROUND_2: [query a journalist would use for background research, 6-15 words]
DETAIL_1: [query about a specific fact, name, or number in the article, 5-15 words]

Rules:
- Do NOT copy phrases verbatim from the article
- Each query must be unique and substantially different from the others
- Output ONLY the 10 labeled lines, nothing else

Article title: {title}
Article text (excerpt): {content}
"""

EVAL_QUERY_PROMPT = """\
Read this news article. Write ONE factual question that:
1. Can be specifically answered by this article
2. Is specific enough that very few other articles could answer it
3. Is a natural question someone might type into a search engine
4. Is 8-15 words long

Article title: {title}
Article text (excerpt): {content}

Write ONLY the question (no quotes, no labels, no explanation):
"""


def truncate_content(content: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."


def parse_training_queries(response: str) -> dict[str, str]:
    """Parse the structured 10-query response."""
    queries = {}
    expected_labels = [
        "KEYWORD_1", "KEYWORD_2",
        "QUESTION_1", "QUESTION_2", "QUESTION_3",
        "TOPICAL_1", "TOPICAL_2",
        "BACKGROUND_1", "BACKGROUND_2",
        "DETAIL_1",
    ]
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for label in expected_labels:
            if line.upper().startswith(label + ":"):
                query_text = line[len(label) + 1:].strip().strip('"').strip("'")
                if query_text and 3 <= len(query_text.split()) <= 30:
                    queries[label] = query_text
                break
    return queries


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)


async def call_gemini_async(client: genai.Client, prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Async Gemini call with concurrency control."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            _executor,
            lambda: client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        )
        return response.text.strip()


async def generate_for_doc_async(
    client: genai.Client,
    doc: dict,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Generate all queries for a single document (async)."""
    title = doc["title"]
    content = truncate_content(doc["content"])
    doc_id = doc["doc_id"]
    task_id = doc["task_id"]
    year_month = doc["year_month"]
    phase = doc["phase"]

    results = []

    # Training queries
    prompt = TRAINING_QUERIES_PROMPT.format(title=title, content=content)
    for attempt in range(MAX_RETRIES):
        try:
            response = await call_gemini_async(client, prompt, semaphore)
            queries = parse_training_queries(response)
            if len(queries) >= 7:
                for qtype, qtext in queries.items():
                    results.append({
                        "doc_id": doc_id,
                        "task_id": task_id,
                        "year_month": year_month,
                        "phase": phase,
                        "query": qtext,
                        "query_type": qtype,
                        "split": "train",
                    })
                break
            else:
                logger.debug("Doc %s: parsed %d/10, retrying...", doc_id, len(queries))
        except Exception as e:
            logger.debug("Doc %s train failed (attempt %d): %s", doc_id, attempt + 1, str(e)[:100])
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    # Eval query
    eval_prompt = EVAL_QUERY_PROMPT.format(title=title, content=content)
    for attempt in range(MAX_RETRIES):
        try:
            eval_query = await call_gemini_async(client, eval_prompt, semaphore)
            eval_query = eval_query.strip().strip('"').strip("'")
            if eval_query and 4 <= len(eval_query.split()) <= 25:
                results.append({
                    "doc_id": doc_id,
                    "task_id": task_id,
                    "year_month": year_month,
                    "phase": phase,
                    "query": eval_query,
                    "query_type": "EVAL",
                    "split": "eval",
                })
                break
        except Exception as e:
            logger.debug("Doc %s eval failed (attempt %d): %s", doc_id, attempt + 1, str(e)[:100])
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    return results


async def process_batch(
    client: genai.Client,
    docs: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Process a batch of documents concurrently."""
    tasks = [generate_for_doc_async(client, doc, semaphore) for doc in docs]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    failed = 0
    for doc, result in zip(docs, results_list):
        if isinstance(result, Exception):
            logger.warning("Doc %s failed entirely: %s", doc["doc_id"], result)
            failed += 1
        elif not result:
            failed += 1
        else:
            all_results.extend(result)

    return all_results, failed


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> set[str]:
    done = set()
    if not CHECKPOINT_DIR.exists():
        return done
    for f in CHECKPOINT_DIR.glob("checkpoint_*.csv"):
        try:
            df = pd.read_csv(f)
            done.update(df["doc_id"].unique())
        except Exception:
            pass
    logger.info("Loaded %d already-processed docs from checkpoints", len(done))
    return done


def save_checkpoint(results: list[dict], batch_num: int):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"checkpoint_{batch_num:04d}.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    logger.info("Checkpoint %d: %d query rows saved", batch_num, len(results))


def merge_checkpoints():
    all_dfs = []
    for f in sorted(CHECKPOINT_DIR.glob("checkpoint_*.csv")):
        try:
            all_dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e)

    if not all_dfs:
        logger.error("No checkpoint files found!")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["doc_id", "query_type"])
    merged.to_csv(OUTPUT_PATH, index=False)
    logger.info("Merged %d query rows (%d unique docs) to %s",
                len(merged), merged["doc_id"].nunique(), OUTPUT_PATH)
    return merged


def build_final_dataset():
    queries = pd.read_csv(OUTPUT_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    dataset = queries.merge(
        metadata[["doc_id", "title", "content", "section", "link", "date"]],
        on="doc_id",
        how="left",
    )
    dataset.to_csv(FINAL_DATASET_PATH, index=False)

    logger.info("\n=== Final Dataset Summary ===")
    logger.info("Total rows: %d", len(dataset))
    logger.info("Unique docs: %d", dataset["doc_id"].nunique())
    logger.info("Split counts: %s", dataset["split"].value_counts().to_dict())
    logger.info("Query type counts:\n%s", dataset["query_type"].value_counts().to_string())

    total_meta = metadata["doc_id"].nunique()
    docs_eval = dataset[dataset["split"] == "eval"]["doc_id"].nunique()
    docs_train = dataset[dataset["split"] == "train"]["doc_id"].nunique()
    logger.info("Coverage - eval: %d/%d (%.1f%%), train: %d/%d (%.1f%%)",
                docs_eval, total_meta, 100 * docs_eval / total_meta,
                docs_train, total_meta, 100 * docs_train / total_meta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_generation(metadata: pd.DataFrame, resume: bool = False):
    """Main async generation loop."""
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    done_ids = load_checkpoint() if resume else set()
    docs = [row.to_dict() for _, row in metadata.iterrows() if row["doc_id"] not in done_ids]
    logger.info("Documents to process: %d (skipped %d already done)", len(docs), len(done_ids))

    batch_num = len(list(CHECKPOINT_DIR.glob("checkpoint_*.csv"))) if CHECKPOINT_DIR.exists() else 0
    total_processed = 0
    total_failed = 0
    batch_results = []
    start_time = time.time()

    # Process in batches for checkpointing
    batch_size = CHECKPOINT_EVERY
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        results, failed = await process_batch(client, batch_docs, semaphore)

        batch_results.extend(results)
        total_processed += len(batch_docs)
        total_failed += failed

        # Checkpoint
        if batch_results:
            save_checkpoint(batch_results, batch_num)
            batch_num += 1
            batch_results = []

        elapsed = time.time() - start_time
        rate = total_processed / elapsed * 60 if elapsed > 0 else 0
        remaining = (len(docs) - total_processed) / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d docs (%.0f/min), %d failed, ETA: %.1f min",
            total_processed, len(docs), rate, total_failed, remaining,
        )

    # Save any remaining
    if batch_results:
        save_checkpoint(batch_results, batch_num)

    logger.info("=== Generation complete: %d processed, %d failed ===", total_processed, total_failed)


def main():
    parser = argparse.ArgumentParser(description="Generate queries for BBC 25K dataset")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--build-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 docs")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    if args.merge_only:
        merge_checkpoints()
        return

    if args.build_final:
        build_final_dataset()
        return

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set")
        return

    metadata = pd.read_csv(METADATA_PATH)
    logger.info("Loaded %d documents", len(metadata))

    if args.end:
        metadata = metadata.iloc[args.start:args.end]
    elif args.start > 0:
        metadata = metadata.iloc[args.start:]

    if args.dry_run:
        metadata = metadata.head(5)
        logger.info("DRY RUN: %d docs", len(metadata))

    asyncio.run(run_generation(metadata, resume=args.resume))

    # Merge and build
    merge_checkpoints()
    if OUTPUT_PATH.exists():
        build_final_dataset()


if __name__ == "__main__":
    main()
