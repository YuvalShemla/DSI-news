"""
==============================================================================
TEMPLATE -- LLM-Based Query Generation for DSI-CL
==============================================================================
STATUS: Draft template. Prompts, chunking strategy, and output format all
        need iteration. Run on a small sample first, inspect outputs, then
        refine before scaling to the full corpus.

==============================================================================
COST ANALYSIS (Gemini models, as of March 2026)
==============================================================================

Corpus size:
  - CC-News:  ~708K articles  (~1,000 tokens avg)
  - BBC News: ~160K articles  (~800 tokens avg)
  - Total:    ~868K articles

Chunking assumptions:
  - Chunk size: ~400 tokens, overlap ~50 tokens
  - Avg chunks per article: ~2.5
  - Total chunks: ~2.17M
  - Prompt overhead per call: ~300 tokens
  - Output per query: ~40 tokens

Token totals:
  - Input:  2.17M calls x 700 tokens = ~1.52B input tokens
  - Output: 2.17M calls x  40 tokens = ~87M output tokens

Cost estimates (paid tier):
  ┌─────────────────────────┬────────────┬─────────────┬───────────┐
  │ Model                   │ Input cost │ Output cost │ Total     │
  ├─────────────────────────┼────────────┼─────────────┼───────────┤
  │ Gemini 2.0 Flash        │    $152    │     $35     │   ~$187   │
  │ Gemini 2.5 Flash-Lite   │    $152    │     $35     │   ~$187   │
  │ Gemini 2.5 Flash        │    $456    │    $218     │   ~$674   │
  │ Gemini 3.1 Flash-Lite   │    $380    │    $131     │   ~$511   │
  └─────────────────────────┴────────────┴─────────────┴───────────┘

Free tier option:
  Gemini 2.0 Flash and 2.5 Flash-Lite both offer UNLIMITED free tokens
  during preview, but with rate limits (~15 RPM free, ~1500 RPM paid).
  At 15 RPM: 2.17M calls / 15 = ~2,413 hours = ~100 days.
  At 1500 RPM (paid): ~24 hours. Batching helps significantly.

Recommendation:
  Use Gemini 2.0 Flash (cheapest, fast, good quality for query generation).
  Start with the free tier on a sample of ~1,000 articles to validate
  prompt quality, then switch to paid tier for the full run.

What to expect:
  - ~2.17M queries total (one per chunk)
  - Each query is a natural-language information need that the chunk answers
  - Output: Parquet file with columns [doc_id, chunk_id, chunk_text, query]
  - Quality will vary -- expect ~70-80% of queries to be usable out of the
    box. The rest may be too generic, too specific, or poorly formed.
    Post-filtering (length, dedup, LLM-as-judge) is recommended.
  - Full pipeline wall-clock time: ~24-48 hours (paid tier, single worker)

==============================================================================
"""

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from google import genai as google_genai
except ImportError as e:  # pragma: no cover - import guard for clearer errors
    google_genai = None
    _GENAI_IMPORT_ERROR = e
else:
    _GENAI_IMPORT_ERROR = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 400  # tokens (approximate, splitting by words as proxy)
CHUNK_OVERLAP = 50
WORDS_PER_TOKEN = 0.75  # rough approximation: 1 token ≈ 0.75 words
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between generate_query attempts on validation / API errors

# Transient 429 / quota blips (bounded retries, then fail)
_RATE_LIMIT_MAX_ATTEMPTS = 12
_RATE_LIMIT_BACKOFF_BASE = 2.0
_RATE_LIMIT_BACKOFF_CAP = 90.0

MAX_OUTPUT_TOKENS = 1000
TEMPERATURE = 0.5

# ---------------------------------------------------------------------------
# Prompts -- THESE ARE TEMPLATES, ITERATE ON THEM
# ---------------------------------------------------------------------------

QUERY_GENERATION_PROMPT = """\
You are generating search queries for an information retrieval dataset.

Given the following passage from a news article, write a single natural-language \
search query that a user might type into a search engine, where this passage would \
be a highly relevant result.

Requirements:
- The query should be a realistic information need (something a person would actually search)
- The query should be answerable by the passage content
- Do NOT copy phrases verbatim from the passage
- Keep the query concise (5-15 words)
- Do NOT include quotes or prefixes like "Query:" -- just output the query text

Passage:
{chunk_text}
"""

# Alternative prompt styles to experiment with:

QUERY_GENERATION_PROMPT_V2_BACKGROUND_LINKING = """\
You are a journalist researching background context for a story.

Read this passage from a news article and write a search query you would use \
to find this article as background material for a related story you're writing.

The query should reflect a broader topic or event that this passage provides \
useful context for -- not a query about the specific details in the passage.

Passage:
{chunk_text}

Write only the search query (5-15 words, no quotes or labels):
"""

QUERY_GENERATION_PROMPT_V3_QUESTION = """\
Read this passage from a news article. Write a factual question that this \
passage answers. The question should be natural and specific enough that this \
passage is clearly the best answer.

Passage:
{chunk_text}

Write only the question:
"""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word count (as a proxy for tokens).

    Args:
        text: The article text to chunk.
        chunk_size: Target chunk size in approximate tokens.
        overlap: Overlap between chunks in approximate tokens.

    Returns:
        List of chunk strings. Returns [text] if text is shorter than chunk_size.
    """
    if not text or not text.strip():
        return []

    # Convert token counts to word counts
    chunk_words = int(chunk_size * WORDS_PER_TOKEN)
    overlap_words = int(overlap * WORDS_PER_TOKEN)
    step = max(chunk_words - overlap_words, 1)

    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        chunks.append(chunk)
        if start + chunk_words >= len(words):
            break

    return chunks


# ---------------------------------------------------------------------------
# LLM client (Google Gemini via google-genai; see https://ai.google.dev/gemini-api/docs/migrate)
# ---------------------------------------------------------------------------

_gemini_client = None
_generate_content_config = None


try:
    from google.api_core import exceptions as google_api_exceptions
except ImportError:  # pragma: no cover
    google_api_exceptions = None


def _is_rate_limit_error(exc: BaseException) -> bool:
    if google_api_exceptions is not None and isinstance(exc, google_api_exceptions.ResourceExhausted):
        return True
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource exhausted" in msg
        or "too many requests" in msg
        or "rate limit" in msg
    )


def _ensure_gemini_available() -> None:
    if google_genai is None:
        raise RuntimeError(
            "The `google-genai` package is required. Install dependencies: "
            "`pip install -r requirements.txt`"
        ) from _GENAI_IMPORT_ERROR


def _get_generate_content_config():
    """Build config once; prefer types.GenerateContentConfig when available."""
    global _generate_content_config
    if _generate_content_config is not None:
        return _generate_content_config
    try:
        from google.genai import types as genai_types

        _generate_content_config = genai_types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception:
        _generate_content_config = {"max_output_tokens": 256, "temperature": TEMPERATURE}
    return _generate_content_config


def _configure_gemini() -> None:
    global _gemini_client
    if _gemini_client is not None:
        return
    _ensure_gemini_available()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export your key before running, e.g. "
            "`export GEMINI_API_KEY=...`"
        )
    _gemini_client = google_genai.Client(api_key=api_key)


def _extract_response_text(response) -> str:
    """Return plain text from a generate_content response, or empty string if unavailable."""
    try:
        text = getattr(response, "text", None)
        if text is not None:
            return text.strip()
    except ValueError:
        pass
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        prompt_fb = getattr(response, "prompt_feedback", None)
        logger.warning("Gemini returned no candidates (prompt_feedback=%s)", prompt_fb)
        return ""
    first = candidates[0]
    content = getattr(first, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    pieces = []
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            pieces.append(t)
    return "\n".join(pieces).strip()


def call_gemini(prompt: str, model: str = GEMINI_MODEL) -> str:
    """Call Gemini API; retry transient rate limits with bounded backoff."""
    _configure_gemini()
    config = _get_generate_content_config()

    for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
        try:
            response = _gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return _extract_response_text(response)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < _RATE_LIMIT_MAX_ATTEMPTS - 1:
                wait = min(
                    _RATE_LIMIT_BACKOFF_BASE * (2**attempt),
                    _RATE_LIMIT_BACKOFF_CAP,
                )
                logger.warning(
                    "Gemini rate limited (attempt %d/%d, sleep %.1fs): %s",
                    attempt + 1,
                    _RATE_LIMIT_MAX_ATTEMPTS,
                    wait,
                    e,
                )
                time.sleep(wait)
                continue
            raise


def _fill_prompt(template: str, chunk_text: str) -> str:
    """Insert chunk text without str.format (passages may contain `{` / `}`)."""
    return template.replace("{chunk_text}", chunk_text)


def generate_query(
    chunk_text: str,
    prompt_template: str = QUERY_GENERATION_PROMPT,
    model: str = GEMINI_MODEL,
) -> str | None:
    """Generate a single query for a text chunk with retries."""
    prompt = _fill_prompt(prompt_template, chunk_text)

    for attempt in range(MAX_RETRIES):
        try:
            query = call_gemini(prompt, model=model)
            # Basic validation
            if query and 3 < len(query.split()) < 30:
                return query
            logger.warning("Query failed validation (len=%d words), retrying...", len(query.split()) if query else 0)
        except Exception as e:
            logger.warning("API call failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return None


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    parquet_path: str | Path,
    text_col: str,
    id_col: str | None,
    date_col: str | None,
    output_path: str | Path,
    prompt_template: str = QUERY_GENERATION_PROMPT,
    sample_n: int | None = None,
    model: str = GEMINI_MODEL,
):
    """Process a dataset: chunk articles, generate queries, save results.

    Args:
        parquet_path: Path to input Parquet file.
        text_col: Column name containing article text.
        id_col: Column name for document ID (or None to use row index).
        date_col: Column name for publication date (carried through to output).
        output_path: Path to save output Parquet file and CSV
        prompt_template: The prompt template to use for query generation.
        sample_n: If set, only process this many articles (for testing).
        model: Gemini model id passed to the API.
    """
    logger.info("Loading dataset from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    missing_cols = [c for c in (text_col, id_col, date_col) if c and c not in df.columns]
    if missing_cols:
        raise ValueError(f"Parquet missing required columns: {missing_cols}")

    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
        logger.info("Sampled %d articles for testing", len(df))

    results = []
    total_chunks = 0
    failed = 0

    for idx, row in df.iterrows():
        if id_col and id_col in row.index:
            raw_id = row[id_col]
            doc_id = str(raw_id) if pd.notna(raw_id) else str(idx)
        else:
            doc_id = str(idx)

        raw_text = row.get(text_col, "")
        if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
            continue
        text = raw_text.strip() if isinstance(raw_text, str) else str(raw_text).strip()
        if len(text) < 50:
            continue

        date = None
        if date_col and date_col in row.index:
            date = row[date_col]
            if pd.isna(date):
                date = None

        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            total_chunks += 1
            query = generate_query(chunk, prompt_template, model=model)

            if query:
                results.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk{chunk_idx}",
                    "date": date,
                    "chunk_text": chunk,
                    "query": query,
                })
            else:
                failed += 1

            if total_chunks % 100 == 0:
                logger.info(
                    "Progress: %d chunks processed, %d queries generated, %d failed",
                    total_chunks, len(results), failed,
                )

    out_df = pd.DataFrame(results)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_p, index=False)
    csv_path = out_p.with_suffix(".csv")
    out_df.to_csv(csv_path, index=False)

    logger.info("Done. %d queries saved to %s and %s", len(out_df), out_p, csv_path)
    logger.info("Stats: %d chunks total, %d queries generated, %d failed (%.1f%% success)",
                total_chunks, len(results), failed,
                100 * len(results) / max(total_chunks, 1))

    return out_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run query generation on both datasets.

    Usage:
        # Test on small sample first:
        python src/data/generate_queries.py --sample 100

        # Full run:
        python src/data/generate_queries.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate queries for DSI-CL datasets")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process N articles (for testing prompts)")
    parser.add_argument("--dataset", choices=["cc-news", "bbc-news", "both"], default="both",
                        help="Which dataset to process")
    parser.add_argument("--prompt", choices=["v1", "v2", "v3"], default="v1",
                        help="Prompt template: v1=general, v2=background-linking, v3=question")
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL,
        help=f"Gemini model id (default: {GEMINI_MODEL})",
    )
    args = parser.parse_args()

    try:
        _ensure_gemini_available()
        if not os.environ.get("GEMINI_API_KEY"):
            logger.error(
                "GEMINI_API_KEY is not set. Set it before running, e.g. export GEMINI_API_KEY=..."
            )
            sys.exit(1)
        _configure_gemini()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    prompt_templates = {
        "v1": QUERY_GENERATION_PROMPT,
        "v2": QUERY_GENERATION_PROMPT_V2_BACKGROUND_LINKING,
        "v3": QUERY_GENERATION_PROMPT_V3_QUESTION,
    }
    prompt = prompt_templates[args.prompt]

    if args.dataset in ("cc-news", "both"):
        cc_news_path = DATA_DIR / "cc-news" / "cc_news.parquet"
        if cc_news_path.exists():
            logger.info("=== Processing CC-News ===")
            process_dataset(
                parquet_path=cc_news_path,
                text_col="text",
                id_col="url",  # use URL as doc ID (unique per article)
                date_col="date",
                output_path=DATA_DIR / "cc-news" / f"cc_news_queries_{args.prompt}.parquet",
                prompt_template=prompt,
                sample_n=args.sample,
                model=args.model,
            )
        else:
            logger.warning("CC-News not found at %s. Run download_datasets.py first.", cc_news_path)

    if args.dataset in ("bbc-news", "both"):
        bbc_path = DATA_DIR / "bbc-news" / "bbc_news_alltime.parquet"
        if bbc_path.exists():
            logger.info("=== Processing BBC News ===")
            process_dataset(
                parquet_path=bbc_path,
                text_col="content",
                id_col="link",  # use link as doc ID
                date_col="published_date",
                output_path=DATA_DIR / "bbc-news" / "bbc_news_queries.parquet",
                prompt_template=prompt,
                sample_n=args.sample,
                model=args.model,
            )
        else:
            logger.warning("BBC News not found at %s. Run download_datasets.py first.", bbc_path)


if __name__ == "__main__":
    main()
