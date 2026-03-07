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

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 400  # tokens (approximate, splitting by words as proxy)
CHUNK_OVERLAP = 50
WORDS_PER_TOKEN = 0.75  # rough approximation: 1 token ≈ 0.75 words
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

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
# LLM client -- placeholder, replace with your Gemini client
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str = GEMINI_MODEL) -> str:
    """Call Gemini API to generate a query from a prompt.

    TODO: Replace this placeholder with actual Gemini API call.

    Option A -- google-generativeai SDK:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text.strip()

    Option B -- REST API with requests:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        response = requests.post(url, params={"key": os.environ["GEMINI_API_KEY"]},
                                 json={"contents": [{"parts": [{"text": prompt}]}]})
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    Option C -- litellm (unified interface):
        from litellm import completion
        response = completion(model=f"gemini/{model}", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content.strip()
    """
    raise NotImplementedError(
        "Replace this with your Gemini API call. See docstring for examples. "
        "Set GEMINI_API_KEY in your environment."
    )


def generate_query(chunk_text: str, prompt_template: str = QUERY_GENERATION_PROMPT) -> str | None:
    """Generate a single query for a text chunk with retries."""
    prompt = prompt_template.format(chunk_text=chunk_text)

    for attempt in range(MAX_RETRIES):
        try:
            query = call_gemini(prompt)
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
):
    """Process a dataset: chunk articles, generate queries, save results.

    Args:
        parquet_path: Path to input Parquet file.
        text_col: Column name containing article text.
        id_col: Column name for document ID (or None to use row index).
        date_col: Column name for publication date (carried through to output).
        output_path: Path to save output Parquet file.
        prompt_template: The prompt template to use for query generation.
        sample_n: If set, only process this many articles (for testing).
    """
    logger.info("Loading dataset from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
        logger.info("Sampled %d articles for testing", len(df))

    results = []
    total_chunks = 0
    failed = 0

    for idx, row in df.iterrows():
        doc_id = row[id_col] if id_col and id_col in row.index else str(idx)
        text = row.get(text_col, "")
        date = row.get(date_col, None) if date_col else None

        if not text or not isinstance(text, str) or len(text.strip()) < 50:
            continue

        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            total_chunks += 1
            query = generate_query(chunk, prompt_template)

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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    logger.info("Done. %d queries saved to %s", len(out_df), output_path)
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
    args = parser.parse_args()

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
                output_path=DATA_DIR / "cc-news" / "cc_news_queries.parquet",
                prompt_template=prompt,
                sample_n=args.sample,
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
            )
        else:
            logger.warning("BBC News not found at %s. Run download_datasets.py first.", bbc_path)


if __name__ == "__main__":
    main()
