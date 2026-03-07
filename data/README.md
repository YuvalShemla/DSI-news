# Data Directory

## Datasets

### `cc-news/` -- CC-News (708K articles, 2017-2019)

Source: [vblagoje/cc_news](https://huggingface.co/datasets/vblagoje/cc_news) on HuggingFace.

A large-scale news dataset extracted from Common Crawl news archives. Articles span 2017-2019, covering a wide range of English-language news sources.

| Field | Description |
|-------|-------------|
| `title` | Article headline |
| `text` | Full article text |
| `domain` | Source domain (e.g., nytimes.com) |
| `date` | Publication date |
| `description` | Article summary/description |
| `url` | Original article URL |

**Size:** ~708K articles, stored as a single Parquet file (`cc_news.parquet`).

### `bbc-news/` -- BBC News AllTime (~160K+ articles, 2017-2025)

Source: [RealTimeData/bbc_news_alltime](https://huggingface.co/datasets/RealTimeData/bbc_news_alltime) on HuggingFace.

Comprehensive BBC News articles with monthly partitions, providing excellent temporal coverage for continual learning experiments.

| Field | Description |
|-------|-------------|
| `title` | Article headline |
| `content` | Full article text |
| `description` | Article summary |
| `published_date` | Publication date |
| `authors` | Article authors |
| `section` | News section (e.g., Business, Technology) |
| `link` | Original article URL |

**Size:** ~160K+ articles, stored as a single Parquet file (`bbc_news_alltime.parquet`).

## Download Instructions

```bash
pip install -r requirements.txt
python src/data/download_datasets.py
```

This downloads both datasets from HuggingFace and saves them as Parquet files in `data/cc-news/` and `data/bbc-news/`.

## Temporal Split Strategy

Both datasets are split into **monthly tasks** for continual learning:
- Each month's articles form one training task
- The model learns tasks sequentially in chronological order
- Evaluation measures retrieval performance across all seen documents, with emphasis on recency

## Query Generation

Queries are generated via an **LLM labeling pipeline** using Gemini (`src/data/generate_queries.py`).

**How it works:**
1. Each article is split into overlapping chunks (~400 tokens each)
2. Each chunk is sent to Gemini with a prompt asking for a realistic search query that the chunk answers
3. Output is saved as Parquet: `[doc_id, chunk_id, date, chunk_text, query]`

**Three prompt templates** are included (general, background-linking, question-style) -- see the script header for details and cost analysis.

```bash
# Test on a small sample first
python src/data/generate_queries.py --sample 100 --prompt v1

# Full run (requires GEMINI_API_KEY env var)
python src/data/generate_queries.py
```

**Estimated cost:** ~$187 for the full corpus using Gemini 2.0 Flash (paid tier), or free with rate-limited preview tier (~100 days at 15 RPM). See detailed cost breakdown in the script header.
