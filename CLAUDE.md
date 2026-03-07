# DSI-CL Project Instructions

## Project Context

This is a research project on **Positive Forgetting in Differentiable Search Indices (DSI)** using continual learning. The core idea: in temporal corpora, forgetting old documents is beneficial when they're replaced by newer, more relevant ones.

**Base framework:** MixLoRA-DSI (Mixture-of-LoRA Experts for rehearsal-free continual learning)
**Key intervention:** Chrono-Semantic DocIDs -- `[Year]-[Month]-[Semantic_Cluster_ID]` format to bias the decoder toward recent documents
**Evaluation datasets:** CC-News (708K articles, 2017-2019) + BBC News AllTime (~160K+ articles, 2017-2025)

## Repository Layout

- `proposal.ltx` -- LaTeX research proposal (the authoritative spec)
- `docs/` -- Literature reviews and surveys
- `data/cc-news/` -- CC-News dataset (Parquet, downloaded from HuggingFace)
- `data/bbc-news/` -- BBC News AllTime dataset (Parquet, downloaded from HuggingFace)
- `src/data/` -- Data download and processing scripts
- `src/` -- Source code
- `experiments/` -- Experiment configs and results

## Dataset Details

### CC-News
- Source: `vblagoje/cc_news` on HuggingFace
- ~708K articles from Common Crawl news archives (2017-2019)
- Fields: title, text, domain, date, description, url
- Broad domain coverage across English-language news sources

### BBC News AllTime
- Source: `RealTimeData/bbc_news_alltime` on HuggingFace
- ~160K+ articles from BBC News (2017-2025)
- Fields: title, content, description, published_date, authors, section, link
- Monthly partitions -- ideal for temporal continual learning tasks

### Query Generation (LLM Pipeline)
Queries and relevance labels are generated via an LLM labeling pipeline rather than using pre-existing TREC judgments. The pipeline generates background linking queries and multi-level relevance judgments for sampled articles.

## Key Metrics

- **Positive Forgetting Rate (PFR):** Our novel metric -- see `proposal.ltx` Section 3.2 for the formal definition
- **Standard IR:** nDCG@5, MAP, Recall@K
- **Baselines:** FIFO Expert Pruning, Least Recently Activated (LRA) Expert Pruning

## Conventions

- Python for all experiments
- LaTeX for the proposal and final report
- Keep large data files out of git (use `.gitignore`)
