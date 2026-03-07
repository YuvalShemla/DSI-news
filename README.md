# DSI-CL: Exploiting Positive Forgetting in Differentiable Search Indices via Temporal Document Representations

**Team:** Vedik Agarwal, Kirthana Natarajan, Yuval Shemla, Yiyao Zhang

**Project Type:** Analysis, Experimentation and Extension of a Research Paper

## Overview

This project proposes that in dynamically evolving databases, forgetting is a feature, not a bug. We formalize and evaluate "Positive Forgetting" -- the phenomenon where a DSI's loss of a previously relevant document identifier (DocID) is actively beneficial because it is superseded by a newly learned, more recent, and equally or more relevant DocID.

We introduce **Chrono-Semantic DocIDs** -- temporal document representations designed to encourage recency bias within a **MixLoRA-DSI** continual learning framework.

## Repository Structure

```
DSI-CL/
├── proposal.ltx                   # LaTeX research proposal
├── requirements.txt               # Python dependencies
├── docs/                          # Literature reviews and surveys
│   ├── continual_learning_literature_review.md
│   ├── dsi_generative_retrieval_survey.md
│   ├── generative_text_retrieval_survey.md
│   └── literature_review_beneficial_forgetting.md
├── data/
│   ├── cc-news/                   # CC-News dataset (708K articles, 2017-2019)
│   └── bbc-news/                  # BBC News AllTime (~160K+ articles, 2017-2025)
├── src/
│   └── data/
│       └── download_datasets.py   # Dataset download script
└── experiments/                   # Experiment configs and results (TBD)
```

## Datasets

### CC-News (708K articles, 2017-2019)

From [vblagoje/cc_news](https://huggingface.co/datasets/vblagoje/cc_news). A large-scale news dataset from Common Crawl news archives with broad domain coverage.

**Fields:** title, text, domain, date, description, url

### BBC News AllTime (~160K+ articles, 2017-2025)

From [RealTimeData/bbc_news_alltime](https://huggingface.co/datasets/RealTimeData/bbc_news_alltime). Comprehensive BBC News articles with monthly partitions -- ideal for temporal continual learning.

**Fields:** title, content, description, published_date, authors, section, link

### Query Generation

Queries and relevance labels will be generated via an **LLM labeling pipeline** (to be built). This replaces the TREC News Track evaluation setup.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (saves Parquet files to data/)
python src/data/download_datasets.py
```

## Key References

- **MixLoRA-DSI:** Huynh et al. (2025). *Dynamically Expandable Mixture-of-LoRA Experts for Rehearsal-Free Generative Retrieval over Dynamic Corpora.* EMNLP 2025.
- **DSI:** Tay et al. (2022). *Transformer Memory as a Differentiable Search Index.* NeurIPS 2022.
