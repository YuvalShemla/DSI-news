# BBC News Scripts

This folder is reserved for BBC News AllTime scripts that are too
dataset-specific to live in the shared `src/data` pipeline.

The upstream MixLoRA-DSI codebase organizes its workflow as:

1. `marginmse_pretraining.sh`
2. `train_mixloradsi_d0.sh`
3. `train_mixloradsi.sh`

What should go here:

- Scripts to convert `data/bbc-news/bbc_news_alltime.parquet` into chronological
  training splits, likely monthly or quarterly tasks.
- Utilities to normalize BBC-specific fields such as `published_date`,
  `section`, `authors`, and `link`.
- Query-generation or query-cleaning helpers tailored to BBC article style.
- Builders for DSI training artifacts such as `query_to_docid`,
  `docid_to_tokenids`, or chrono-semantic DocID mappings.
- Training launchers that mirror the paper's sequence:
  pretraining, `d0` initialization, then continual training/evaluation.

