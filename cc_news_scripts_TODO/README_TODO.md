# CC-News Scripts

This folder is reserved for CC-News scripts that adapt the shared data pipeline
to the continual-learning workflow used in MixLoRA-DSI.

The paper we build on uses a staged script flow:

1. `marginmse_pretraining.sh`
2. `train_mixloradsi_d0.sh`
3. `train_mixloradsi.sh`

What should go here:

- Scripts to turn `data/cc-news/cc_news.parquet` into chronological tasks for
  continual training.
- Preprocessing helpers for CC-News fields such as `date`, `domain`, `title`,
  `description`, and `url`.
- Deduplication, filtering, or domain-quality controls, since CC-News is more
  heterogeneous and noisy than BBC News.
- Builders for pseudo-query training sets, `query_to_docid` maps, and
  semantic or chrono-semantic token IDs.
- Shell launchers or Python wrappers for CC-News-specific baseline and main
  experiments.


