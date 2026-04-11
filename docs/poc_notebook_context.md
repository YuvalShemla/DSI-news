# DSI-CL PoC Notebook — Context & Next Steps

## What We Built

We created a proof-of-concept Colab notebook (`experiments/poc_t5gemma_dsi.ipynb`) that demonstrates end-to-end DSI (Differentiable Search Index) using **T5Gemma 2** (google/t5gemma-2-270m-270m) with **PEFT LoRA** and **chrono-semantic DocIDs**.

### Repository
- GitHub: `https://github.com/YuvalShemla/DSI-news.git`
- The notebook clones this repo on Colab, uses data from `data/queries/poc_data.csv`

### Data
- **Source**: 3 LLM-generated query files from CC-News articles:
  - `cc_news_queries_v1 (2).csv` — keyword-style queries (avg 58 chars)
  - `cc_news_queries_v2 (1).csv` — topical/broad queries (avg 63 chars)
  - `cc_news_queries_v3 (1).csv` — factual questions (avg 93 chars)
- **Combined** into `data/queries/poc_data.csv`: 1,713 query-doc pairs, 572 chunks from 303 articles
- **Schema**: `doc_id, chunk_id, date, chunk_text, query, query_version`
- **Date range**: Jan 2017 – Jul 2018
- 49 rows with null dates were dropped during combination
- Each chunk has ~3 queries (one per version)
- `chunk_id` is the unique document identifier (not `doc_id`, which is the article-level URL)

---

## T5Gemma 2 Model Findings

### Config Structure
- **Model**: `google/t5gemma-2-270m-270m` (~0.8B params total)
- Requires **Gemma license** acceptance on HuggingFace + `HF_TOKEN`
- Requires `transformers>=4.51.0` (NOT >=5.0 as we initially assumed)
- **Encoder config** uses a DIFFERENT class (`T5Gemma2EncoderConfig`) than the decoder — it does NOT have a `hidden_size` attribute accessible via `getattr`. You must discover attributes dynamically by iterating `vars(enc_cfg)` or fall back to reading `model.get_input_embeddings().weight.shape[1]`.
- **Decoder config** has standard attributes: `hidden_size=640`, `num_hidden_layers=18`, `num_attention_heads=4`, `head_dim=256`
- **Embedding dim**: 640 (shared between encoder and decoder via `tie_word_embeddings=True`)
- **Vocab size**: 256,000 base → embedding matrix is 262,144 (padded). After adding special tokens it becomes 262,286.

### LoRA Target Modules
- Auto-discovery finds: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `out_proj`
- **CRITICAL**: `out_proj` is the lm_head output projection, NOT an attention layer. Including it in LoRA targets causes `RuntimeError: The size of tensor a (262286) must match the size of tensor b (262144)` because with `tie_word_embeddings=True` + vocab resize, the lm_head's LoRA adapter gets initialized with original vocab size.
- **Correct LoRA targets**: only `q_proj`, `k_proj`, `v_proj`, `o_proj` (use exact string matching, NOT `'proj' in name`)

### Generation / Beam Search Issues
1. **Deprecation warning**: Passing `generation_config` together with explicit generation args (`num_beams`, `max_new_tokens`, etc.) is deprecated. Fix: pass EITHER a `GenerationConfig` object OR explicit args, not both.
2. **`prefix_allowed_tokens_fn returned an empty list`**: The prefix trie was built with `tokenizer.pad_token_id` as the sequence start, but `model.generate()` uses `decoder_start_token_id` which may differ. Fix: use `model.config.decoder_start_token_id` (or fall back to `pad_token_id`) when building the trie AND in the collator's `decoder_input_ids`.
3. **NEEDS FIXING in the notebook** — these two bugs exist in the current pushed version.

---

## FAISS RQ Training

- **PCA required**: With 572 docs and 640-dim embeddings, `n_docs < embed_dim` causes FAISS RQ's internal PCA to fail (`PCA matrix cannot output 640 dimensions from 640`). Fix: PCA-reduce embeddings to `min(256, n_docs-1)` before RQ training.
- **Current PoC params**: 4 codebooks, nbits=5 (32 centroids each)
- RQ centroids are 256-dim (after PCA), projected to 640-dim via random projection for embedding initialization

---

## Training Results & Key Insight

### Without masked loss (standard CE over full vocab)
Best result after 50 epochs (LoRA r=16, alpha=32, lr=1e-3):
```
Epoch 1:  25.18
Epoch 50: 2.23  (still high — not converging)
```
**Problem**: At each DocID position, CE loss competes against all 262K vocab entries when only 2-32 tokens are actually valid. The model wastes capacity suppressing irrelevant tokens.

### With masked-logit loss (IMPLEMENTED but not yet fully tested)
Masks logits to only valid tokens at each position before CE:
- Position 0 (year): 2 valid tokens → 2-class problem
- Position 1 (month): 12 valid tokens → 12-class problem
- Positions 2-5 (RQ): 32 valid tokens each → 32-class problem

This is implemented in the current notebook (cell 28) but hasn't been tested to completion yet.

---

## Current Notebook State (what works, what's broken)

### Working cells (tested on Colab A100):
- Setup, clone, install, HF login
- Data loading and visualization
- Model loading and architecture inspection
- Document encoding (T5Gemma encoder, mean pooling)
- t-SNE visualization
- FAISS RQ training (with PCA fix)
- DocID construction (chrono-semantic)
- Tokenizer extension + embedding initialization
- Prefix trie construction
- Dataset/Collator creation
- LoRA application
- Training loop (both standard CE and masked-logit versions)

### Broken / needs fixing:
- **Evaluation cells (beam search)**: Two bugs — generation_config deprecation + prefix trie start token mismatch
- **No train/test split**: Currently train=test (pure memorization)

---

## NEXT GOAL: Restructure the Notebook

The notebook needs to be restructured with this new flow:

### 1. Two-pass experiment design
**Experiment A — Pure RQ DocIDs**: Train with ONLY semantic RQ codes: `[rq_0, rq_1, rq_2, rq_3]` (4 tokens). No year/month prefix. This is the baseline.

**Experiment B — Chrono-Semantic DocIDs**: Train with temporal prefix + RQ codes: `[year, month, rq_0, rq_1, rq_2, rq_3]` (6 tokens). This is our intervention.

Both experiments use the SAME RQ codes, SAME documents, SAME model architecture. The only difference is the DocID format.

### 2. Add a test set
Split queries into train/test. Suggested approach: train on `v1_keyword` + `v2_topical` queries, test on `v3_factual` queries. This tests query-style generalization (all 572 chunks are in the corpus for both train and test, but test queries are a different style the model hasn't seen).

Alternatively: hold out ~100 random chunks entirely (but then the model CAN'T retrieve them — only useful for measuring false positives).

### 3. Make training converge better
- Use the **masked-logit loss** (already implemented)
- Higher LoRA rank if needed (r=16 or r=32)
- Enough epochs to get loss < 0.5
- Learning rate with warmup + cosine

### 4. Fix evaluation
- Remove `generation_config` parameter from `model.generate()`, use explicit args only
- Fix prefix trie to use `model.config.decoder_start_token_id` instead of `tokenizer.pad_token_id`
- Make prefixer_fn robust (return all valid tokens for position if trie lookup fails)

### 5. Comparison section
After both experiments run:
- Side-by-side metrics table (MRR, Hits@1/5/10) on train AND test
- Loss curve comparison
- Per-query-style breakdown
- Visualizations

### 6. Interactive query cell
Query function using the best model (whichever won the comparison).

---

## File Map

```
experiments/poc_t5gemma_dsi.ipynb  — THE NOTEBOOK (needs restructuring)
data/queries/poc_data.csv          — Combined 1713 queries, 572 chunks
data/queries/cc_news_queries_v1 (2).csv  — Keyword queries
data/queries/cc_news_queries_v2 (1).csv  — Topical queries
data/queries/cc_news_queries_v3 (1).csv  — Factual questions
configs/default.yaml               — Experiment config (for full-scale runs)
src/model/backbone.py              — Model loading + LoRA
src/model/docid_tokenizer.py       — Tokenizer extension + embedding init
src/model/constrained_decoding.py  — Prefixer, FilteredPrefixer, parse_filter_from_query
src/training/dataset.py            — ChronoDocIDDataset, collator
src/training/train_d0.py           — D0 training script
src/training/train_cl.py           — Continual learning training
src/training/replay_buffer.py      — Replay buffer
src/training/lora_merging.py       — LoRA merge/compose/prune
src/evaluation/evaluate.py         — Beam search eval
src/evaluation/metrics.py          — MRR, nDCG, Recall, MAP
src/evaluation/pfr_metric.py       — Positive Forgetting Rate
src/evaluation/filtered_retrieval.py — Filtered retrieval eval
```

## Key Training Parameters (best so far)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | google/t5gemma-2-270m-270m | 0.8B params, encoder-decoder |
| LoRA r | 16 | Could try 32 |
| LoRA alpha | 32 | 2x rank |
| LoRA targets | q_proj, k_proj, v_proj, o_proj | NOT out_proj |
| Learning rate | 2e-3 | With warmup |
| Warmup steps | 200 | |
| Epochs | 30-50 | With masked loss, 30 may suffice |
| Batch size | 16 | |
| Max query len | 96 | For factual questions |
| RQ codebooks | 4 | (6 for full scale) |
| RQ nbits | 5 (32 centroids) | (8→256 for full scale) |
| Loss | Masked-logit CE | Only valid tokens per position |
| Num beams | 50 | For eval |
