# Task 2: Upgrade Notebook to 25K Docs + Continual Learning Experiments

## Goal
Upgrade the V9 PoC notebook to work with the 25K BBC dataset, add visualizations, run the 3-way DocID comparison at scale, and implement continual learning experiments to measure forgetting.

---

## Part A: Dataset Visualization Section

Add a new section (after data loading) with:

1. **Dataset overview card:**
   - Source: BBC News AllTime
   - Date range: Jan 2023 - Jul 2024
   - Total docs: ~25K (20K train + 5K CL)
   - Number of months/tasks: 19

2. **Monthly distribution bar chart:**
   - X-axis: months (Jan 2023 - Jul 2024)
   - Y-axis: document count
   - Color-coded: blue for training months (T0-T15), orange for CL months (T16-T18)

3. **Sample documents table:**
   - Show 5-10 sample articles with title, date, section, content preview (first 200 chars)
   - For each, show its 10 training queries and 1 eval query

4. **Query style distribution:**
   - Bar chart showing counts by query_type (KEYWORD, QUESTION, TOPICAL, BACKGROUND, DETAIL, EVAL)
   - Example queries for each type

---

## Part B: Scale Up RQ to Handle 20K Documents

### Current PoC Settings
- RQ_NBITS = 5 (32 centroids per codebook)
- NUM_RQ_CODEBOOKS = 4
- ID space: 32^4 = 1,048,576 unique IDs
- 572 docs → 0 collisions

### Assessment for 20K docs
- 20K docs in 1M ID space = 1.9% occupancy → very low collision probability
- **Verdict: current RQ settings should work fine for 20K docs**
- But we should verify empirically and report collision count

### If collisions are too high
- Option A: Increase RQ_NBITS to 6 (64 centroids) → 64^4 = 16.7M IDs
- Option B: Increase NUM_RQ_CODEBOOKS to 5 → 32^5 = 33.5M IDs
- Prefer Option A (shorter DocIDs are better for training)

### Action Items
1. After FAISS RQ encoding, count and print collision rate
2. If collision rate > 1%, increase RQ_NBITS
3. Display collision analysis in the notebook

---

## Part C: Run 3-Way Comparison at 20K Scale

Train all three experiments on the full 20K training set:

### Experiment A: Atomic DocIDs
- Random unique codes per document
- DocID length: 4 tokens
- Tokens: `<atomic_CB_CODE>` (128 tokens for 32 centroids x 4 codebooks)

### Experiment B: Semantic RQ DocIDs
- FAISS Residual Quantization codes
- DocID length: 4 tokens
- Tokens: `<rq_CB_CODE>`

### Experiment C: Chrono-Semantic DocIDs
- Task prefix + RQ codes
- DocID length: 5 tokens (1 task prefix + 4 RQ codes)
- Tokens: `<task_N>` (19 task tokens) + `<rq_CB_CODE>`
- Task ID is the month ordinal (0-18)

### Training Configuration (scale up from PoC)
- Same model: google/t5gemma-2-270m-270m
- LoRA r=16, alpha=32
- Batch size: 32 (or 64 if GPU memory allows)
- Learning rate: 1e-3 with cosine decay
- Epochs: 50-100 with early stopping (loss < 0.01)
- Train on: all 10 training query styles
- Evaluate on: eval queries only

### Evaluation Metrics
- MRR, Hits@1, Hits@5, Hits@10
- Per-month breakdown (to see if some months are harder)
- Collision analysis per DocID scheme

---

## Part D: Continual Learning Experiments

### Overview
After training on 20K docs (T0-T15), introduce CL months (T16-T18) incrementally and measure forgetting.

### CL Strategy: LoRA Expansion

**Approach:** When learning new tasks, add new LoRA adapters while keeping old ones frozen. This is inspired by MixLoRA-DSI from the proposal.

**Option 1: Shared LoRA (Fine-tune existing)**
- Continue training the same LoRA on new data
- Simple, but may cause catastrophic forgetting

**Option 2: LoRA Expansion (Add new heads)**
- Freeze the original LoRA adapter
- Add a new LoRA adapter for the new task
- Merge predictions via attention-based routing or simple averaging
- More complex, better for preserving old knowledge

**Option 3: Replay + Fine-tune**
- Continue training on new data + small replay buffer from old data
- Replay buffer: random 5-10% sample from original training set
- Good balance of simplicity and forgetting prevention

**Recommended: Start with Option 1 (simplest), then Option 3 (replay), then Option 2 if needed.**

### CL Training Protocol

1. **Phase 1: Base Training**
   - Train on T0-T15 (20K docs) — all three DocID schemes
   - Save model checkpoint
   - Evaluate on T0-T15 eval queries → baseline accuracy

2. **Phase 2: CL Update**
   - Load Phase 1 checkpoint
   - Train on T16-T18 (~4,657 new docs)
   - For chrono-semantic: new docs get task IDs T16, T17, T18 (new prefix tokens)
   - For semantic RQ: use the SAME trained RQ quantizer to encode new docs (do NOT retrain RQ)
   - For atomic: assign new random IDs to new docs

3. **Phase 3: Evaluation**
   - Evaluate on T0-T15 eval queries → measure forgetting
   - Evaluate on T16-T18 eval queries → measure new learning
   - Compare across the three DocID schemes

### Semantic Encoding of New Documents (CL Phase)
**Critical: Use the SAME RQ quantizer from Phase 1.**
1. Embed the new 5K docs with the same sentence-transformer
2. Apply PCA with the same fitted PCA transform
3. Quantize with the same FAISS RQ codebook
4. This ensures the new documents get IDs in the same semantic space
5. Save the PCA and RQ objects after Phase 1 for reuse

### Metrics for Forgetting Analysis

1. **Backward Transfer (BWT):**
   - Accuracy on old tasks (T0-T15) AFTER learning new tasks
   - BWT = Acc_after - Acc_before (negative = forgetting)

2. **Forward Transfer (FWT):**
   - Accuracy on new tasks (T16-T18) after CL update
   - Compare to training on only T16-T18 from scratch

3. **Positive Forgetting Rate (PFR):**
   - Our novel metric from the proposal
   - Among documents whose retrieval accuracy dropped, what fraction were "supposed to" become less relevant?
   - Requires cross-temporal relevance labels (may skip for now, or use simple heuristic: old docs on the same topic as new docs)

4. **Per-month accuracy heatmap:**
   - Rows: evaluation time (after base, after CL)
   - Columns: months T0-T18
   - Cells: Hits@1 per month
   - Visually shows which months are forgotten

### Visualization for CL Analysis

1. **Forgetting curve:**
   - X-axis: months (T0-T18)
   - Y-axis: Hits@1 accuracy
   - Three lines per DocID scheme (before CL, after CL)
   - Shows which months lose accuracy

2. **Comparison bar chart:**
   - Grouped bars: Atomic vs Semantic vs Chrono
   - Metrics: Old accuracy, New accuracy, BWT
   - Shows which DocID scheme resists forgetting best

3. **Task-level heatmap:**
   - Per the heatmap description above

---

## Implementation Order

1. Load the 25K dataset CSV (from Task 1)
2. Add visualization section (Part A)
3. Scale up RQ, verify no collision issues (Part B)
4. Run 3-way comparison on 20K (Part C)
5. Implement CL training loop (Part D)
6. Run CL experiments and generate analysis (Part D)

---

## Expected Findings (Hypotheses)

1. **Semantic RQ > Atomic** at 20K scale (structural IDs help more with larger vocabulary)
2. **Chrono-Semantic ≈ Semantic RQ** on base training (temporal prefix doesn't help when all data is seen at once)
3. **Chrono-Semantic > Semantic RQ on CL** (temporal prefix helps the model separate old vs new knowledge)
4. **Chrono-Semantic shows "positive forgetting"** — accuracy drops more on old months that share topics with new months, which is desirable
