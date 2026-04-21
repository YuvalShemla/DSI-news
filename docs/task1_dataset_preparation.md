# Task 1: Prepare 25K BBC News Dataset with Pseudo-Queries

## Goal
Create a production-ready dataset of ~25K BBC News articles (20K for initial training + ~5K for continual learning) with 11 pseudo-queries per document (10 training + 1 evaluation ground truth).

---

## Dataset Selection: BBC News AllTime

**Why BBC over CC-News:**
- Single source = consistent article quality and formatting
- Clean temporal coverage (2017-2025) with monthly granularity
- Median content length: 3,185 chars (well-suited for embeddings)
- Clean section labels, proper dates, good English

**Raw stats:** 159,821 articles total
**After dedup (by link) + filter (content >= 200 chars):** ~108,457 articles

---

## Month Selection

### Training Set: Jan 2023 - Apr 2024 (16 months, ~20,471 docs)

| Month | Task ID | Doc Count | Cumulative |
|-------|---------|-----------|------------|
| 2023-01 | T0 | 1,184 | 1,184 |
| 2023-02 | T1 | 1,125 | 2,309 |
| 2023-03 | T2 | 1,149 | 3,458 |
| 2023-04 | T3 | 1,180 | 4,638 |
| 2023-05 | T4 | 1,232 | 5,870 |
| 2023-06 | T5 | 1,202 | 7,072 |
| 2023-07 | T6 | 1,205 | 8,277 |
| 2023-08 | T7 | 1,224 | 9,501 |
| 2023-09 | T8 | 1,238 | 10,739 |
| 2023-10 | T9 | 1,179 | 11,918 |
| 2023-11 | T10 | 1,109 | 13,027 |
| 2023-12 | T11 | 1,039 | 14,066 |
| 2024-01 | T12 | 1,128 | 15,194 |
| 2024-02 | T13 | 1,441 | 16,635 |
| 2024-03 | T14 | 2,172 | 18,807 |
| 2024-04 | T15 | 1,664 | 20,471 |

### Continual Learning Set: May - Jul 2024 (3 months, ~4,657 docs)

| Month | Task ID | Doc Count | Cumulative |
|-------|---------|-----------|------------|
| 2024-05 | T16 | 739 | 21,210 |
| 2024-06 | T17 | 1,966 | 23,176 |
| 2024-07 | T18 | 1,952 | 25,128 |

**Total: 19 months, ~25,128 documents**

### Task IDs (Chrono Prefix)
Each month gets a sequential task ID (T0 - T18). In the chrono-semantic DocID scheme, this task ID becomes the temporal prefix token: `<task_0>`, `<task_1>`, ..., `<task_18>`. This is simpler and more flexible than year-month encoding since within a single-year experiment we just need ordinal month distinction.

---

## Data Cleaning Pipeline

### Step 1: Deduplicate
1. Remove exact duplicates by `link` column (removes ~47,834 dups)
2. Remove near-duplicate content: fuzzy dedup on title (Jaccard similarity > 0.8 on word sets)
3. Filter out articles with content < 200 characters

### Step 2: Content Cleaning
1. Strip HTML artifacts (if any remain)
2. Remove boilerplate footers (e.g., "BBC News", "Follow us on...")
3. Truncate very long articles to first 2,000 words (keep them manageable for embedding)
4. Ensure each article has a non-empty title

### Step 3: Select & Assign IDs
1. Filter to the 19 target months (Jan 2023 - Jul 2024)
2. Assign sequential `doc_id` within each month: `doc_{task_id}_{seq}` (e.g., `doc_T0_0001`)
3. Assign `task_id` (0-18) based on month

---

## Query Generation Pipeline

### Approach
Use Gemini 2.0 Flash via the API to generate pseudo-queries. For each document, generate **11 queries** total:
- **10 training queries**: diverse query styles to teach the model multiple access patterns
- **1 evaluation query**: held out as ground truth for retrieval evaluation

### Query Styles (for the 10 training queries)
Generate in batches of 10 per document using a single LLM call with a structured prompt:

1. **Keyword query** (2x): Short keyword-style queries (3-6 words)
2. **Natural question** (3x): Factual questions the article answers
3. **Topical query** (2x): Broader topic queries where the article is relevant
4. **Background linking** (2x): Queries from a journalist needing this as background
5. **Specific detail** (1x): Query about a specific fact/figure in the article

### Evaluation Query (1 per doc)
A separate call with a stricter prompt:
- Must be a factual question
- Must be answerable ONLY by this specific article
- Must not overlap with any training query

### Prompt Design (batch of 10 + 1 eval)

**Training queries prompt:**
```
Given this news article, generate 10 diverse search queries that would lead
a user to find this article. Follow this exact format:

KEYWORD_1: [3-6 word keyword query]
KEYWORD_2: [3-6 word keyword query]
QUESTION_1: [factual question this article answers]
QUESTION_2: [factual question this article answers]
QUESTION_3: [factual question this article answers]
TOPICAL_1: [broader topic query]
TOPICAL_2: [broader topic query]
BACKGROUND_1: [query from journalist needing background]
BACKGROUND_2: [query from journalist needing background]
DETAIL_1: [query about a specific fact/number in the article]

Rules:
- Each query must be 5-20 words
- Do NOT copy phrases verbatim from the article
- Queries should be diverse (not variations of the same question)
- Output ONLY the labeled queries, nothing else

Article title: {title}
Article text: {text}
```

**Evaluation query prompt:**
```
Read this news article carefully. Write ONE factual question that:
1. Can be answered specifically by this article
2. Is specific enough that few other articles could answer it
3. Is a natural question someone might search for
4. Is 8-15 words long

Article title: {title}
Article text: {text}

Write ONLY the question, nothing else:
```

### Cost Estimate
- ~25K articles x 2 calls each = ~50K API calls
- Input: ~25K * (800 avg article tokens + 200 prompt) = ~25M input tokens
- Output: ~25K * (300 training queries + 30 eval query) = ~8.25M output tokens
- **Gemini 2.0 Flash cost: ~$2.50 input + $0.65 output ≈ $3.15 total**
- At 1500 RPM paid tier: ~33 minutes wall clock

### Rate Limiting & Reliability
- Batch requests with asyncio (10-20 concurrent)
- Retry failed calls up to 3 times with exponential backoff
- Save progress incrementally (every 500 docs) to avoid losing work
- Log all failures for manual review

---

## Output Format

### File: `data/queries/bbc_25k_dataset.csv`

| Column | Description | Example |
|--------|-------------|---------|
| `doc_id` | Unique document ID | `doc_T0_0001` |
| `task_id` | Temporal task (0-18) | `0` |
| `year_month` | Source month | `2023-01` |
| `title` | Article headline | `UK interest rates...` |
| `content` | Article text (truncated to 2000 words) | `The Bank of England...` |
| `section` | BBC section | `Business` |
| `link` | Original BBC URL | `https://bbc.co.uk/...` |
| `query` | Query text | `what happened to UK interest rates` |
| `query_type` | Query style | `QUESTION_1` |
| `split` | train or eval | `train` |

Each document appears 11 times (10 train queries + 1 eval query).
**Expected total rows: ~25,128 * 11 = ~276,408 rows**

### Separate metadata file: `data/queries/bbc_25k_metadata.csv`

| Column | Description |
|--------|-------------|
| `doc_id` | Unique document ID |
| `task_id` | Temporal task (0-18) |
| `year_month` | Source month |
| `title` | Article headline |
| `content` | Full article text |
| `section` | BBC section |
| `link` | Original URL |

**Expected rows: ~25,128**

---

## Implementation Steps

1. **`src/data/prepare_bbc_25k.py`** - Main data preparation script:
   - Load BBC parquet, dedup, filter, select months
   - Assign doc_ids and task_ids
   - Save cleaned metadata CSV

2. **`src/data/generate_queries_batch.py`** - Query generation script:
   - Load metadata CSV
   - Generate 10 training + 1 eval query per doc via Gemini
   - Parse structured output, validate query quality
   - Save incrementally with checkpointing
   - Merge into final dataset CSV

3. **Validation checks:**
   - Every doc_id has exactly 11 queries (10 train + 1 eval)
   - No empty queries, no queries shorter than 4 words
   - No exact duplicate queries within same document
   - Eval query doesn't overlap with training queries
   - Monthly distribution matches expected counts

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Gemini rate limits | Use paid tier, async batching, checkpointing |
| Poor query quality | Post-filter by length, dedup, manual spot-check |
| May 2024 has only 739 docs | This is fine - real-world CL has uneven task sizes |
| Dataset too large for GitHub | Use git-lfs or compress. At ~276K rows x ~1KB avg = ~270MB. Will compress well. Or split into metadata + queries. |
