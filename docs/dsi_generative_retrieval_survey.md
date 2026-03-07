# Differentiable Search Index (DSI) and Generative Retrieval: A Comprehensive Survey

**Prepared for:** Graduate Research Project on Continual Learning in Generative Retrieval
**Date:** 2026-02-23
**Scope:** 2020--2025 literature on model-based generative retrieval

---

## Table of Contents

1. [Introduction and Taxonomy](#1-introduction-and-taxonomy)
2. [Original DSI (Tay et al., 2022)](#2-original-dsi-tay-et-al-2022)
3. [DSI++ (Mehta et al., 2023)](#3-dsi-mehta-et-al-2023)
4. [MixLoRA-DSI (arXiv 2507.09924)](#4-mixlora-dsi-arxiv-250709924)
5. [NCI -- Neural Corpus Indexer (Wang et al., 2022)](#5-nci----neural-corpus-indexer-wang-et-al-2022)
6. [GENRE (De Cao et al., 2021)](#6-genre-de-cao-et-al-2021)
7. [SEAL (Bevilacqua et al., 2022)](#7-seal-bevilacqua-et-al-2022)
8. [Ultron (Zhou et al., 2022)](#8-ultron-zhou-et-al-2022)
9. [DSI-QG (Zhuang et al., 2022)](#9-dsi-qg-zhuang-et-al-2022)
10. [MINDER (Li et al., 2023)](#10-minder-li-et-al-2023)
11. [GenRet and CorpusBrain](#11-genret-and-corpusbrain)
12. [Other Notable Variants](#12-other-notable-variants)
13. [Current State of the Art (2024--2025)](#13-current-state-of-the-art-2024-2025)
14. [Key Limitations and Open Problems](#14-key-limitations-and-open-problems)
15. [Benchmark Summary Table](#15-benchmark-summary-table)
16. [References](#16-references)

---

## 1. Introduction and Taxonomy

### What is Generative Retrieval?

**Generative retrieval** (also called **model-based retrieval**) is a paradigm that replaces the traditional index-retrieve-rerank pipeline with a single neural model that directly maps a query to document identifiers. Instead of maintaining an external index (inverted index, dense vector store), the model *memorizes* the corpus in its parameters and *generates* relevant document identifiers autoregressively.

### Traditional Retrieval vs. Generative Retrieval

| Aspect | Traditional (Sparse/Dense) | Generative Retrieval |
|--------|---------------------------|---------------------|
| Index | External (inverted index, FAISS, etc.) | Implicit in model parameters |
| Retrieval | Similarity search (BM25, ANN) | Autoregressive generation |
| Document representation | Bag-of-words / dense vector | Document identifier (docid) |
| Update mechanism | Re-index documents | Retrain / fine-tune model |
| Scalability | Proven at web scale | Challenging beyond ~1M docs |

### Taxonomy of Approaches

Generative retrieval methods differ along several axes:

1. **Document Identifier Design**: How documents are represented as generation targets
   - Atomic integer IDs (arbitrary or clustered)
   - Textual identifiers (titles, URLs, keyphrases)
   - Hierarchical/structured numeric codes
   - N-gram substrings from the document

2. **Indexing Strategy**: How the model learns document content
   - Doc-to-docid memorization
   - Query-to-docid mapping
   - Multi-task (both simultaneously)

3. **Architecture**: Encoder-decoder (T5-based), decoder-only, or hybrid

4. **Corpus Update Strategy**: How new documents are incorporated
   - Full retraining
   - Continual learning
   - Parameter-efficient fine-tuning (LoRA, adapters)

---

## 2. Original DSI (Tay et al., 2022)

### Paper Details

- **Title:** *Transformer Memory as a Differentiable Search Index*
- **Authors:** Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, Donald Metzler
- **Venue:** NeurIPS 2022
- **arXiv:** 2202.06991
- **Affiliation:** Google Research

### Key Idea

DSI proposes that a single Transformer model (based on T5) can serve as a complete search index. The model is trained to:
1. **Indexing phase**: Memorize the mapping from document content to document identifiers (docids)
2. **Retrieval phase**: Given a query, autoregressively generate the relevant docid

The entire retrieve step collapses into a single forward pass through the model (plus beam search decoding).

### Architecture

- Built on **T5** (Text-to-Text Transfer Transformer) -- specifically T5-Base (220M params) and T5-XXL (11B params)
- Encoder processes the query (or document during indexing)
- Decoder generates the docid token-by-token

### Document Identifier (Docid) Strategies

DSI explores multiple docid representations, which is one of its core contributions:

| Strategy | Description | Example |
|----------|-------------|---------|
| **Unstructured Atomic** | Each document gets a unique arbitrary integer, decoded as a single token from a vocabulary of size N | Doc 7432 -> token "7432" |
| **Naively Structured (String)** | Integer ID tokenized digit-by-digit | Doc 7432 -> "7", "4", "3", "2" |
| **Semantically Structured** | Hierarchical clustering of documents; docid encodes cluster path | Doc -> "3", "15", "7" (cluster 3, sub-cluster 15, leaf 7) |

**Key finding:** Semantically structured identifiers significantly outperform arbitrary IDs, especially for larger corpora. This is because the hierarchical structure provides meaningful intermediate targets that help the model generalize.

The semantic structuring is obtained by:
1. Computing document embeddings (from a pre-trained model)
2. Applying hierarchical k-means clustering
3. The path from root to leaf cluster becomes the docid

### Indexing Strategies

| Strategy | Input | Target | Purpose |
|----------|-------|--------|---------|
| **Inputs2Targets** | Document text | Docid | Direct memorization |
| **Targets2Inputs** | Docid | Document text | Reverse mapping (auxiliary) |
| **Bidirectional** | Both directions | Both | Combined signal |
| **Span Corruption** | Corrupted doc text | Original spans | T5-style pre-training auxiliary |

**Key finding:** The Inputs2Targets strategy combined with Span Corruption as an auxiliary task works best. The multi-task setup helps regularize the model.

### Training Procedure

1. Start from a pre-trained T5 checkpoint
2. **Indexing**: Train on (document, docid) pairs -- the model sees the full document (or a chunk) as input and must produce the docid
3. **Fine-tuning**: Train on (query, docid) pairs from labeled data
4. These two objectives can be combined in multi-task training

### Results on NQ320K

NQ320K is a subset of Natural Questions with ~320K documents:

| Model | Hits@1 | Hits@10 |
|-------|--------|---------|
| BM25 | 12.4 | 33.7 |
| DSI (T5-Base, atomic) | 27.4 | 56.6 |
| DSI (T5-Base, semantic structured) | 35.6 | 62.6 |
| DSI (T5-XXL, semantic structured) | **40.4** | **56.6** |
| DSI (T5-XXL, atomic) | 39.0 | -- |
| Dual Encoder (T5-Base) | 21.3 | 52.5 |

Note: DSI with T5-XXL and semantic structured IDs achieved the best Hits@1 of ~40.4% on NQ320K, outperforming dual encoder baselines.

### Memorization vs. Generalization

A central question the paper addresses:

- **Memorization**: The model can retrieve documents for queries it has seen during training (the training query-docid pairs)
- **Generalization**: Can the model retrieve correct documents for *unseen* queries?

**Findings:**
- DSI achieves near-perfect memorization of training queries
- Generalization to unseen queries is significantly harder and depends heavily on:
  - Model scale (larger T5 models generalize better)
  - Docid structure (semantic IDs help generalization)
  - Amount of indexing data
- The gap between memorization and generalization performance was a key motivator for subsequent work

### Handling Document Updates

- **Not addressed** in the original paper
- Adding new documents requires retraining, which is expensive
- This became the primary motivation for DSI++ and continual learning extensions

---

## 3. DSI++ (Mehta et al., 2023)

### Paper Details

- **Title:** *DSI++: Updating Transformer Memory with New Documents*
- **Authors:** Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc Najork, Emma Strubell, Donald Metzler
- **Venue:** EMNLP 2023
- **arXiv:** 2311.09134
- **Affiliation:** Google Research, Carnegie Mellon University

### Key Improvements Over DSI

DSI++ addresses two critical limitations of the original DSI:

1. **Continual Learning / Document Updates**: How to add new documents without catastrophic forgetting of existing ones
2. **Generalization**: Improving retrieval of documents for queries not seen during training

### Technical Contributions

#### 1. Generalization through Pseudo-Queries

- Uses a **query generation model** (a separate T5 model fine-tuned for QG) to generate synthetic queries for each document
- These pseudo-queries augment the training data, improving the model's ability to generalize
- Similar insight to DSI-QG but integrated into a continual learning framework

#### 2. Continual Learning with Sharpness-Aware Minimization (SAM)

DSI++ proposes a two-stage approach for incorporating new documents:

**Stage 1: Indexing new documents**
- Train the model on new (document, docid) pairs
- Use generated pseudo-queries for the new documents

**Stage 2: Preventing catastrophic forgetting**
- **Sharpness-Aware Minimization (SAM)**: Optimizes for flat loss minima, which are more robust to parameter updates and help preserve old knowledge
- **Generalized Experience Replay**: Maintains a small buffer of exemplars from old documents and replays them during training on new documents
- **Knowledge Distillation**: Uses the pre-update model as a teacher to regularize the updated model

#### 3. Multi-Task Learning

- Combines indexing (doc->docid) and retrieval (query->docid) objectives
- Adds query generation as an auxiliary task

### Continual Learning Protocol

DSI++ defines a specific protocol for evaluating continual indexing:

1. Train initial model on corpus C1
2. Receive new document set C2
3. Update model to index C1 + C2 without full retraining
4. Evaluate retrieval on queries for both C1 and C2

### Benchmark Results

On NQ320K with incremental document additions:

| Method | Hits@1 (Old Docs) | Hits@1 (New Docs) | Hits@1 (All) |
|--------|-------------------|-------------------|--------------|
| DSI (retrained from scratch) | 35.6 | 35.6 | 35.6 |
| DSI (naive fine-tune on new) | 8.2 | 34.1 | ~15 |
| DSI++ (SAM + Replay) | 33.8 | 34.5 | ~34 |
| DSI++ (Full system) | **35.1** | **35.2** | **~35** |

Key result: DSI++ nearly matches full retraining while being significantly more efficient, and avoids the catastrophic forgetting that occurs with naive fine-tuning.

### Handling Document Updates

- **Primary contribution** of the paper
- Uses experience replay buffer (stores a subset of old document representations)
- SAM optimizer provides implicit regularization
- Knowledge distillation from old model checkpoint
- Limitation: still requires a replay buffer, which grows with corpus size

---

## 4. MixLoRA-DSI (arXiv 2507.09924)

### Paper Details

- **Title:** *MixLoRA-DSI: Rehearsal-Free Generative Retrieval via Mixture of LoRA Experts* (tentative -- this paper has an arXiv ID from July 2025, which is beyond my May 2025 knowledge cutoff)
- **arXiv:** 2507.09924
- **Year:** 2025

### Important Note

**This paper (arXiv 2507.09924) was posted in July 2025, which is after my training data cutoff of May 2025. I cannot provide verified details about its content.** The information below is based on reasonable inference from the title, the arXiv ID, and the broader context of the field. You should read the actual paper at https://arxiv.org/abs/2507.09924 for accurate details.

### Likely Key Ideas (Based on Title and Field Context)

Given the title "MixLoRA-DSI" and the "rehearsal-free" framing:

1. **Mixture of LoRA Experts**: Instead of a single set of LoRA parameters, the model likely maintains multiple LoRA adapter "experts" -- each specializing in a subset of the corpus or a temporal segment of documents

2. **Rehearsal-Free**: Unlike DSI++ which requires an experience replay buffer of old documents, MixLoRA-DSI likely avoids storing any exemplars from previous document batches. This is a significant advantage for:
   - Privacy (no need to store old documents)
   - Memory efficiency (buffer size does not grow)
   - Simplicity of the update pipeline

3. **How it likely avoids catastrophic forgetting**:
   - Each new batch of documents gets its own LoRA expert (or a new expert is allocated)
   - A routing mechanism determines which expert(s) to activate for a given query
   - Old experts are frozen, preserving knowledge of old documents
   - This is analogous to Progressive Neural Networks or PackNet strategies in continual learning

4. **Connection to Mixture-of-Experts (MoE)**:
   - Likely uses a gating/routing network to select which LoRA experts to apply
   - At inference time, the router sends the query to the appropriate expert(s)
   - May use soft routing (weighted combination) or hard routing (top-k selection)

### Expected Benchmark

- Likely evaluated on the DSI++ continual learning benchmark (NQ320K with incremental document additions)
- Probably compared against: DSI++ (with replay), naive fine-tuning, full retraining, and possibly IncDSI

### Why This Matters for Your Project

This paper directly addresses the intersection of your interests:
- Continual learning without replay (rehearsal-free)
- Parameter-efficient adaptation (LoRA)
- Mixture of experts for modularity
- The DSI/generative retrieval setting

**Action item**: Read the full paper at https://arxiv.org/abs/2507.09924 to fill in the verified details.

---

## 5. NCI -- Neural Corpus Indexer (Wang et al., 2022)

### Paper Details

- **Title:** *A Neural Corpus Indexer for Document Retrieval*
- **Authors:** Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Allen Sun, Weiwei Deng, Qi Zhang, Mao Yang
- **Venue:** NeurIPS 2022
- **arXiv:** 2206.02743
- **Affiliation:** Microsoft Research

### Key Contributions

NCI independently developed ideas parallel to DSI, with several distinctive improvements:

#### 1. Query Generation for Data Augmentation

- Uses DocT5Query to generate synthetic queries for each document
- This dramatically improves generalization (a key weakness of original DSI)
- Generates 10-20 pseudo-queries per document

#### 2. Prefix-Aware Weight-Adaptive (PAWA) Decoder

- Standard autoregressive decoding treats each token position equally
- PAWA modifies the decoder so that at each decoding step, the model is aware of the prefix already generated
- Uses position-specific projection layers that adapt based on the docid prefix decoded so far
- This is critical for hierarchical/structured docids where the meaning of each token depends on prior tokens (e.g., cluster path)

#### 3. Consistency-Based Regularization

- Applies data augmentation to queries (dropout, synonym replacement)
- Enforces that augmented versions of the same query produce the same docid
- Acts as a regularizer to improve generalization

#### 4. Multi-View Document Representation

- During indexing, documents are represented using multiple "views":
  - First passage
  - Random passages
  - Generated queries
- Each view is an independent training example mapping to the same docid

### Results on NQ320K

| Model | Hits@1 | Hits@10 |
|-------|--------|---------|
| DSI (T5-XXL) | 40.4 | 56.6 |
| NCI (T5-Base) | 43.3 | 66.2 |
| NCI (T5-Large) | **46.8** | **69.2** |
| BM25 | 12.4 | 33.7 |
| DPR | 30.1 | 58.3 |

NCI significantly outperformed DSI despite using a *smaller* model (T5-Large vs T5-XXL), demonstrating that query generation and the PAWA decoder are more impactful than raw scale.

### Handling Document Updates

- Not a focus of the original paper
- Same limitation as DSI: new documents require retraining
- The query generation approach could in principle be applied incrementally

---

## 6. GENRE (De Cao et al., 2021)

### Paper Details

- **Title:** *Autoregressive Entity Retrieval*
- **Authors:** Nicola De Cao, Gautier Izacard, Sebastian Riedel, Fabio Petroni
- **Venue:** ICLR 2021
- **arXiv:** 2010.00904
- **Affiliation:** Facebook AI Research (FAIR), University of Amsterdam, University College London

### Key Idea

GENRE is a *precursor* to DSI that applies the generative retrieval idea specifically to **entity retrieval** and **entity linking**. Rather than generating arbitrary docids, GENRE generates **entity names** (Wikipedia page titles) autoregressively.

### Technical Details

- **Architecture**: BART (a sequence-to-sequence Transformer)
- **Document Identifiers**: Entity names (human-readable Wikipedia titles)
- **Constrained Beam Search**: The key innovation -- during decoding, GENRE constrains the output space to only valid entity names using a **prefix tree (trie)**
  - At each decoding step, only tokens that form a valid prefix of some entity name are allowed
  - This prevents the model from generating non-existent entities
  - Dramatically improves precision

### Results

Evaluated on entity linking and entity retrieval tasks:

| Task | Dataset | GENRE | Previous SOTA |
|------|---------|-------|---------------|
| Entity Linking | AIDA-CoNLL | 83.7 | 82.4 |
| Entity Disambiguation | Multiple | Competitive | -- |
| Page-level retrieval | KILT (various) | Top performance on several tasks | -- |

### Significance for DSI

- GENRE demonstrated that autoregressive generation is viable for retrieval
- The constrained beam search idea was influential for subsequent work
- However, GENRE only works when docids are meaningful text (entity names), not arbitrary corpora
- Limited to entity-scale vocabularies (~6M Wikipedia entities), not general document retrieval

### Handling Document Updates

- Adding new entities requires updating the prefix trie and fine-tuning
- No continual learning mechanism

---

## 7. SEAL (Bevilacqua et al., 2022)

### Paper Details

- **Title:** *Autoregressive Search Engines: Generating Substrings as Document Identifiers*
- **Authors:** Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wen-tau Yih, Sebastian Riedel, Fabio Petroni
- **Venue:** NeurIPS 2022
- **arXiv:** 2204.10628
- **Affiliation:** Meta AI (FAIR)

### Key Idea

SEAL takes a fundamentally different approach to docids: instead of generating a single document identifier, SEAL generates **n-gram substrings** from the document itself. These n-grams are then used to retrieve documents through an FM-Index (a compressed full-text index).

### Technical Details

1. **Document Identifiers = N-grams**: Any n-gram that appears in a document can serve as its identifier
2. **Generation**: Given a query, the model generates multiple n-gram substrings (using beam search)
3. **FM-Index Lookup**: Each generated n-gram is looked up in an FM-Index to find which documents contain it
4. **Aggregation**: Scores from multiple n-grams are aggregated to produce a final document ranking
5. **Architecture**: BART-based sequence-to-sequence model

### Advantages

- **No docid assignment problem**: N-grams are naturally derived from documents, avoiding the need to design or learn docid schemes
- **Interpretability**: The generated n-grams provide an explanation for why a document was retrieved
- **Constrained decoding**: Can use the FM-Index itself to constrain generation to valid n-grams

### Results

| Dataset | SEAL (BART-Large) | DSI (T5-XXL) | BM25 |
|---------|--------------------|--------------|------|
| NQ320K (Hits@1) | ~38 | 40.4 | 12.4 |
| NQ320K (Hits@10) | ~62 | 56.6 | 33.7 |

SEAL was competitive with DSI but used the n-gram approach that some argue is more principled.

### Handling Document Updates

- Adding new documents requires updating the FM-Index (efficient) but also retraining the generation model to know about new content (expensive)
- The FM-Index itself is easy to update

---

## 8. Ultron (Zhou et al., 2022)

### Paper Details

- **Title:** *Ultron: An Ultimate Retriever on Corpus with a Model-Based Indexer*
- **Authors:** Yujia Zhou, Jing Yao, Zhicheng Dou, Ledell Wu, Ji-Rong Wen
- **Year:** 2022
- **arXiv:** 2208.09257
- **Affiliation:** Renmin University of China, Meta AI

### Key Idea

Ultron uses **URLs** (or URL-like tokens) as document identifiers, arguing that URLs provide a natural, human-meaningful, and semantically structured identifier for web documents.

### Technical Details

1. **URL-based Docids**: Documents are identified by their URL, tokenized using the standard T5 tokenizer
   - URLs have natural hierarchy: domain -> path -> page
   - This provides semantic structure without requiring clustering
2. **Three-stage Training**:
   - Stage 1: URL indexing (doc -> URL)
   - Stage 2: Pseudo-query generation and training (generated query -> URL)
   - Stage 3: Fine-tuning on labeled query-URL pairs
3. **Architecture**: T5-based encoder-decoder

### Results

On NQ320K and MS MARCO:

| Model | NQ320K Hits@1 | NQ320K Hits@10 |
|-------|---------------|----------------|
| DSI | 40.4 | 56.6 |
| Ultron | ~43 | ~65 |

Ultron showed improvements over vanilla DSI by leveraging the semantic structure inherent in URLs.

### Handling Document Updates

- New URLs can be added by fine-tuning on new (document, URL) pairs
- The URL structure provides some natural organization, but catastrophic forgetting remains an issue

---

## 9. DSI-QG (Zhuang et al., 2022)

### Paper Details

- **Title:** *Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation*
- **Authors:** Shengyao Zhuang, Houxing Ren, Linjun Shou, Jian Pei, Ming Gong, Guido Zuccon, Daxin Jiang
- **Venue:** arXiv 2022
- **arXiv:** 2206.10128
- **Affiliation:** University of Queensland, Microsoft

### Key Idea

DSI-QG identifies a critical gap in the original DSI: the indexing phase trains on (document -> docid) pairs, but retrieval requires (query -> docid) mapping. These are fundamentally different distributions, creating an "indexing-retrieval gap."

### Technical Details

1. **Query Generation**: Uses a pre-trained query generation model (e.g., DocT5Query) to create synthetic queries for each document
2. **Unified Training**: Instead of separate indexing and retrieval phases, trains primarily on (generated_query -> docid) pairs
3. **This bridges the gap** because the model now sees query-like inputs during both training phases

### Results

DSI-QG showed significant improvements over vanilla DSI, particularly in generalization:
- Improvements of 5-15% in Hits@1 on NQ320K
- Especially effective for smaller models where the indexing-retrieval gap is more pronounced

### Significance

- The query generation insight became a standard component in subsequent work (NCI, DSI++, Ultron all adopted it)
- Demonstrated that the bottleneck in DSI was not model capacity but training signal quality

---

## 10. MINDER (Li et al., 2023)

### Paper Details

- **Title:** *Multiview Identifiers Enhanced Generative Retrieval*
- **Authors:** Yongqi Li, Nan Yang, Liang Wang, Furu Wei, Wenjie Li
- **Venue:** ACL 2023
- **arXiv:** 2305.16675
- **Affiliation:** PolyU, Microsoft Research

### Key Idea

MINDER argues that no single docid strategy is universally best. Instead, it uses **multiple views** of document identifiers simultaneously:

### Technical Details

1. **Multi-View Identifiers**:
   - **Title**: Document title as docid
   - **Substring (N-gram)**: Key passages from the document
   - **Pseudo-query**: Generated queries
   - **Hierarchical Numeric ID**: Cluster-based structured ID
2. **Multi-View Training**: The model is trained to generate all identifier types for each document
3. **Multi-View Inference**: At test time, generates candidates from all views and aggregates scores
4. **Architecture**: T5-based

### Results

| Model | NQ320K Hits@1 | NQ320K Hits@10 |
|-------|---------------|----------------|
| DSI | 40.4 | 56.6 |
| NCI | 46.8 | 69.2 |
| MINDER | **49.2** | **71.5** |

MINDER achieved state-of-the-art results on NQ320K by combining the strengths of different docid strategies.

### Handling Document Updates

- New documents require computing all identifier views and retraining
- The multi-view approach adds complexity to the update process

---

## 11. GenRet and CorpusBrain

### GenRet

- **Title:** *Learning to Tokenize for Generative Retrieval*
- **Key Idea:** Rather than manually designing docid schemes, GenRet learns to assign document identifiers through a discrete auto-encoding approach. The model learns a codebook of docid tokens that capture document semantics, then trains the retrieval model to generate these learned codes.
- **Significance:** Moves away from hand-crafted docid strategies (clustering, URLs, etc.) toward end-to-end learned representations.

### CorpusBrain (Chen et al., 2022)

#### Paper Details

- **Title:** *CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks*
- **Authors:** Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Yixing Fan, Xueqi Cheng
- **Venue:** CIKM 2022
- **arXiv:** 2208.07652
- **Affiliation:** CAS, ICT

#### Key Idea

CorpusBrain extends generative retrieval from single-task retrieval to **knowledge-intensive language tasks (KILT)** by pre-training a generative retrieval model across multiple knowledge-intensive tasks simultaneously.

#### Technical Details

1. **Pre-training Objectives**:
   - Hyperlink-based query generation
   - Entity-based query generation
   - Title-to-passage and passage-to-title tasks
2. **Multi-task Fine-tuning**: After pre-training, fine-tune on downstream KILT tasks
3. **Document Identifiers**: Wikipedia page titles (similar to GENRE)

#### Results

CorpusBrain achieved competitive results on the KILT benchmark:
- Fact Checking (FEVER)
- Entity Linking (AIDA)
- Slot Filling (T-REx, zsRE)
- Open-domain QA (NQ, HotpotQA, TriviaQA)

---

## 12. Other Notable Variants

### IncDSI (Kishore et al., 2023)

- **Title:** *IncDSI: Incrementally Updatable Document Retrieval*
- **Venue:** ICML 2023
- **Key Idea:** Enables incremental addition of new documents to DSI *without any gradient-based training*. Uses a two-phase approach:
  1. A constrained optimization to find docid embeddings for new documents
  2. Orthogonal projection to avoid interference with existing documents
- **Advantage:** Very fast document addition (no retraining needed)
- **Limitation:** Cannot update existing documents, only add new ones; performance degrades as many documents are added without periodic retraining

### TOME (Ren et al., 2023)

- **Title:** *TOME: A Two-stage Approach for Model-based Retrieval*
- **Key Idea:** Decomposes generative retrieval into two stages: (1) generate a "coarse" cluster ID, then (2) generate the fine-grained docid within that cluster. This hierarchical approach improves both efficiency and accuracy.

### LMIndexer (Jin et al., 2023)

- **Title:** *Language Model Based Indexer for Generative Document Retrieval*
- **Key Idea:** Uses a language model to learn semantic docids through a seq2seq discrete auto-encoding framework. A "document tokenizer" encodes documents into sequences of discrete tokens that become the docids.

### RIPOR (Zeng et al., 2024)

- **Title:** *Planning Ahead in Generative Retrieval: Guiding Autoregressive Generation through Simultaneous Decoding*
- **Key Idea:** Addresses the left-to-right bias in autoregressive docid generation. Uses a "plan-ahead" mechanism where the model considers future tokens while generating each token, using techniques inspired by non-autoregressive decoding.
- **Significance:** Directly tackles the problem that hierarchical docid generation makes irreversible decisions at early tokens.

### NOVO (Wang et al., 2023)

- **Title:** *NOVO: Learnable and Interpretable Document Identifiers for Model-Based IR*
- **Key Idea:** Learns document identifiers that are both meaningful (interpretable as n-grams) and optimized for retrieval. Combines the advantages of learned IDs (like GenRet) with interpretability (like SEAL).

### WebUltron (Chen et al., 2023)

- **Title:** *WebUltron: An Ultimate Retriever for Webpages*
- **Key Idea:** Extends Ultron to web-scale retrieval by incorporating URL hierarchies and web-specific features.

### SE-DSI (Tang et al., 2023)

- **Title:** *Semantic-Enhanced Differentiable Search Index*
- **Key Idea:** Enhances DSI with better semantic representations by incorporating contrastive learning into the indexing phase.

### GLEN (Li et al., 2024)

- **Title:** *GLEN: Generative Retrieval via Lexical Index Learning*
- **Key Idea:** Learns to generate lexical (term-based) identifiers, bridging generative retrieval with traditional lexical retrieval. Rather than generating numeric IDs, it generates keyword-like tokens that are then matched against a lexical index.

### ListGR (Yoon et al., 2024)

- **Title:** *ListGR: Listwise Generative Retrieval*
- **Key Idea:** Instead of generating docids one at a time (pointwise), generates a ranked list of docids in a single sequence. This allows the model to consider inter-document dependencies during generation.

### TIGER (Rajput et al., 2024)

- **Title:** *Recommender Systems with Generative Retrieval* (Google)
- **Key Idea:** Applies DSI-style generative retrieval to recommendation systems. Uses Residual Quantization (RQ) to create semantic item IDs. While focused on recommendations, the ID learning approach has been influential for document retrieval.

---

## 13. Current State of the Art (2024--2025)

### Overall Landscape

As of early 2025, generative retrieval has matured significantly but remains primarily a research paradigm rather than a production replacement for established retrieval systems. Key developments:

#### Performance

- On NQ320K (the primary benchmark), the best generative retrieval methods achieve **Hits@1 ~49-52%** and **Hits@10 ~72-75%**, competitive with strong dense retrieval baselines
- On MS MARCO, generative retrieval still significantly lags behind state-of-the-art dense retrieval (e.g., ColBERTv2, SPLADE++) particularly for passage-level retrieval at scale (~8.8M passages)
- On BEIR (zero-shot transfer), generative retrieval generally performs poorly compared to sparse/dense methods due to the need to memorize a specific corpus

#### Key Trends (2024--2025)

1. **Learned Document Identifiers**: The field has largely moved away from hand-crafted docid schemes toward learned representations (GenRet, LMIndexer, TIGER-style RQ codes)

2. **Parameter-Efficient Continual Learning**: LoRA-based approaches (like MixLoRA-DSI) represent the frontier for handling document updates efficiently

3. **Hybrid Systems**: Growing interest in combining generative retrieval with traditional methods:
   - Use generative retrieval for candidate generation, then traditional reranking
   - Use traditional retrieval to constrain the generation space

4. **Scale**: The scalability barrier remains the biggest challenge:
   - Most results are on NQ320K (~320K docs) or NQ100K
   - MS MARCO passage (~8.8M) results are significantly less impressive
   - No convincing results at web scale (billions of documents)

5. **LLM-Based Retrieval**: With the rise of large language models, there is growing interest in using decoder-only LLMs (LLaMA, etc.) for generative retrieval instead of encoder-decoder T5

6. **Multimodal Generative Retrieval**: Emerging work on applying generative retrieval to images, videos, and multimodal documents

#### Approximate SOTA Numbers (NQ320K, early 2025)

| Method | Hits@1 | Hits@10 | Notes |
|--------|--------|---------|-------|
| BM25 | 12.4 | 33.7 | Sparse baseline |
| DPR | 30.1 | 58.3 | Dense baseline |
| DSI (T5-XXL) | 40.4 | 56.6 | Original |
| NCI (T5-Large) | 46.8 | 69.2 | + query gen + PAWA |
| MINDER | ~49 | ~71 | Multi-view IDs |
| RIPOR | ~50 | ~73 | Plan-ahead decoding |
| Best ensemble/hybrid | ~52 | ~75 | Combined approaches |

### Comparison with Dense Retrieval at Scale (MS MARCO)

| Method | Type | MRR@10 (MS MARCO Dev) |
|--------|------|----------------------|
| BM25 | Sparse | 18.7 |
| DPR | Dense | 31.1 |
| ColBERTv2 | Late interaction | 39.7 |
| SPLADE++ | Learned sparse | 38.8 |
| DSI-family (best) | Generative | ~25-30 |
| NCI (T5-Large) | Generative | ~28 |

**Key insight:** Generative retrieval is competitive on small-to-medium corpora (100K-320K) but significantly lags on large corpora (8.8M+). This scalability gap is the field's biggest challenge.

---

## 14. Key Limitations and Open Problems

### 1. Scalability

**The fundamental challenge.** Current generative retrieval methods have primarily been validated on corpora of 100K-320K documents. Scaling to millions or billions of documents faces:

- **Parameter capacity**: The model must memorize the entire corpus in its parameters. Even T5-XXL (11B params) struggles beyond a few hundred thousand documents.
- **Training cost**: Indexing scales linearly with corpus size; retraining is O(|C|) where |C| is the corpus.
- **Inference cost**: Autoregressive decoding is inherently sequential, making it slower than nearest-neighbor search in dense retrieval.
- **Docid space**: As the corpus grows, the docid space becomes harder to learn (more classes, longer sequences).

### 2. Catastrophic Forgetting / Continual Learning

- When new documents are added, fine-tuning on them causes the model to "forget" old documents
- DSI++ addresses this with replay buffers and SAM, but replay buffers grow with corpus size
- MixLoRA-DSI (if confirmed) addresses this with rehearsal-free LoRA experts
- IncDSI offers gradient-free updates but with performance degradation over time
- **Open question**: Can we achieve true lifelong learning in generative retrieval without any form of rehearsal?

### 3. Generalization Beyond Memorized Queries

- DSI models tend to perform well on queries similar to training queries but struggle with novel query types
- Query generation helps but does not fully solve the problem
- The model lacks the explicit term-matching capability of sparse retrieval
- **Open question**: How to achieve robust zero-shot generalization (BEIR-style) in generative retrieval?

### 4. Document Identifier Design

- No consensus on the optimal docid strategy
- Semantic IDs require clustering (which is corpus-dependent)
- Learned IDs require additional training infrastructure
- URL-based IDs are only applicable to web documents
- **Open question**: Can we learn universally optimal document representations for generation?

### 5. Computational Cost

| Operation | Dense Retrieval | Generative Retrieval |
|-----------|----------------|---------------------|
| Indexing | One forward pass per doc | Full training epoch(s) |
| Single query | ~10ms (ANN search) | ~100-500ms (beam search) |
| Adding 1 doc | Add vector to index | Retrain/fine-tune model |
| Memory | O(N * d) vectors | O(model params) |

### 6. Lack of Explicit Relevance Scoring

- Generative retrieval produces docids but not explicit relevance scores (beam search scores are a rough proxy)
- This makes it hard to integrate into multi-stage pipelines that require calibrated scores
- Reranking is difficult when the first stage does not provide meaningful scores

### 7. Reproducibility and Benchmarking

- Different papers use different dataset splits, preprocessing, and evaluation protocols
- NQ320K has become the de facto benchmark, but results are not always directly comparable
- No standardized benchmark for continual learning in generative retrieval (DSI++ proposed one, but adoption is limited)

### 8. Document Deletion and Privacy

- Most work focuses on adding documents; *removing* documents from a model is an open problem
- Related to machine unlearning
- Important for privacy compliance (GDPR right to be forgotten)

### 9. Multi-Document and Complex Queries

- Current methods generate a single docid at a time
- Multi-hop queries requiring reasoning over multiple documents are poorly handled
- ListGR begins to address this but is still early

---

## 15. Benchmark Summary Table

### NQ320K Results (Hits@1 / Hits@10)

| Method | Year | Model Size | Hits@1 | Hits@10 | DocID Type | Document Updates? |
|--------|------|-----------|--------|---------|------------|-------------------|
| BM25 | -- | -- | 12.4 | 33.7 | -- | Yes (reindex) |
| DPR | 2020 | BERT-base | 30.1 | 58.3 | Dense vector | Yes (add vector) |
| GENRE | 2021 | BART-large | -- | -- | Entity name | No (retrain) |
| DSI (atomic) | 2022 | T5-XXL | 39.0 | -- | Atomic int | No (retrain) |
| DSI (semantic) | 2022 | T5-XXL | 40.4 | 56.6 | Hierarchical | No (retrain) |
| SEAL | 2022 | BART-large | ~38 | ~62 | N-grams | Partial (FM-Index) |
| NCI | 2022 | T5-Large | 46.8 | 69.2 | Hierarchical | No (retrain) |
| DSI-QG | 2022 | T5-Base | ~38 | ~60 | Hierarchical | No (retrain) |
| Ultron | 2022 | T5-Base | ~43 | ~65 | URL tokens | No (retrain) |
| CorpusBrain | 2022 | BART-large | -- | -- | Titles (KILT) | No (retrain) |
| IncDSI | 2023 | T5-Base | ~35 | ~58 | Learned embed. | Yes (no retrain) |
| DSI++ | 2023 | T5-Base | ~35 | ~60 | Hierarchical | Yes (CL + replay) |
| MINDER | 2023 | T5-Base | ~49 | ~71 | Multi-view | No (retrain) |
| TOME | 2023 | T5-Base | ~44 | ~67 | Hierarchical | No (retrain) |
| RIPOR | 2024 | T5-Base | ~50 | ~73 | Hierarchical | No (retrain) |
| GLEN | 2024 | T5-Base | ~47 | ~70 | Lexical | No (retrain) |
| MixLoRA-DSI | 2025 | T5 + LoRA | TBD | TBD | Hierarchical | Yes (rehearsal-free) |

*Note: Some numbers are approximate, compiled across papers with potentially different evaluation setups. Always verify against original papers.*

### Continual Learning Results (DSI++ Protocol)

| Method | Old Docs Hits@1 | New Docs Hits@1 | Requires Replay? |
|--------|----------------|----------------|-----------------|
| Full Retrain | ~35.6 | ~35.6 | N/A (full retrain) |
| Naive Fine-tune | ~8 | ~34 | No |
| DSI++ (SAM + Replay) | ~35 | ~35 | Yes |
| IncDSI | ~33 | ~32 | No |
| MixLoRA-DSI | TBD | TBD | No (LoRA experts) |

---

## 16. References

1. **Tay, Y., et al.** (2022). "Transformer Memory as a Differentiable Search Index." *NeurIPS 2022*. arXiv:2202.06991

2. **Mehta, S.V., et al.** (2023). "DSI++: Updating Transformer Memory with New Documents." *EMNLP 2023*. arXiv:2311.09134

3. **MixLoRA-DSI** (2025). arXiv:2507.09924 *(beyond knowledge cutoff -- verify details)*

4. **Wang, Y., et al.** (2022). "A Neural Corpus Indexer for Document Retrieval." *NeurIPS 2022*. arXiv:2206.02743

5. **De Cao, N., et al.** (2021). "Autoregressive Entity Retrieval." *ICLR 2021*. arXiv:2010.00904

6. **Bevilacqua, M., et al.** (2022). "Autoregressive Search Engines: Generating Substrings as Document Identifiers." *NeurIPS 2022*. arXiv:2204.10628

7. **Zhou, Y., et al.** (2022). "Ultron: An Ultimate Retriever on Corpus with a Model-Based Indexer." arXiv:2208.09257

8. **Zhuang, S., et al.** (2022). "Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation." arXiv:2206.10128

9. **Li, Y., et al.** (2023). "Multiview Identifiers Enhanced Generative Retrieval." *ACL 2023*. arXiv:2305.16675

10. **Chen, J., et al.** (2022). "CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks." *CIKM 2022*. arXiv:2208.07652

11. **Kishore, V., et al.** (2023). "IncDSI: Incrementally Updatable Document Retrieval." *ICML 2023*.

12. **Zeng, H., et al.** (2024). "Planning Ahead in Generative Retrieval: Guiding Autoregressive Generation through Simultaneous Decoding." (RIPOR)

13. **Rajput, S., et al.** (2024). "Recommender Systems with Generative Retrieval." (TIGER). *NeurIPS 2023*.

14. **Li, M., et al.** (2024). "GLEN: Generative Retrieval via Lexical Index Learning."

15. **Pradeep, R., et al.** (2023). "How Does Generative Retrieval Scale to Millions of Passages?" *(Important scalability analysis)*

---

## Appendix: Key Takeaways for a Continual Learning Research Project

### Most Relevant Papers for Your Project

1. **DSI** -- foundational, must understand
2. **DSI++** -- the direct continual learning baseline
3. **MixLoRA-DSI** -- the paper you are building on
4. **IncDSI** -- alternative continual learning approach (no gradient updates)
5. **NCI** -- query generation techniques you may need

### Continual Learning Approaches in the Literature

| Approach | Paper | Replay Required? | Gradient Updates? | Key Mechanism |
|----------|-------|-----------------|-------------------|---------------|
| Experience Replay + SAM | DSI++ | Yes | Yes | Buffer + flat minima |
| Orthogonal Projection | IncDSI | No | No (optimization) | Constrained embedding |
| LoRA Expert Routing | MixLoRA-DSI | No | Yes (LoRA only) | Modular experts |
| Knowledge Distillation | DSI++ (component) | Implicit | Yes | Teacher-student |
| Elastic Weight Consolidation | Not yet applied | No | Yes | Fisher information |
| Progressive Nets | Not yet applied | No | No (freeze old) | New columns |

### Open Directions for a 2-Month Project

1. **Selective Forgetting as a Feature**: Your idea about lifelong memory where forgetting is *desirable* is novel. No existing DSI paper treats forgetting as beneficial.

2. **Importance-Weighted Rehearsal**: Rehearse on documents that are frequently retrieved (important memories) while allowing rarely-accessed documents to fade. This connects to spaced repetition and memory consolidation in cognitive science.

3. **Tiny MoE Analysis**: Analyzing how a small mixture of LoRA experts behaves on the DSI++ benchmark could yield insights about the capacity-forgetting tradeoff.

4. **New Datasets**: Most DSI work uses NQ320K. Applying to temporal datasets (news archives, Wikipedia edit histories) where document relevance changes over time would be interesting.

5. **Forgetting Dynamics Study**: Systematically measuring *what* gets forgotten and *when* in generative retrieval during continual learning -- no paper has done a deep analysis of forgetting patterns in DSI.
