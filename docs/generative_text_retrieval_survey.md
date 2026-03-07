# Generative Retrieval of Text Content: When Models Generate Memories as Passages, Not IDs

## A Survey of Parametric Knowledge Retrieval, Recitation, and Reconstructive Memory

**Prepared for:** Graduate Research Project on Continual Learning in Generative Retrieval
**Date:** 2026-02-23
**Scope:** Literature on models that generate actual text content (passages, answers, documents) from parameters, as opposed to generating document identifiers. Covers closed-book QA, recitation-augmented models, generative context creation, knowledge neuron localization, reconstructive memory in cognitive science, and the implications for continual learning.

---

## Why This Document Exists

The companion survey (`dsi_generative_retrieval_survey.md`) covers generative retrieval systems that generate **document IDs** -- DSI, NCI, GENRE, SEAL, etc. Those systems answer the question: "Given a query, which document is relevant?" by producing an identifier.

This document covers a fundamentally different paradigm: systems that generate the **actual text content** -- passages, answers, reconstructed documents -- directly from their parameters. These systems answer the question: "Given a query, what does the relevant passage say?" by producing the text itself.

The distinction matters enormously for continual learning and lifelong memory:
- In DSI, forgetting manifests as **generating wrong document IDs** (pointing to the wrong item)
- In text-generative retrieval, forgetting manifests as **degraded, hallucinated, or altered text** (the memory itself changes)
- The latter is more analogous to how human memory works: we do not lose a "pointer" to a memory; the memory itself transforms, blurs, or reconstructs differently over time

---

## Table of Contents

1. [Taxonomy: ID-Generative vs. Text-Generative Retrieval](#1-taxonomy-id-generative-vs-text-generative-retrieval)
2. [Closed-Book QA and Parametric Knowledge](#2-closed-book-qa-and-parametric-knowledge)
3. [RECITE: Recitation-Augmented Language Models](#3-recite-recitation-augmented-language-models)
4. [GenRead: Generate Rather Than Retrieve](#4-genread-generate-rather-than-retrieve)
5. [Self-RAG and Self-Generated Context](#5-self-rag-and-self-generated-context)
6. [Knowledge Neurons and Parametric Knowledge Storage](#6-knowledge-neurons-and-parametric-knowledge-storage)
7. [Generative and Reconstructive Memory in Cognitive Science](#7-generative-and-reconstructive-memory-in-cognitive-science)
8. [Memorization, Verbatim Recall, and Copy Mechanisms](#8-memorization-verbatim-recall-and-copy-mechanisms)
9. [Continual Learning, Forgetting, and Parametric Text Memory](#9-continual-learning-forgetting-and-parametric-text-memory)
10. [Synthesis: Text Generation as Memory Retrieval](#10-synthesis-text-generation-as-memory-retrieval)
11. [Implications for the DSI-CL Project](#11-implications-for-the-dsi-cl-project)
12. [References](#12-references)

---

## 1. Taxonomy: ID-Generative vs. Text-Generative Retrieval

### The Spectrum of Generative Retrieval

Generative retrieval can be placed on a spectrum from generating minimal identifiers to generating full content:

| Approach | What is Generated | Example | Knowledge Location |
|----------|-------------------|---------|-------------------|
| **Atomic ID** | Single token (document number) | DSI with atomic docids | Mapping stored in weights; content in external corpus |
| **Structured ID** | Multi-token identifier (hierarchical code) | DSI with semantic IDs, TIGER | Mapping stored in weights; content in external corpus |
| **Textual ID** | Document title or URL | GENRE, Ultron | ID is meaningful text, but still a pointer |
| **N-gram / Substring** | Fragment of document text | SEAL | Bridge: generates actual document content, but only fragments |
| **Short answer** | Brief factual answer | Closed-book QA (T5, GPT) | Answer extracted from parametric knowledge |
| **Recited passage** | Multi-sentence passage from memory | RECITE | Full passage reconstructed from parameters |
| **Generated document** | Complete synthetic document | GenRead | Novel text that resembles what a relevant document would contain |
| **Self-generated context** | Context + reflection tokens | Self-RAG | Model generates its own retrieval context |

**Key insight**: As we move down this spectrum, the model's parameters must store increasingly rich representations -- not just "which document" but "what the document says." This has profound implications for:
- **Capacity**: Text generation requires far richer parametric representations than ID generation
- **Forgetting**: Degradation is gradual and content-level (text becomes less accurate) rather than discrete (wrong ID)
- **Faithfulness**: The model may generate plausible but incorrect text (hallucination) rather than pointing to a wrong but real document
- **Continual learning**: New knowledge can blend with and modify old knowledge at the content level, not just reassign pointers

### The Content Generation vs. Pointer Generation Dichotomy

| Dimension | ID-Generative (DSI) | Text-Generative (This Survey) |
|-----------|---------------------|-------------------------------|
| **Output** | Document identifier | Text passage / answer |
| **Verification** | Can check if ID is valid | Cannot easily verify text accuracy |
| **Forgetting mode** | Wrong pointer (catastrophic) | Degraded/altered text (gradual) |
| **Hallucination** | Non-existent IDs | Plausible but fabricated content |
| **Capacity scaling** | IDs are compact | Full text requires much more capacity |
| **Human memory analog** | Losing a library card number | Misremembering what a book said |
| **Update mechanism** | Change ID mapping | Change stored knowledge directly |
| **Compositionality** | IDs are atomic | Text can be partially correct |

---

## 2. Closed-Book QA and Parametric Knowledge

### 2.1 The Foundational Insight: Knowledge in Parameters

#### "How Much Knowledge Can You Pack Into the Parameters of a Language Model?"

- **Title**: "How Much Knowledge Can You Pack Into the Parameters of a Language Model?"
- **Authors**: Adam Roberts, Colin Raffel, Noam Shazeer
- **Year**: 2020
- **Venue**: EMNLP 2020
- **arXiv**: 2002.08910
- **Affiliation**: Google Research

**Key Insight**: This paper demonstrated that a pre-trained T5 model, fine-tuned on question-answer pairs *without access to any external knowledge source at test time*, can answer factual questions competitively with open-book systems that have access to retrieval. The model's parameters serve as an implicit knowledge base.

**Technical Details**:
- Fine-tuned T5-11B on Natural Questions, WebQuestions, and TriviaQA in a closed-book setting
- Input: question text only (no retrieved passages, no context)
- Output: direct answer text
- Achieved 34.5% exact match on NQ (open-domain), 37.4% on WebQuestions, 50.1% on TriviaQA
- These scores approached or matched contemporary open-book systems that used explicit retrieval (e.g., the original DPR + reader pipeline)

**Results Comparison**:

| System | NQ (EM) | WebQ (EM) | TriviaQA (EM) | Uses Retrieval? |
|--------|---------|-----------|---------------|-----------------|
| T5-11B Closed-Book | 34.5 | 37.4 | 50.1 | No |
| T5-11B + SSM (Closed-Book) | 36.6 | 44.7 | 60.5 | No |
| BM25 + BERT Reader | 26.5 | -- | 47.1 | Yes |
| DPR + Reader (2020) | 41.5 | -- | 56.8 | Yes |
| REALM | 40.4 | -- | -- | Yes |

**Scaling behavior**: Performance scaled log-linearly with model size. T5-Small (60M) achieved only ~8% on NQ, while T5-XXL (11B) reached 34.5%. This suggests knowledge storage capacity grows with model parameters, but not proportionally.

**Relation to text-generative retrieval**: This paper established the foundational paradigm: the model's weights ARE the knowledge base, and generating an answer IS retrieving from that knowledge base. The model does not produce an ID pointing to Wikipedia; it generates the text of the answer directly from what it has "memorized" in its parameters.

**Limitations acknowledged**:
- Closed-book models struggle with rare/long-tail facts (seen infrequently during pre-training)
- No mechanism to update knowledge without retraining
- Cannot cite sources or verify answers
- Performance degrades sharply for questions requiring precise numerical or temporal information

---

#### "Language Models as Knowledge Bases?"

- **Title**: "Language Models as Knowledge Bases?"
- **Authors**: Fabio Petroni, Tim Rocktaschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel
- **Year**: 2019
- **Venue**: EMNLP 2019
- **arXiv**: 1909.01066
- **Affiliation**: Facebook AI Research (FAIR), University College London

**Key Insight**: This earlier paper (preceding Roberts et al.) probed whether pre-trained language models like BERT already contain relational knowledge by testing them on cloze-style knowledge queries (e.g., "Dante was born in [MASK]"). Found that BERT-large could answer a surprising fraction of relational queries correctly without any fine-tuning, sometimes matching or outperforming knowledge base completion methods that explicitly store facts.

**LAMA Probe**: Introduced the LAMA (LAnguage Model Analysis) benchmark -- a set of factual queries expressed as cloze sentences derived from knowledge bases (Wikidata, ConceptNet, etc.).

**Results**: BERT-large achieved ~30-50% precision@1 on various LAMA subsets, demonstrating substantial factual knowledge encoded in pre-training.

**Relation to text-generative retrieval**: Established that language model parameters implicitly store factual knowledge accessible through generation. The generation IS the retrieval -- the model generates the missing word/phrase from its parametric storage. This was one of the earliest demonstrations that neural network weights can serve as a "soft" knowledge base.

---

#### "UnifiedQA: Crossing Format Boundaries with a Single QA System"

- **Title**: "UnifiedQA: Crossing Format Boundaries with a Single QA System"
- **Authors**: Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, Hannaneh Hajishirzi
- **Year**: 2020
- **Venue**: EMNLP 2020 (Findings)
- **arXiv**: 2005.00700

**Key Insight**: Trained a single T5 model on multiple QA formats (extractive, abstractive, multiple-choice, yes/no) and demonstrated strong cross-format generalization. In its closed-book variant, the model generates answers purely from parametric knowledge, showing that a unified text-to-text framework can serve as a general parametric knowledge retrieval system.

**Relation to text-generative retrieval**: Demonstrates that parametric knowledge retrieval is robust across question formats. The model generates appropriate answer text regardless of whether the expected output is a single entity, a sentence, or a yes/no judgment.

---

### 2.2 Scaling Laws for Parametric Knowledge

#### "Scaling Data-Constrained Language Models"

- **Authors**: Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, Colin Raffel
- **Year**: 2023
- **Venue**: NeurIPS 2023

**Key Insight**: Established scaling laws for how language models absorb knowledge from training data. Relevant finding: repeating training data multiple epochs leads to diminishing returns for knowledge acquisition, suggesting that parametric knowledge storage has inherent capacity limits related to both model size and data diversity.

#### "Large Language Models Struggle to Learn Long-Tail Knowledge"

- **Title**: "Large Language Models Struggle to Learn Long-Tail Knowledge"
- **Authors**: Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel
- **Year**: 2023
- **Venue**: ICML 2023
- **arXiv**: 2211.08411

**Key Insight**: Demonstrated that LLMs can answer questions about entities/facts proportionally to how many times those facts appeared in the pre-training data. Facts seen <5 times are almost never answerable in closed-book settings. This establishes a "forgetting threshold" -- knowledge that does not receive sufficient repetition during training is effectively never stored.

**Relation to text-generative retrieval**: Provides the first empirical scaling law for parametric knowledge capacity. For a text-generative memory system, this implies:
- Frequently encountered information will be reliably generated
- Rare information will be poorly stored and quickly degraded
- There is a natural frequency-based importance weighting built into parametric storage
- This parallels human memory: frequently rehearsed memories are stronger

---

### 2.3 Comparing Open-Book vs. Closed-Book Approaches

#### "When Not to Trust Language Models"

- **Title**: "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories"
- **Authors**: Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, Hannaneh Hajishirzi
- **Year**: 2023
- **Venue**: ACL 2023
- **arXiv**: 2212.10511

**Key Insight**: Systematically compared parametric (closed-book) vs. non-parametric (retrieval-augmented) approaches across entity popularity. Key finding: parametric models are reliable for popular entities but fail on long-tail entities, while retrieval-augmented models maintain more uniform accuracy across popularity levels.

**Critical data point**: For the top 1% most popular entities, GPT-3.5 closed-book achieved ~80% accuracy. For the bottom 20%, accuracy dropped to ~15%. RAG maintained ~50-60% across all popularity levels.

**Relation to text-generative retrieval and forgetting**: This directly maps to memory dynamics -- parametric knowledge is inherently biased toward frequent/popular information. In a lifelong memory system using text generation as retrieval, memories that are frequently relevant will be well-preserved, while rare memories will degrade. This is not a bug -- it mirrors human memory's natural prioritization.

---

## 3. RECITE: Recitation-Augmented Language Models

### 3.1 Core Paper

- **Title**: "Recitation-Augmented Language Models"
- **Authors**: Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, Denny Zhou
- **Year**: 2023
- **Venue**: ICLR 2023
- **arXiv**: 2210.01296
- **Affiliation**: Carnegie Mellon University, Google Research

### 3.2 The Key Idea: Recite-Then-Answer

RECITE introduces a two-step paradigm that makes the connection between text generation and memory retrieval explicit:

1. **Recitation step**: Given a question, the model first generates (recites) one or more relevant passages from its parametric memory. These passages are NOT retrieved from an external corpus -- they are generated by the model from its weights.
2. **Answer step**: The recited passages are then used as context to answer the question, similar to how RAG uses retrieved passages.

```
Traditional RAG:     Query -> [External Retriever] -> Retrieved Passages -> [Reader] -> Answer
RECITE:              Query -> [LLM Recitation] -> Recited Passages -> [LLM Reader] -> Answer
Closed-Book:         Query -> [LLM] -> Answer (no intermediate passages)
```

### 3.3 Technical Details

**Prompting strategy**: RECITE uses few-shot prompting to elicit recitation. The prompt contains examples of the format:
```
Question: [Q]
Background Passage: [Passage from memory]
Answer: [A]
```

The model is prompted with a new question and asked to first generate the "Background Passage" before producing the answer.

**Recitation diversity**: The paper uses sampling (temperature > 0) to generate multiple diverse recitations for the same question, then uses majority voting across the resulting answers. This is analogous to retrieving multiple passages and aggregating evidence.

**Self-consistency**: Combined with self-consistency (Wang et al., 2022), where multiple reasoning chains are sampled and the most common answer is selected.

### 3.4 Results

Evaluated on Natural Questions, TriviaQA, and HotpotQA:

| Method | NQ (EM) | TriviaQA (EM) | HotpotQA (EM) |
|--------|---------|---------------|----------------|
| PaLM-62B (Direct) | 23.7 | 71.2 | 21.1 |
| PaLM-62B (Chain-of-Thought) | 25.0 | 71.8 | 28.5 |
| **PaLM-62B (RECITE)** | **28.9** | **74.8** | **30.3** |
| PaLM-540B (Direct) | 29.3 | 81.4 | 25.5 |
| **PaLM-540B (RECITE)** | **34.2** | **83.4** | **33.6** |
| Code-davinci-002 (Direct) | 29.8 | 72.7 | 28.2 |
| **Code-davinci-002 (RECITE)** | **37.2** | **78.6** | **37.8** |

**Key finding**: RECITE consistently and significantly outperformed direct closed-book answering across all models and datasets. The recitation step forces the model to first "retrieve" relevant information from its parameters in a structured way, making the implicit parametric knowledge explicit.

### 3.5 Analysis of Recited Passages

The paper provides fascinating analysis of what the model "recites":

- **Passage quality**: A substantial fraction of recited passages closely match actual Wikipedia passages (40-60% have high ROUGE overlap with ground-truth evidence passages)
- **Creative reconstruction**: Many recitations are not verbatim copies but reconstructions -- the model blends information from multiple sources and paraphrases
- **Hallucinated recitations**: Some recitations contain fabricated but plausible-sounding information, which can lead to wrong answers
- **Recitation diversity**: Sampling multiple recitations increases the chance that at least one contains the correct information

### 3.6 Relation to Text-Generative Retrieval

RECITE is the most explicit demonstration of **text generation as memory retrieval**:
- The model literally "recites" passages from its parametric memory
- These recitations are imperfect reconstructions, not exact copies -- mirroring human memory
- The quality of recitation degrades for rare/infrequent information
- The two-step process (recite then answer) decomposes parametric retrieval into an observable intermediate step

**For the DSI-CL project**: RECITE demonstrates that a model can be prompted to *externalize* its parametric memories as text. In a continual learning setting:
- As memories are overwritten by new training, recitation quality would degrade -- providing a measurable signal of forgetting
- Recitation diversity (via sampling) could help surface partially-forgotten memories that are still partially accessible
- The recitation step could serve as a diagnostic: "recite what you know about X" before answering

---

## 4. GenRead: Generate Rather Than Retrieve

### 4.1 Core Paper

- **Title**: "Generate rather than Retrieve: Large Language Models are Strong Context Generators"
- **Authors**: Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Sandeep Subramanian, Chenguang Zhu, Michael Zeng
- **Year**: 2023
- **Venue**: ICLR 2023
- **arXiv**: 2209.10063
- **Affiliation**: University of Virginia, Microsoft Research, Amazon

### 4.2 The Key Idea: LLM as Context Generator

GenRead proposes replacing the retriever in a RAG pipeline with a large language model that *generates* a relevant document given the question:

```
Traditional RAG:     Query -> [Retriever (BM25/DPR)] -> Retrieved Document -> [Reader] -> Answer
GenRead:             Query -> [LLM Generator] -> Generated Document -> [Reader] -> Answer
```

The generated document is synthetic -- it does not exist in any corpus. The LLM creates a plausible document that would contain the answer, drawing on its parametric knowledge.

### 4.3 Technical Details

**Generation prompt**: Given a question Q, GenRead prompts the LLM with:
```
Generate a background document from Wikipedia to answer the given question.
Question: {Q}
Document:
```

**Clustering-based prompting**: GenRead also proposes clustering training questions and creating cluster-specific prompts to improve generation quality for different question types.

**Reader model**: The generated documents are fed to a separate reader model (FiD or a simple extractive reader) that produces the final answer. Multiple generated documents can be used (analogous to retrieving top-k passages).

### 4.4 Results

| Method | NQ (EM) | TriviaQA (EM) | WebQ (EM) |
|--------|---------|---------------|-----------|
| BM25 + FiD | 44.8 | 65.6 | -- |
| DPR + FiD | 51.4 | 73.2 | -- |
| **GenRead (InstructGPT + FiD)** | **54.4** | **75.8** | **54.0** |
| GenRead (Codex + FiD) | 52.2 | 71.6 | -- |

**Striking result**: GenRead with InstructGPT as the generator **outperformed** DPR-based retrieval on all benchmarks. A generated document (which is synthetic and may contain inaccuracies) provided better context for answering than a real retrieved document.

### 4.5 Why Does Generating Beat Retrieving?

The paper offers several explanations:

1. **Query-document alignment**: Generated documents are perfectly aligned with the question because the LLM created them to answer that specific question. Retrieved documents may contain relevant information but are not tailored to the query.

2. **Implicit multi-source synthesis**: The LLM synthesizes information from its entire pre-training corpus into a single coherent document, whereas retrieval returns a single passage from a single source.

3. **Coverage**: Generated documents tend to directly address the question, while retrieved documents may only tangentially contain the answer.

4. **Format consistency**: Generated documents follow a consistent format expected by the reader, reducing format mismatch between retriever and reader.

### 4.6 Analysis of Generated Documents

- **Factual accuracy**: ~70-80% of generated documents contain the correct answer to the question (vs. ~60-70% for top-1 DPR retrieval)
- **Hallucination risk**: Generated documents sometimes contain plausible but wrong facts surrounding the correct answer
- **Compositionality**: Generated documents often blend information from multiple real sources, creating novel text that does not exist in any single document
- **Length and structure**: Generated documents tend to be more concise and focused than retrieved Wikipedia paragraphs

### 4.7 Relation to Text-Generative Retrieval

GenRead takes the text-generative retrieval paradigm to its logical extreme: **the retrieved document does not need to exist at all**. The model's parametric knowledge is sufficient to *construct* a relevant document on-the-fly.

**For the DSI-CL project**: GenRead raises a fundamental question about what "memory" means in a parametric system:
- If the model can generate a plausible passage that answers the question correctly, does it matter whether that passage was ever stored as a specific document?
- In the context of lifelong memory: forgetting a specific passage is not necessarily a failure if the model can still generate an accurate reconstruction from its parametric knowledge
- This connects to the human memory concept of "gist memory" -- retaining the meaning/essence without the verbatim text

---

## 5. Self-RAG and Self-Generated Context

### 5.1 Self-RAG

- **Title**: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
- **Authors**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- **Year**: 2024
- **Venue**: ICLR 2024
- **arXiv**: 2310.11511
- **Affiliation**: University of Washington, IBM Research

**Key Idea**: Self-RAG trains a single LM to adaptively decide when to retrieve, what to retrieve, and how to critique the retrieved passages. The model uses special "reflection tokens" to self-assess:
- **[Retrieve]**: Should I retrieve? (yes/no)
- **[IsRel]**: Is the retrieved passage relevant?
- **[IsSup]**: Does the passage support my generation?
- **[IsUse]**: Is my overall response useful?

**How it relates to text-generative retrieval**: When the model decides NOT to retrieve (based on its self-assessment), it generates the answer purely from parametric knowledge -- effectively performing closed-book text generation. The model learns to distinguish between questions it can answer parametrically and questions requiring external retrieval. This is a hybrid that adaptively chooses between text-generative and retrieval-augmented modes.

**Training**: Self-RAG is trained on a dataset annotated with reflection tokens, using a critic model to provide training signal. The base model learns when its parametric knowledge is sufficient vs. when external retrieval is needed.

**Results**: Self-RAG outperformed both vanilla RAG and ChatGPT on multiple benchmarks, with the key advantage being selective retrieval -- it avoids the "lost in the middle" problem by only retrieving when necessary and critically evaluating retrieved content.

### 5.2 SKR: Self-Knowledge Guided Retrieval

- **Title**: "Self-Knowledge Guided Retrieval Augmentation for Large Language Models"
- **Authors**: Yile Wang, Peng Li, Maosong Sun, Yang Liu
- **Year**: 2023
- **arXiv**: 2310.05002

**Key Idea**: Before retrieving, the model first determines whether it already knows the answer from its parameters. If the model's parametric knowledge is sufficient (high confidence in its generated answer), it skips retrieval entirely. If uncertain, it retrieves.

**Relation to text-generative retrieval**: SKR explicitly models the boundary between parametric and non-parametric knowledge. The model must assess "can I generate the answer from my parameters?" -- making parametric text generation a first-class citizen in the retrieval pipeline.

### 5.3 Self-Memory and Iterative Refinement

#### "Demonstrate-Search-Predict" (DSP)

- **Title**: "Demonstrate-Search-Predict: Composing retrieval and language model modules for knowledge-intensive NLP"
- **Authors**: Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, Matei Zaharia
- **Year**: 2023

**Key Idea**: Decomposes knowledge-intensive tasks into modular steps: demonstrate (generate examples), search (retrieve), predict (answer). The "demonstrate" step involves the model generating its own examples from parametric knowledge, which then guide retrieval.

#### "Selfmem: An Iterative Self-Memory Framework"

- **Title**: "Selfmem: An Iterative Self-Memory Framework for Text Generation"
- **Authors**: Various
- **Year**: 2023

**Key Idea**: Uses the model's own previous outputs as "self-memories" that are retrieved and used as context for subsequent generation. The model builds up an iterative memory from its own generations, creating a feedback loop between parametric knowledge and in-context learning.

**Relation to text-generative retrieval**: These systems blur the line between retrieval and generation -- the model retrieves from its own previous generations, creating a self-referential memory system that operates entirely on generated text.

---

## 6. Knowledge Neurons and Parametric Knowledge Storage

This section addresses the mechanistic question: **How** do transformers store the factual knowledge that enables text-generative retrieval? Where in the weights are "memories" located?

### 6.1 Knowledge Neurons in Pretrained Transformers

- **Title**: "Knowledge Neurons in Pretrained Transformers"
- **Authors**: Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei
- **Year**: 2022
- **Venue**: ACL 2022
- **arXiv**: 2104.08696
- **Affiliation**: Peking University, Microsoft Research

**Key Insight**: Identified specific neurons in the feed-forward (MLP) layers of transformers that are responsible for expressing specific factual knowledge. These "knowledge neurons" activate strongly when the model processes inputs related to particular facts and suppressing them degrades the model's ability to generate those facts.

**Method**:
1. For a given factual query (e.g., "The capital of France is [MASK]"), compute the attribution of each intermediate neuron to the correct prediction
2. Use an integrated gradients-based attribution method to identify which MLP neurons contribute most
3. Validate by suppressing (zeroing) the identified neurons and observing performance degradation
4. Also validate by activating the neurons and observing that the associated fact becomes more likely even in unrelated contexts

**Key Findings**:
- Factual knowledge is primarily stored in the **MLP layers** (feed-forward networks), not in the attention layers
- Knowledge is somewhat **localized** -- a relatively small number of neurons encode each fact
- There is **redundancy** -- suppressing a single neuron rarely completely eliminates a fact; multiple neurons contribute
- Knowledge neurons for related facts (e.g., facts about the same entity) tend to cluster in the same layers
- Upper MLP layers store more factual knowledge than lower layers

**Relation to text-generative retrieval**: This provides the mechanistic basis for understanding how models store the knowledge they use for text generation. When a closed-book model generates "Paris" in response to "The capital of France is ___", it is activating specific knowledge neurons in MLP layers that encode this association. Understanding this mechanism is crucial for:
- Predicting what will be forgotten (which neurons are most vulnerable to overwriting)
- Understanding interference (new facts competing for the same neurons)
- Potentially implementing targeted memory editing or protection

### 6.2 ROME: Rank-One Model Editing

- **Title**: "Locating and Editing Factual Associations in GPT"
- **Authors**: Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
- **Year**: 2022
- **Venue**: NeurIPS 2022
- **arXiv**: 2202.05262
- **Affiliation**: MIT, Northeastern University, Technion

**Key Insight**: ROME demonstrates that factual associations in GPT-style models can be precisely located in specific MLP layers and edited with rank-one updates to the MLP weight matrices. A single fact (e.g., "The Eiffel Tower is located in Paris") can be changed to a new fact (e.g., "The Eiffel Tower is located in London") by modifying a small number of parameters.

**Method -- Causal Tracing**:
1. Corrupt the input embedding for the subject token ("Eiffel Tower") by adding noise
2. Observe how this corruption propagates through the network, degrading the output
3. Restore ("uncorrupt") hidden states one layer at a time to see which restoration most recovers the correct output
4. The layer where restoration has the strongest effect is where the factual association is stored

**Key Findings**:
- Factual associations are stored in the **MLP layers at an identifiable "critical layer"** (typically in the middle layers for GPT-2/GPT-J)
- The MLP acts as a key-value memory: the first MLP matrix encodes "keys" (subject representations) and the second MLP matrix encodes "values" (associated facts)
- A rank-one update to the value matrix can precisely edit a single fact
- The edit generalizes: changing "Eiffel Tower is in Paris" to "Eiffel Tower is in London" also causes the model to say "London" in paraphrased queries

**The MLP-as-Memory View**:

ROME formalized the understanding that MLP layers function as associative memories:
```
MLP(x) = W_out * sigma(W_in * x + b_in) + b_out

Where:
- W_in acts as "keys" (what subjects/patterns to respond to)
- W_out acts as "values" (what facts to produce)
- sigma is the activation function
- Each row of W_in is a "key" and the corresponding column of W_out is the "value"
```

This view directly parallels key-value stores used in retrieval systems, except the "retrieval" happens through matrix multiplication rather than explicit lookup.

### 6.3 MEMIT: Mass-Editing Memory in a Transformer

- **Title**: "Mass-Editing Memory in a Transformer"
- **Authors**: Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau
- **Year**: 2023
- **Venue**: ICLR 2023
- **arXiv**: 2210.07229
- **Affiliation**: MIT, Northeastern University, Technion

**Key Insight**: Extends ROME to edit thousands of facts simultaneously. While ROME edits one fact at a time (rank-one update), MEMIT distributes edits across multiple MLP layers to minimize interference. This is crucial for large-scale knowledge updates.

**Technical Innovation**:
- Spreads each edit across a range of "critical layers" rather than concentrating in a single layer
- Uses a least-squares optimization to find minimal parameter changes that achieve all desired edits
- Maintains a balance between editing new facts and preserving existing knowledge

**Results**: Successfully edited 10,000+ facts in GPT-J with minimal degradation of unrelated knowledge.

**Relation to text-generative retrieval**: MEMIT demonstrates that parametric knowledge can be surgically edited at scale. For a text-generative memory system, this means:
- Memories are not just passively stored but can be actively updated
- Knowledge editing is an alternative to retraining for updating specific memories
- The distributed nature of knowledge (across multiple layers) provides some resilience against forgetting
- But large-scale edits can cause interference, analogous to catastrophic forgetting

### 6.4 Other Knowledge Localization Work

#### "Transformer Feed-Forward Layers Are Key-Value Memories"

- **Title**: "Transformer Feed-Forward Layers Are Key-Value Memories"
- **Authors**: Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy
- **Year**: 2021
- **Venue**: EMNLP 2021
- **arXiv**: 2012.14913
- **Affiliation**: Tel Aviv University, Allen AI

**Key Insight**: Provided the first systematic evidence that transformer MLP layers function as key-value memories. Each neuron in the first linear layer acts as a "key" that matches certain input patterns, and the corresponding column in the second linear layer acts as a "value" that promotes certain output tokens. Analysis of individual neurons showed that they activate for semantically coherent sets of inputs and promote semantically related outputs.

#### "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space"

- **Title**: Same research line continuation
- **Authors**: Mor Geva, Avi Caciularu, Kevin Ro Wang, Yoav Goldberg
- **Year**: 2022
- **Venue**: EMNLP 2022

**Key Insight**: Extended the key-value memory view by showing that MLP updates (the output of each MLP layer) can be projected into vocabulary space to reveal that each layer "promotes" specific concepts/words. The model builds its prediction incrementally across layers, with each MLP contributing different aspects of the answer.

### 6.5 Summary: The Mechanistic Basis of Parametric Text Memory

| Finding | Source | Implication for Text-Generative Memory |
|---------|--------|---------------------------------------|
| MLP layers are key-value memories | Geva et al. (2021) | Memory is stored in specific architectural components |
| Knowledge neurons can be identified | Dai et al. (2022) | Individual facts can be attributed to specific neurons |
| Facts are localized in mid-layer MLPs | Meng et al. (2022) - ROME | Knowledge has a specific "address" in the network |
| Facts can be edited with rank-one updates | Meng et al. (2022) - ROME | Memories can be surgically modified |
| Mass editing across layers is possible | Meng et al. (2023) - MEMIT | Bulk memory updates are feasible |
| Knowledge builds incrementally across layers | Geva et al. (2022) | Memory retrieval is a progressive computation |

**For the DSI-CL project**: Understanding where knowledge lives in the network is essential for understanding how it is forgotten. If knowledge neurons for old memories are overwritten by new training, the old memories degrade at the text level (not the ID level). This is a fundamentally different forgetting mechanism than DSI's ID-level forgetting, and potentially more graceful -- the model may generate a partially correct answer rather than a completely wrong document ID.

---

## 7. Generative and Reconstructive Memory in Cognitive Science

### 7.1 Bartlett's Schema Theory: Memory as Reconstruction

- **Title**: "Remembering: A Study in Experimental and Social Psychology"
- **Author**: Sir Frederic C. Bartlett
- **Year**: 1932
- **Key Insight**: Bartlett's landmark work demonstrated that human memory is fundamentally **reconstructive**, not reproductive. In his famous "War of the Ghosts" experiment, participants read an unfamiliar Native American folk tale and were asked to recall it at various intervals. Their recollections systematically changed over time:
  - Details were **omitted** (especially unfamiliar or culturally strange elements)
  - The narrative was **rationalized** (modified to fit the participants' cultural schemas)
  - Elements were **transformed** (names changed, causal relationships reinterpreted)
  - The gist was **preserved** even as specific details shifted

**Schema theory**: Bartlett proposed that memories are stored not as exact copies but as **schemas** -- organized knowledge frameworks. Recall involves reconstructing the memory from these schemas, filling in gaps with plausible information consistent with the schema. Each act of remembering is an act of construction.

**Relation to text-generative retrieval**: This is the most direct cognitive science parallel to parametric text generation:
- A language model generating text from its parameters is performing **reconstruction** -- assembling a plausible passage from its learned "schemas" (distributed representations)
- Like human memory, the reconstruction may be accurate in gist but altered in detail
- "Hallucination" in LLMs mirrors confabulation in human memory -- filling gaps with plausible but inaccurate information
- **Forgetting is not binary** -- it is a gradual transformation of the memory, not a loss of a pointer

### 7.2 Constructive Memory and Source Monitoring

- **Title**: "Constructive Memory: Past and Future"
- **Authors**: Daniel L. Schacter, Donna Rose Addis
- **Year**: 2007
- **Venue**: Chapter in "Remembering: Attributions, Processes, and Control in Human Memory"

**Key Insight**: Extended Bartlett's ideas with modern neuroscience. Constructive memory theory posits that the same neural machinery used for remembering the past is used for imagining the future -- both involve constructing a coherent scene/narrative from stored elements. The hippocampus binds together elements (who, what, where, when) stored in cortical areas.

**Source monitoring**: Humans often confuse the source of a memory -- did I experience this, or did someone tell me, or did I imagine it? This is precisely the problem that text-generative retrieval faces: the model cannot distinguish between accurately recalled content and self-generated plausible content.

- **Title**: "Source Monitoring Framework"
- **Authors**: Marcia K. Johnson, Shahin Hashtroudi, D. Stephen Lindsay
- **Year**: 1993
- **Venue**: Psychological Bulletin

**Key Insight**: Memory for the source of information (reality monitoring, external source monitoring, internal source monitoring) is separate from memory for the content itself. People can remember *what* happened but misattribute *where* they learned it.

**Relation to text-generative retrieval**: LLMs face an extreme version of the source monitoring problem -- they generate text that blends genuine training data with plausible constructions, with no mechanism to distinguish the two. In a lifelong memory system, this means:
- The model may confidently generate a "memory" that is actually a plausible reconstruction
- There is no built-in way to assess whether generated text is accurately recalled vs. confabulated
- This parallels human "false memories" -- confidently held memories of events that did not occur

### 7.3 Episodic vs. Semantic Memory

- **Title**: "Episodic and Semantic Memory"
- **Author**: Endel Tulving
- **Year**: 1972
- **Venue**: Chapter in "Organization of Memory" (ed. Tulving & Donaldson)

**Key Insight**: Tulving's foundational distinction:
- **Episodic memory**: Memory for specific events bound to a particular time and place ("I had coffee with Sarah at Blue Bottle on Tuesday")
- **Semantic memory**: Memory for general facts and knowledge ("Coffee contains caffeine")

Episodic memories are rich in contextual detail and decay faster. Semantic memories are abstracted from specific episodes and are more durable.

**Relation to text-generative retrieval**:
- Pre-trained language models primarily encode **semantic memory** -- general knowledge extracted from the statistical patterns of text
- Fine-tuning on specific documents can encode something closer to **episodic memory** -- specific content from particular sources
- Forgetting in LLMs may follow different patterns for these two types:
  - Semantic knowledge (general facts) is highly redundant and resilient
  - Episodic knowledge (specific passages) is less redundant and more vulnerable to forgetting
- DSI aims to store something more episodic (specific documents), while closed-book QA exploits semantic knowledge

### 7.4 Memory Consolidation and Sleep

- **Title**: "About Sleep's Role in Memory"
- **Authors**: Jan Born, Bjorn Rasch, Steffen Gais
- **Year**: 2006
- **Venue**: Physiological Reviews

**Key Insight**: During sleep, the brain replays recent experiences (hippocampal replay), gradually transferring them from hippocampal (episodic) to neocortical (semantic) storage. This consolidation is **selective** -- emotionally significant, novel, or schema-relevant memories are preferentially consolidated. Unimportant memories are not consolidated and naturally decay.

**The transformation during consolidation**: Memories do not transfer as exact copies. During consolidation:
- Specific details are lost (episodic -> semantic transformation)
- Connections to existing knowledge are strengthened (schema integration)
- Gist is preserved while surface form changes (abstraction)
- Contradictory or schema-inconsistent elements may be modified or dropped

**Relation to text-generative retrieval**: If we train a model on specific documents (episodic indexing) and then continue training on new data (simulating "waking" experience followed by "sleep" consolidation), we would expect:
- Specific document text to become less exactly reproducible over time
- General facts from those documents to be more durable
- The model's "recitation" of a document to become more gist-like and less verbatim
- This is not a failure -- it mirrors the natural episodic-to-semantic transformation in human memory

### 7.5 Interference and Forgetting in Human Memory

#### Proactive and Retroactive Interference

- **Key concepts**:
  - **Retroactive interference**: New learning disrupts old memories (new facts overwrite old ones)
  - **Proactive interference**: Old learning interferes with new encoding (existing knowledge makes it harder to learn contradictory information)
  - Both are well-established phenomena in cognitive psychology (McGeogh, 1942; Underwood, 1957)

**Relation to text-generative retrieval**: These directly parallel catastrophic forgetting in neural networks:
- **Retroactive**: Fine-tuning on new documents degrades the model's ability to generate text about old documents (standard catastrophic forgetting)
- **Proactive**: A model strongly trained on one domain may resist learning conflicting information in a new domain (negative transfer / the "stability" side of stability-plasticity)

#### Retrieval-Induced Forgetting

- **Title**: "Retrieval-Induced Forgetting"
- **Authors**: Michael C. Anderson, Robert A. Bjork, Elizabeth L. Bjork
- **Year**: 1994
- **Venue**: Journal of Experimental Psychology: Learning, Memory, and Cognition

**Key Insight**: Practicing retrieval of some items from a category actively suppresses memory for related but unpracticed items. For example, rehearsing "fruits: orange, apple" makes it *harder* to later recall "fruits: banana, grape."

**Relation to text-generative retrieval**: In a lifelong memory system:
- Frequently rehearsing some documents (via selective replay) may actively suppress the model's ability to generate text about related but unrehearsed documents
- This is both a risk (accidentally forgetting important related memories) and an opportunity (rehearsing the most important items naturally suppresses less important related items)
- This maps to the project's "beneficial forgetting" thesis: selective rehearsal does not just preserve rehearsed memories, it actively clears away related noise

### 7.6 Summary: Cognitive Science Parallels

| Human Memory Phenomenon | LLM/Text-Generative Parallel | Implication |
|--------------------------|------------------------------|-------------|
| Reconstructive recall (Bartlett) | LLM generates plausible but potentially altered text | "Memories" are reconstructions, not exact copies |
| Schema-based distortion | LLM biased by training distribution | Generated text conforms to learned patterns |
| Source monitoring failures | Cannot distinguish recalled vs. generated text | Hallucination is the AI analog of confabulation |
| Episodic -> Semantic transformation | Fine-tuned knowledge -> general knowledge shift | Specific documents gradually become general knowledge |
| Sleep consolidation (selective) | Selective rehearsal during continual learning | Important memories are preferentially preserved |
| Retrieval-induced forgetting | Rehearsal of some items suppresses related items | Selective rehearsal has competitive dynamics |
| Gist memory | Model retains meaning but not exact wording | Forgetting surface form while retaining content |
| Interference (pro/retroactive) | Catastrophic forgetting / negative transfer | New knowledge disrupts old; old knowledge resists new |
| Ebbinghaus forgetting curve | Performance decay over time without rehearsal | Exponential decay of retrieval accuracy |

---

## 8. Memorization, Verbatim Recall, and Copy Mechanisms

### 8.1 Training Data Memorization in LLMs

#### "Quantifying Memorization Across Neural Language Models"

- **Title**: "Quantifying Memorization Across Neural Language Models"
- **Authors**: Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, Chiyuan Zhang
- **Year**: 2023
- **Venue**: ICLR 2023
- **arXiv**: 2202.07646
- **Affiliation**: Google, CMU, ETH Zurich

**Key Insight**: Systematically measured how much training data LLMs memorize verbatim. Key findings:
- **Memorization scales with model size**: Larger models memorize more training data. GPT-Neo 6B memorized ~10x more sequences than GPT-Neo 125M
- **Memorization scales with repetition**: Sequences that appear multiple times in the training data are much more likely to be memorized
- **Memorization is extractable**: Given a prefix from a training sequence, the model can often complete it verbatim
- **The "k-extractability" metric**: A sequence is k-extractable if providing k tokens of prefix causes the model to generate the remaining tokens verbatim

**Quantitative results**:
- GPT-Neo 6B: ~1% of test set sequences are memorized (extractable with a 50-token prefix)
- The fraction increases to ~5-10% for sequences appearing 10+ times in training
- Memorization is concentrated in "high-surprise" sequences (unique phrasing, proper nouns, code snippets)

**Relation to text-generative retrieval**: This establishes the *upper bound* of verbatim text generation from parameters. A model CAN store and reproduce exact text, but only for a small fraction of its training data (primarily frequently-seen sequences). For a lifelong memory system:
- Most "memories" will be reconstructions rather than verbatim recalls
- Frequently rehearsed memories may achieve near-verbatim recall
- This mirrors human memory: we can recite poems we have practiced, but most memories are reconstructive

### 8.2 Extracting Training Data from Language Models

- **Title**: "Extracting Training Data from Large Language Models"
- **Authors**: Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, Colin Raffel
- **Year**: 2021
- **Venue**: USENIX Security 2021
- **arXiv**: 2012.07805

**Key Insight**: Demonstrated practical attacks for extracting verbatim training data from GPT-2. Generated text from GPT-2 and identified sequences that exactly match training data (names, phone numbers, code snippets, URLs). Showed that language models are not merely learning statistical patterns -- they store and can reproduce specific training sequences.

**Relation to text-generative retrieval**: Proves that language models DO perform a form of "retrieval" from their training data, not just statistical generation. The boundary between "generating plausible text" and "recalling memorized text" is blurry and depends on the specificity and frequency of the training data.

### 8.3 Copy Mechanisms in Neural Networks

#### Pointer Networks

- **Title**: "Pointer Networks"
- **Authors**: Oriol Vinyals, Meire Fortunato, Navdeep Jaitly
- **Year**: 2015
- **Venue**: NeurIPS 2015

**Key Insight**: Introduced an attention-based mechanism that allows seq2seq models to "point to" (copy) tokens from the input sequence. Instead of generating output tokens from a fixed vocabulary, the model can directly copy input tokens to the output.

#### CopyNet

- **Title**: "Incorporating Copying Mechanism in Sequence-to-Sequence Learning"
- **Authors**: Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li
- **Year**: 2016
- **Venue**: ACL 2016

**Key Insight**: CopyNet combines a standard generative vocabulary with a copy mechanism that can reproduce tokens from the input. The model learns when to generate from the vocabulary and when to copy from the input. This is particularly useful for tasks that require reproducing specific entities or phrases from the input.

#### Pointer-Generator Networks

- **Title**: "Get To The Point: Summarization with Pointer-Generator Networks"
- **Authors**: Abigail See, Peter J. Liu, Christopher D. Manning
- **Year**: 2017
- **Venue**: ACL 2017

**Key Insight**: Combines pointing/copying with generation in a soft-switching framework. At each time step, the model computes a "generation probability" p_gen that determines whether to generate from the vocabulary or copy from the source. This allows faithful reproduction of key terms while maintaining fluent generation.

**Relation to text-generative retrieval**: Copy mechanisms represent an explicit architectural decision to enable text reproduction. In the context of text-generative memory:
- Standard transformers (without explicit copy mechanisms) must store everything in parameters
- Copy mechanisms provide a shortcut for verbatim recall by directly attending to stored content
- The lack of copy mechanisms in modern LLMs means they must "memorize" text through weight-based storage, leading to the reconstructive (rather than reproductive) memory behavior observed

### 8.4 Memorization vs. Generalization Tension

#### "Does Learning Require Memorization? A Short Tale about a Long Tail"

- **Title**: "Does Learning Require Memorization? A Short Tale about a Long Tail"
- **Authors**: Vitaly Feldman
- **Year**: 2020
- **Venue**: STOC 2020

**Key Insight**: Proved theoretically that for long-tailed distributions (where many examples are rare), memorization of individual training examples is *necessary* for good generalization. This is counterintuitive -- memorization and generalization are not opposites but complements. The model must memorize rare examples because their contribution to the loss cannot be captured by general patterns alone.

**Relation to text-generative retrieval**: For a text-generative memory system:
- Common/frequent information can be captured by general patterns (semantic memory)
- Rare/specific information requires memorization of individual instances (episodic memory)
- A model that only generalizes will fail on rare memories; one that only memorizes will fail on novel queries
- The optimal system balances both -- just like human memory balances episodic and semantic components

#### "Memorization Without Overfitting"

- **Title**: "What Can Neural Networks Reason About?"
- **Authors**: Keyulu Xu, Jingling Li, Mozhi Zhang, Simon S. Du, Ken-ichi Kawarabayashi, Stefanie Jegelka
- **Year**: 2020

And related work establishing that large models can memorize entire training sets while still generalizing well (the "double descent" phenomenon). This is relevant because it suggests that models can simultaneously serve as both generalizable reasoning engines and specific memory stores.

---

## 9. Continual Learning, Forgetting, and Parametric Text Memory

### 9.1 How Forgetting Manifests Differently in Text vs. ID Generation

This is a critical distinction for the DSI-CL project:

| Aspect | ID-Generative Forgetting (DSI) | Text-Generative Forgetting |
|--------|-------------------------------|---------------------------|
| **Failure mode** | Generates wrong document ID | Generates degraded/altered text |
| **Gradedness** | Binary (right ID or wrong ID) | Continuous (partially correct text) |
| **Diagnosability** | Easy (check if ID matches) | Hard (how "wrong" is the text?) |
| **Human analog** | Forgetting a library call number | Misremembering what a book said |
| **Graceful degradation** | No (wrong ID = complete failure) | Yes (text may retain gist) |
| **Hallucination risk** | Low (IDs are constrained) | High (text space is unconstrained) |
| **Evaluation** | Exact match on ID | ROUGE, BERTScore, factual accuracy |
| **Recovery** | Cannot partially recover | May recover gist from degraded text |

### 9.2 Continual Knowledge Learning for LLMs

- **Title**: "Towards Continual Knowledge Learning of Language Models"
- **Authors**: Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, Stanley Jungkyu Choi, Minjoon Seo
- **Year**: 2022
- **Venue**: ICLR 2022
- **arXiv**: 2110.03215

**Key Insight**: Studied how continued pre-training on new data causes LLMs to forget previously learned factual knowledge. Introduced the CKL (Continual Knowledge Learning) benchmark and showed that:
- Standard continued pre-training causes severe forgetting of rare facts (up to 40% accuracy drop)
- Frequently-seen facts are more resilient (5-10% drop)
- Knowledge distillation from the old model significantly reduces forgetting
- Replay of old data is the most effective strategy

**Forgetting patterns**: The paper documented that forgetting in LLMs follows a pattern where:
1. Rare facts are forgotten first
2. Facts encoded across multiple training contexts are more resilient
3. Facts that are "schema-consistent" (fitting general patterns) are preserved longer than "schema-inconsistent" (surprising/unusual) facts

**Relation to text-generative retrieval**: These forgetting patterns directly predict how a text-generative memory system would degrade:
- Frequently relevant memories will be well-preserved
- Rare or unusual memories will degrade first
- Memories consistent with the model's general knowledge will persist even without rehearsal
- Memories that contradict common patterns will be overwritten most readily

### 9.3 Knowledge Editing and Its Limits

#### "Can We Edit Factual Knowledge by In-Context Learning?"

- **Title**: "Can We Edit Factual Knowledge by In-Context Learning?"
- **Authors**: Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, Baobao Chang
- **Year**: 2023
- **arXiv**: 2305.12740

**Key Insight**: Explored whether in-context learning (providing corrected facts in the prompt) can override parametric knowledge. Found that:
- In-context corrections work for surface-level edits but often fail for deep inferences
- The model may generate the corrected fact when asked directly but revert to its parametric knowledge for follow-up reasoning
- Parametric knowledge has a "gravitational pull" -- the model tends to return to its trained beliefs

#### "Editing Large Language Models: Problems, Methods, and Opportunities"

- **Authors**: Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang
- **Year**: 2023
- **Venue**: EMNLP 2023

**Key Insight**: Comprehensive survey of knowledge editing methods. Identifies key challenges:
- **Specificity**: Edits should change the target fact without affecting unrelated facts
- **Generalization**: Edits should transfer to paraphrased queries
- **Portability**: Edits should affect downstream reasoning
- **Locality**: Edits should not cause cascading failures

**Relation to text-generative retrieval and continual learning**: Knowledge editing provides surgical tools for updating specific memories without full retraining. However, for a lifelong memory system with thousands of updates, per-fact editing may not scale. The interplay between:
- Per-fact editing (ROME/MEMIT for precise updates)
- Continual learning (batch updates via fine-tuning)
- Natural forgetting (gradual degradation without intervention)
defines the design space for text-generative memory systems.

### 9.4 Catastrophic Forgetting Specifically in Text Generation

- **Title**: "Continual Learning for Language Generation"
- **Authors**: Various research groups (2021-2024)

**Key observations specific to text generation**:

1. **Error compounding**: In autoregressive text generation, forgetting at one token position cascades through subsequent tokens. If the model forgets a key entity, the rest of the generated passage becomes increasingly incoherent. This is worse than in classification where each prediction is independent.

2. **Output distribution shift**: As the model learns new text patterns, its output distribution shifts. Old text patterns (characteristic of older training data) may be "smoothed out" by newer patterns. This manifests as the model's generated text becoming stylistically homogeneous over time.

3. **Vocabulary drift**: For models fine-tuned on domain-specific text, continual learning on new domains can shift vocabulary preferences. Medical terms might become less likely after extensive fine-tuning on legal text.

4. **Context sensitivity**: Forgetting in text generation is highly context-dependent. The model may still generate correct text when given strong contextual cues but fail with ambiguous or minimal prompting.

### 9.5 Elastic Weight Consolidation for Language Models

- **Title**: "Overcoming Catastrophic Forgetting in Neural Networks" (applied to language models)
- **Authors**: Kirkpatrick et al. (2017), with subsequent applications to LLMs

**When applied to text-generative models**, EWC reveals:
- The Fisher Information for text generation is distributed very unevenly -- attention head weights tend to have high Fisher information (important for syntactic patterns) while MLP weights have heterogeneous Fisher information (some encode specific facts, others encode general patterns)
- Protecting high-Fisher weights preserves the model's ability to generate fluent text but may not preserve specific factual content
- Protecting fact-specific MLP weights (identified via knowledge neuron techniques) may preserve specific memories but at the cost of plasticity

### 9.6 Progressive Memory Systems

#### Progressive Networks for Text (Conceptual)

While progressive neural networks (Rusu et al., 2016) have not been widely applied to text generation, the concept maps naturally:

- **Base column**: Pre-trained LLM with general knowledge
- **New columns**: Fine-tuned additions for specific document collections
- **Lateral connections**: Allow new columns to access base knowledge
- **Forgetting**: Impossible within frozen columns; new knowledge can only be added

**The LoRA version**: In practice, LoRA adapters serve as lightweight "columns" for progressive text-generative memory:
- Base model contains general knowledge (semantic memory)
- Each LoRA adapter encodes knowledge from a specific time period or document set (episodic memory)
- Multiple adapters can be composed (analogous to lateral connections)
- Old adapters are frozen (no forgetting of their specific knowledge)
- This IS the MixLoRA-DSI approach, but applied to text generation rather than ID generation

---

## 10. Synthesis: Text Generation as Memory Retrieval

### 10.1 The Unified View

Across all the literature surveyed, a coherent picture emerges:

**Text generation from trained model parameters IS a form of memory retrieval.** The model stores knowledge distributed across its weights (primarily in MLP layers, as shown by knowledge neuron research) and "retrieves" this knowledge by generating text tokens that activate the stored associations.

This view unifies:
- **Closed-book QA** (Roberts et al.): Direct parametric retrieval of answer text
- **RECITE** (Sun et al.): Explicit recitation of stored passages before answering
- **GenRead** (Yu et al.): Generation of context documents from parametric knowledge
- **Self-RAG** (Asai et al.): Adaptive decision between parametric and external retrieval
- **Knowledge editing** (ROME/MEMIT): Surgical modification of specific stored memories
- **Memorization studies** (Carlini et al.): Evidence that models store and can reproduce specific training text

### 10.2 The Reconstruction Principle

The single most important insight from the cognitive science literature is: **both human memory and parametric text generation are fundamentally reconstructive processes, not reproductive ones.**

When a model generates text about a memorized topic:
1. The query activates relevant patterns in the network (attention + MLP key matching)
2. These patterns trigger value vectors that promote certain tokens (MLP value retrieval)
3. Autoregressive generation assembles these activations into coherent text (reconstruction)
4. The result is influenced by the model's general knowledge, not just the specific stored content (schema effects)
5. The generated text may be plausible but contain distortions or confabulations (source monitoring failures)

This is Bartlett's schema theory, realized in silicon.

### 10.3 The Forgetting Continuum

In text-generative retrieval, forgetting is not binary but exists on a continuum:

```
Full Recall                                                    Complete Forgetting
    |                                                                |
    |-- Verbatim reproduction (rare, requires high repetition)       |
    |-- Accurate paraphrase (gist preserved, wording changed)       |
    |-- Partial recall (some facts correct, some altered)            |
    |-- Schema-based reconstruction (plausible but inaccurate)       |
    |-- Hallucination (plausible but entirely fabricated)             |
    |-- Irrelevant generation (topic knowledge lost entirely)        |
    |                                                                |
```

This continuum is **more natural** and **potentially more useful** than the binary forgetting in DSI:
- DSI: either the correct ID or the wrong ID (no middle ground)
- Text generation: gradual degradation from verbatim to gist to confabulation
- The "partially correct" zone in text generation may still be useful for downstream tasks
- A recited passage that is 80% accurate is far more useful than a wrong document ID

### 10.4 Comparison: Text-Generative vs. ID-Generative Retrieval for Lifelong Memory

| Criterion | ID-Generative (DSI) | Text-Generative | Winner for Lifelong Memory |
|-----------|---------------------|-----------------|---------------------------|
| **Storage efficiency** | Compact (IDs are small targets) | Expensive (must store full text content) | DSI |
| **Graceful degradation** | None (wrong ID = total failure) | Continuous (text degrades gradually) | Text-Generative |
| **Verifiability** | Easy (check ID) | Hard (check text accuracy) | DSI |
| **Scalability** | Scales to ~320K docs (proven) | Scales to general knowledge (proven at ~1T tokens) | Text-Generative |
| **Forgetting naturalness** | Discrete, catastrophic | Gradual, human-like | Text-Generative |
| **Utility under forgetting** | Zero (wrong ID is useless) | Partial (degraded text may be useful) | Text-Generative |
| **Update mechanism** | Retrain / LoRA | ROME/MEMIT / continue training | Tie |
| **Continual learning studied?** | Yes (DSI++, IncDSI) | Partially (CKL, but not for retrieval) | DSI (more studied) |
| **Cognitive plausibility** | Low (humans don't store IDs) | High (humans reconstruct text/narratives) | Text-Generative |

---

## 11. Implications for the DSI-CL Project

### 11.1 Why This Matters for Your Project

Your project studies continual learning in DSI, which generates document IDs. But the text-generative retrieval paradigm offers important contrasts and insights:

1. **Alternative framing**: Instead of generating a docid, what if the model generated the document text itself? This would make forgetting a gradual, content-level phenomenon rather than a discrete pointer-level failure.

2. **Hybrid approach**: A DSI system could be augmented with a RECITE-like mechanism -- before generating the docid, the model recites what it "remembers" about the query topic. This recitation quality could serve as a diagnostic for forgetting severity.

3. **Forgetting metrics**: Text-generative retrieval suggests richer forgetting metrics:
   - ROUGE score between recited and original passages (surface-level retention)
   - BERTScore (semantic retention)
   - Factual accuracy of generated text (fact-level retention)
   - These complement the binary Hits@k metrics used in DSI

4. **Knowledge neuron analysis**: Understanding which neurons store which documents could predict which documents will be forgotten first during continual learning, and inform targeted rehearsal strategies.

### 11.2 Potential Extensions

#### Extension 1: RECITE-DSI Hybrid
Train the model to first recite what it knows about the query topic, then use the recitation as additional context for docid generation. During continual learning, monitor recitation quality as an early warning signal for forgetting.

#### Extension 2: Text-Quality Forgetting Curves
Beyond measuring Hits@k, measure the quality of text the model can generate about each indexed document over time. This gives a richer picture of forgetting:
- Does the model first lose exact wording, then facts, then gist?
- Does this follow Ebbinghaus curves?
- Is the text-quality forgetting curve more gradual than the ID-accuracy forgetting curve?

#### Extension 3: Knowledge Neuron-Guided Rehearsal
Use knowledge neuron identification to find which neurons encode which documents. During continual learning, monitor these neurons for overwriting and selectively protect or rehearse documents whose neurons are under threat.

#### Extension 4: GenRead-Style Memory Reconstruction
Instead of asking the model to generate a docid, ask it to generate a passage that would answer the query. If the generated passage is accurate (verified against the ground truth), credit the system with a "successful memory" even if it cannot produce the exact docid. This gives credit for gist-level retention.

### 11.3 The Bigger Picture: What Kind of Memory System Do We Want?

The literature in this survey points to a fundamental design choice:

| Design Choice | ID-Based (Current DSI) | Text-Based (This Survey) | Hybrid |
|---------------|------------------------|--------------------------|--------|
| **Output** | docid | Generated passage | docid + passage quality score |
| **Memory model** | Library catalog | Human reconstructive memory | Library with a librarian who remembers content |
| **Forgetting** | Lost card | Fading, distorted memory | Lost card + fading memory |
| **Evaluation** | Did you find the right book? | Do you remember what the book said? | Both |
| **Practical use** | Document retrieval | Knowledge assistance | Both |

For a **lifelong personal memory** system, the text-generative approach may actually be more appropriate:
- Users do not want document IDs; they want the content
- Graceful degradation (partially accurate recall) is better than total failure (wrong ID)
- The reconstructive nature of text generation naturally mirrors human memory
- But verifiability and precision favor ID-based systems

**The ideal system** might combine both: use DSI-style ID generation for precise retrieval, and use text-generative mechanisms for situations where IDs are forgotten but gist-level knowledge persists.

---

## 12. References

### Closed-Book QA and Parametric Knowledge

1. **Roberts, A., Raffel, C., & Shazeer, N.** (2020). "How Much Knowledge Can You Pack Into the Parameters of a Language Model?" EMNLP 2020. arXiv:2002.08910

2. **Petroni, F., Rocktaschel, T., Lewis, P., Bakhtin, A., Wu, Y., Miller, A.H., & Riedel, S.** (2019). "Language Models as Knowledge Bases?" EMNLP 2019. arXiv:1909.01066

3. **Khashabi, D., Min, S., Khot, T., Sabharwal, A., Tafjord, O., Clark, P., & Hajishirzi, H.** (2020). "UnifiedQA: Crossing Format Boundaries with a Single QA System." EMNLP 2020. arXiv:2005.00700

4. **Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C.** (2023). "Large Language Models Struggle to Learn Long-Tail Knowledge." ICML 2023. arXiv:2211.08411

5. **Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H.** (2023). "When Not to Trust Language Models." ACL 2023. arXiv:2212.10511

### Recitation and Generation as Retrieval

6. **Sun, Z., Wang, X., Tay, Y., Yang, Y., & Zhou, D.** (2023). "Recitation-Augmented Language Models." ICLR 2023. arXiv:2210.01296

7. **Yu, W., Iter, D., Wang, S., Xu, Y., Ju, M., Subramanian, S., Zhu, C., & Zeng, M.** (2023). "Generate rather than Retrieve: Large Language Models are Strong Context Generators." ICLR 2023. arXiv:2209.10063

### Self-RAG and Self-Generated Context

8. **Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H.** (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR 2024. arXiv:2310.11511

9. **Wang, Y., Li, P., Sun, M., & Liu, Y.** (2023). "Self-Knowledge Guided Retrieval Augmentation for Large Language Models." arXiv:2310.05002

10. **Khattab, O., Santhanam, K., Li, X.L., Hall, D., Liang, P., Potts, C., & Zaharia, M.** (2023). "Demonstrate-Search-Predict: Composing Retrieval and Language Model Modules." arXiv:2212.14024

### Knowledge Neurons and Parametric Knowledge Storage

11. **Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F.** (2022). "Knowledge Neurons in Pretrained Transformers." ACL 2022. arXiv:2104.08696

12. **Meng, K., Bau, D., Andonian, A., & Belinkov, Y.** (2022). "Locating and Editing Factual Associations in GPT." NeurIPS 2022. arXiv:2202.05262

13. **Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., & Bau, D.** (2023). "Mass-Editing Memory in a Transformer." ICLR 2023. arXiv:2210.07229

14. **Geva, M., Schuster, R., Berant, J., & Levy, O.** (2021). "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021. arXiv:2012.14913

15. **Geva, M., Caciularu, A., Wang, K.R., & Goldberg, Y.** (2022). "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space." EMNLP 2022.

### Cognitive Science of Memory

16. **Bartlett, F.C.** (1932). *Remembering: A Study in Experimental and Social Psychology*. Cambridge University Press.

17. **Tulving, E.** (1972). "Episodic and Semantic Memory." In *Organization of Memory* (ed. Tulving & Donaldson).

18. **Schacter, D.L., & Addis, D.R.** (2007). "Constructive Memory: Past and Future." Chapter in *Remembering*.

19. **Johnson, M.K., Hashtroudi, S., & Lindsay, D.S.** (1993). "Source Monitoring." Psychological Bulletin, 114(1), 3-28.

20. **Anderson, M.C., Bjork, R.A., & Bjork, E.L.** (1994). "Retrieval-Induced Forgetting." Journal of Experimental Psychology: Learning, Memory, and Cognition.

21. **Born, J., Rasch, B., & Gais, S.** (2006). "About Sleep's Role in Memory." Physiological Reviews.

22. **McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C.** (1995). "Why There Are Complementary Learning Systems." Psychological Review.

### Memorization and Copy Mechanisms

23. **Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F., & Zhang, C.** (2023). "Quantifying Memorization Across Neural Language Models." ICLR 2023. arXiv:2202.07646

24. **Carlini, N., Tramer, F., Wallace, E., et al.** (2021). "Extracting Training Data from Large Language Models." USENIX Security 2021. arXiv:2012.07805

25. **Vinyals, O., Fortunato, M., & Jaitly, N.** (2015). "Pointer Networks." NeurIPS 2015.

26. **See, A., Liu, P.J., & Manning, C.D.** (2017). "Get To The Point: Summarization with Pointer-Generator Networks." ACL 2017.

27. **Feldman, V.** (2020). "Does Learning Require Memorization? A Short Tale about a Long Tail." STOC 2020.

### Continual Learning for Language Models

28. **Jang, J., Ye, S., Yang, S., et al.** (2022). "Towards Continual Knowledge Learning of Language Models." ICLR 2022. arXiv:2110.03215

29. **Zheng, C., Li, L., Dong, Q., et al.** (2023). "Can We Edit Factual Knowledge by In-Context Learning?" arXiv:2305.12740

30. **Yao, Y., Wang, P., Tian, B., et al.** (2023). "Editing Large Language Models: Problems, Methods, and Opportunities." EMNLP 2023.

### Knowledge Editing

31. **Meng, K., et al.** (2022). "Locating and Editing Factual Associations in GPT" (ROME). NeurIPS 2022.

32. **Meng, K., et al.** (2023). "Mass-Editing Memory in a Transformer" (MEMIT). ICLR 2023.

---

## Appendix: Quick Reference Table of All Papers

| # | Paper | Authors | Year | Key Concept | Relation to Text-Generative Memory |
|---|-------|---------|------|-------------|-------------------------------------|
| 1 | How Much Knowledge in Parameters | Roberts, Raffel, Shazeer | 2020 | T5 closed-book QA | Parameters ARE the knowledge base |
| 2 | Language Models as Knowledge Bases | Petroni et al. | 2019 | LAMA probe | LMs store relational facts |
| 3 | RECITE | Sun, Wang, Tay, Yang, Zhou | 2023 | Recite then answer | Explicit parametric passage retrieval |
| 4 | GenRead | Yu et al. | 2023 | Generate rather than retrieve | Generated docs beat retrieved docs |
| 5 | Self-RAG | Asai et al. | 2024 | Adaptive retrieval with reflection | Chooses parametric vs. external |
| 6 | Knowledge Neurons | Dai et al. | 2022 | MLP neurons store facts | Mechanistic basis of parametric memory |
| 7 | ROME | Meng et al. | 2022 | Rank-one fact editing | Memories can be surgically modified |
| 8 | MEMIT | Meng et al. | 2023 | Mass fact editing | Bulk memory updates possible |
| 9 | FF Layers as Key-Value Memories | Geva et al. | 2021 | MLPs are associative memories | Architectural basis of storage |
| 10 | Bartlett - Remembering | Bartlett | 1932 | Reconstructive memory | Human memory = reconstruction, not retrieval |
| 11 | Episodic vs. Semantic Memory | Tulving | 1972 | Two memory systems | LLMs primarily store semantic memory |
| 12 | Constructive Memory | Schacter & Addis | 2007 | Memory and imagination share machinery | Generation and recall are the same process |
| 13 | Sleep and Memory | Born, Rasch, Gais | 2006 | Selective consolidation during sleep | Rehearsal should be selective |
| 14 | Retrieval-Induced Forgetting | Anderson, Bjork, Bjork | 1994 | Practicing some items suppresses others | Selective rehearsal has competitive dynamics |
| 15 | Memorization in LLMs | Carlini et al. | 2023 | Models memorize training data | Upper bound on verbatim recall |
| 16 | Extracting Training Data | Carlini et al. | 2021 | Can extract memorized sequences | LMs do store specific text |
| 17 | Long-Tail Knowledge | Kandpal et al. | 2023 | Rare facts poorly stored | Frequency determines retention |
| 18 | When Not to Trust LMs | Mallen et al. | 2023 | Parametric fails on rare entities | Popularity predicts recall accuracy |
| 19 | Continual Knowledge Learning | Jang et al. | 2022 | LLMs forget facts with new training | Rare facts forgotten first |
| 20 | CLS Theory | McClelland et al. | 1995 | Fast hippocampal + slow cortical | Two-speed memory system |
| 21 | Pointer Networks | Vinyals et al. | 2015 | Copy from input | Explicit copy vs. parametric generation |
| 22 | Does Learning Require Memorization | Feldman | 2020 | Memorization needed for long-tail | Memorization and generalization complement |

---

*This document complements the DSI-focused survey (`dsi_generative_retrieval_survey.md`) by covering the text-generative side of the retrieval spectrum. Together, they provide a complete picture of how models can "remember" -- either by generating pointers (IDs) or by generating content (text) -- and how forgetting manifests differently in each paradigm.*

*Last updated: 2026-02-23*
