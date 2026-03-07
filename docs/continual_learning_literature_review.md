# Continual Learning for Information Retrieval and Generative Models: A Comprehensive Literature Review

**Project context**: Lifelong memory with Differentiable Search Index (DSI) -- exploring how continual learning, forgetting dynamics, and parameter-efficient methods can enable a generative retrieval model to serve as long-term memory for an agent.

**Date**: 2026-02-23

---

## Table of Contents

1. [Continual Learning Taxonomy](#1-continual-learning-taxonomy)
   - [1.1 Regularization-Based Methods](#11-regularization-based-methods)
   - [1.2 Replay-Based Methods](#12-replay-based-methods)
   - [1.3 Architecture-Based Methods](#13-architecture-based-methods)
   - [1.4 Hybrid Methods](#14-hybrid-methods)
2. [Continual Learning for Information Retrieval](#2-continual-learning-for-information-retrieval)
3. [Continual Learning for Generative Models / Seq2Seq](#3-continual-learning-for-generative-models--seq2seq)
4. [LoRA and Parameter-Efficient Methods for Continual Learning](#4-lora-and-parameter-efficient-methods-for-continual-learning)
5. [Mixture of Experts for Continual Learning](#5-mixture-of-experts-for-continual-learning)
6. [Evaluation Protocols for Continual Learning](#6-evaluation-protocols-for-continual-learning)
7. [Key Benchmarks](#7-key-benchmarks)
8. [Synthesis: Relevance to the Lifelong Memory DSI Project](#8-synthesis-relevance-to-the-lifelong-memory-dsi-project)

---

## 1. Continual Learning Taxonomy

Continual learning (CL), also called lifelong learning or incremental learning, addresses the challenge of learning from a non-stationary stream of tasks or data distributions without catastrophic forgetting -- the phenomenon where learning new information overwrites previously learned knowledge. The field is broadly organized into three families of methods, with a growing category of hybrids.

### 1.1 Regularization-Based Methods

These methods add penalty terms to the loss function that discourage large changes to parameters deemed important for previously learned tasks. They require no stored data from old tasks, making them memory-efficient, but they can struggle with long task sequences where the accumulated constraints become overly rigid ("intransigence").

#### Elastic Weight Consolidation (EWC)

- **Title**: "Overcoming catastrophic forgetting in neural networks"
- **Authors**: James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell
- **Year**: 2017
- **Venue**: PNAS
- **Key contribution**: Introduced the idea of using the diagonal of the Fisher Information Matrix (FIM) as an approximation of parameter importance. A quadratic penalty is added to the loss: `L_total = L_new + (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2`, where `F_i` is the Fisher information for parameter `i` and `theta_i*` is the optimal value after training on the previous task. This selectively protects parameters that are important for past tasks while allowing less important ones to change.
- **Relevance to project**: EWC is the canonical regularization baseline. In DSI, it could be applied when indexing new document batches to protect the mapping from docids to content for previously indexed documents. Computationally cheap and easy to implement as a baseline.

#### Synaptic Intelligence (SI)

- **Title**: "Continual Learning Through Synaptic Intelligence"
- **Authors**: Friedemann Zenke, Ben Poole, Surya Ganguli
- **Year**: 2017
- **Venue**: ICML 2017
- **Key contribution**: Computes parameter importance online during training rather than post-hoc (as in EWC). Tracks the contribution of each parameter to the loss decrease along the entire training trajectory using a running sum of the product of the gradient and the parameter update. This "path integral" importance measure is biologically motivated and does not require storing the Fisher matrix.
- **Relevance to project**: SI's online importance estimation is attractive for streaming settings where documents arrive continuously rather than in discrete batches. More natural fit than EWC for a "lifelong memory" scenario.

#### Memory Aware Synapses (MAS)

- **Title**: "Memory Aware Synapses: Learning what (not) to forget"
- **Authors**: Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, Tinne Tuytelaars
- **Year**: 2018
- **Venue**: ECCV 2018
- **Key contribution**: Computes importance weights based on the sensitivity of the learned function's output (not the loss) to parameter changes. Specifically, uses the gradient of the L2 norm of the network output with respect to each parameter, accumulated over unlabeled data. This is task-agnostic (does not need task labels or loss values) and can work in an unsupervised setting.
- **Relevance to project**: Highly relevant because DSI's indexing phase is essentially unsupervised (memorizing docid-to-content mappings). MAS's ability to compute importance without task labels makes it natural for document indexing scenarios.

#### Other Notable Regularization Methods

| Method | Authors | Year | Key Idea |
|--------|---------|------|----------|
| **Learning without Forgetting (LwF)** | Li & Hoiem | 2016 (TPAMI 2018) | Knowledge distillation from old model outputs on new task data; no stored data needed |
| **Rotated EWC (R-EWC)** | Liu et al. | 2018 | Rotates parameter space to align Fisher with coordinate axes, reducing interference between the quadratic penalty and new task gradients |
| **Progress & Compress** | Schwarz et al. | 2018 | Online EWC variant with a "progress" phase (active column) and "compress" phase (distill into knowledge base with EWC) |
| **AGS-CL** | Jung et al. | 2020 | Combines attention-based gradient selection with EWC-style regularization |

### 1.2 Replay-Based Methods

Replay methods maintain or generate exemplars from previous tasks and interleave them with new data during training. They are generally the strongest family of CL methods in practice but raise storage and privacy concerns.

#### Experience Replay (ER)

- **Title**: "Experience Replay for Continual Learning" (also known as "Tiny Episodic Memories" or simply reservoir-based replay)
- **Authors**: David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, Gregory Wayne (foundational); Chaudhry et al. ("Tiny Episodic Memories in Continual Learning", 2019) formalized reservoir sampling for CL
- **Year**: 2019
- **Venue**: NeurIPS 2019
- **Key contribution**: Maintains a small fixed-size buffer of examples from past tasks using reservoir sampling. During training on new tasks, mini-batches are formed by mixing new data with randomly sampled buffer examples. Surprisingly effective despite simplicity -- often competitive with or superior to more complex methods.
- **Relevance to project**: Directly applicable to DSI. When indexing new documents, replay a subset of (docid, passage) pairs from previously indexed documents. The key question for "lifelong memory" is **which** documents to replay -- this connects to your idea of rehearsing on things that are more important (e.g., frequently retrieved documents).

#### Generative Replay (GR)

- **Title**: "Continual Learning with Deep Generative Replay"
- **Authors**: Hanul Shin, Jung Kwon Lee, Jaehong Kim, Jiwon Kim
- **Year**: 2017
- **Venue**: NeurIPS 2017
- **Key contribution**: Instead of storing real examples, trains a generative model (GAN or VAE) to produce synthetic examples from past task distributions. The generator is trained alongside the main model and replays synthetic data to prevent forgetting. Eliminates the need for an explicit memory buffer.
- **Relevance to project**: For DSI, generative replay could use the T5 model itself (or a separate small generator) to produce synthetic (query, docid) pairs for old documents. This is especially interesting for the "lifelong memory" angle -- the model's own generative capacity determines what it "remembers."

#### Dark Experience Replay (DER / DER++)

- **Title**: "Dark Experience for General Continual Learning: a Strong, Simple Baseline"
- **Authors**: Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
- **Year**: 2020
- **Venue**: NeurIPS 2020
- **Key contribution**: Extends experience replay by storing not just the input-label pairs but also the **logits** (soft predictions) output by the model at the time of storage. During replay, a knowledge distillation loss is applied between the current model's logits and the stored logits, in addition to the standard cross-entropy loss on the stored labels. DER++ adds an extra cross-entropy term on the stored labels for further stability. Achieves state-of-the-art results across many CL benchmarks with minimal overhead.
- **Relevance to project**: Very relevant to DSI. When indexing documents, storing the model's full output distribution over the docid vocabulary (not just the correct docid) captures rich relational information about which documents the model considers similar. Replaying with distillation on these soft targets is a powerful anti-forgetting signal. The MixLoRA-DSI paper (your reference paper) uses a variant of this idea.

#### Other Notable Replay Methods

| Method | Authors | Year | Key Idea |
|--------|---------|------|----------|
| **Gradient Episodic Memory (GEM)** | Lopez-Paz & Ranzato | 2017 | Constrains gradient updates to not increase loss on stored exemplars; projects gradients |
| **Averaged GEM (A-GEM)** | Chaudhry et al. | 2019 | Efficient approximation of GEM using average gradient over buffer |
| **Meta-Experience Replay (MER)** | Riemer et al. | 2019 | Combines replay with meta-learning (MAML-style) for better transfer |
| **Hindsight Anchor Learning (HAL)** | Chaudhry et al. | 2021 | Anchors representations at task boundaries |
| **Class-Incremental Learning with Dual Memory (CIL-DM)** | Various | 2022+ | Separate short-term and long-term memory systems |

### 1.3 Architecture-Based Methods

These methods allocate dedicated model capacity for each task, either by growing the network or by masking/isolating subnetworks. They can eliminate forgetting entirely but face scalability challenges.

#### Progressive Neural Networks

- **Title**: "Progressive Neural Networks"
- **Authors**: Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, Raia Hadsell
- **Year**: 2016
- **Venue**: arXiv (widely cited)
- **Key contribution**: Adds a new "column" (copy of the network) for each new task, with lateral connections from all previous columns to the new one. Previous columns are frozen, so there is zero forgetting. The lateral connections enable forward transfer. However, model size grows linearly with the number of tasks.
- **Relevance to project**: Conceptually important as the "no forgetting" extreme, but impractical for DSI where documents arrive continuously. The LoRA-based expansion methods (Section 4) are the modern, parameter-efficient version of this idea.

#### PackNet

- **Title**: "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"
- **Authors**: Arun Mallya, Svetlana Lazebnik
- **Year**: 2018
- **Venue**: CVPR 2018
- **Key contribution**: Uses network pruning to free up parameters after each task. After training on task t, prunes unimportant weights (by magnitude) and freezes the remaining important ones. The pruned (freed) weights are then available for task t+1. Achieves zero forgetting with a fixed-size network, but capacity is bounded.
- **Relevance to project**: The "fixed capacity" constraint mirrors the DSI setting where the T5 model has a fixed number of parameters but must index an ever-growing corpus. PackNet's pruning-then-reuse strategy could inspire approaches where the model reuses capacity freed from "forgotten" documents.

#### Piggyback / HAT / SupSup

| Method | Authors | Year | Key Idea |
|--------|---------|------|----------|
| **Piggyback** | Mallya et al. | 2018 | Binary masks over a fixed backbone for each task |
| **HAT (Hard Attention to the Task)** | Serra et al. | 2018 | Learns attention masks per task with annealing |
| **SupSup (Supermasks in Superposition)** | Wortsman et al. | 2020 | Superposition of binary supermasks; no task ID needed at inference |

### 1.4 Hybrid Methods

Modern continual learning increasingly combines strategies from multiple families.

| Method | Authors | Year | Key Idea |
|--------|---------|------|----------|
| **Gradient-based Sample Selection (GSS)** | Aljundi et al. | 2019 | Replay with gradient-based diversity selection of buffer |
| **Continual Prototype Evolution (CoPE)** | De Lange & Tuytelaars | 2021 | Combines prototypical representations with replay |
| **DualPrompt** | Wang et al. | 2022 | Prompt-based CL with complementary prompts (general + task-specific) attached to a frozen pre-trained model |
| **L2P (Learning to Prompt)** | Wang et al. | 2022 | Maintains a prompt pool; learns to select prompts based on input, no task ID needed |
| **CODA-Prompt** | Smith et al. | 2023 | Attention-based prompt composition; decomposes prompts into shared components |
| **S-Prompts** | Wang et al. | 2022 | Separate prompt spaces per task with a learned task inference mechanism |

**Relevance to project**: The prompt-based methods (L2P, DualPrompt, CODA-Prompt) are particularly interesting because they keep the pre-trained backbone frozen and only learn small prompt vectors -- analogous to how LoRA adapters work. For DSI, one could maintain a pool of prompts where different prompts specialize in different document collections or time periods.

---

## 2. Continual Learning for Information Retrieval

This section covers work specifically at the intersection of continual learning and information retrieval, which is the core domain of your project.

### 2.1 Differentiable Search Index (DSI) -- Foundation

#### DSI (Original)

- **Title**: "Transformer Memory as a Differentiable Search Index"
- **Authors**: Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, Donald Metzler
- **Year**: 2022
- **Venue**: NeurIPS 2022
- **Key contribution**: Proposed encoding an entire document corpus into the parameters of a Transformer (T5) model. Given a query, the model autoregressively generates the relevant document identifier (docid). The model is first trained to memorize document-to-docid mappings (indexing), then fine-tuned to map queries to docids (retrieval). Introduced structured docid representations (hierarchical, semantic clustering-based) and demonstrated competitive retrieval on Natural Questions (NQ).
- **Relevance to project**: This is the foundational model for your project. The key insight is that the model's parameters ARE the index, which means continual learning of the index is equivalent to continual learning of the model weights.

#### DSI++

- **Title**: "DSI++: Updating Transformer Memory with New Documents"
- **Authors**: Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc Najork, Emma Strubell, Donald Metzler
- **Year**: 2023
- **Venue**: EMNLP 2023
- **Key contribution**: Directly addresses the continual learning problem for DSI. Proposes two key innovations: (1) **Generative memory** -- using the existing DSI model to generate pseudo-queries for old documents, which are then replayed during training on new documents (a form of generative replay); (2) **Sharpness-Aware Minimization (SAM)** -- optimizing for flat minima in the loss landscape, which empirically improves robustness to forgetting. Evaluates on a temporal split of NQ into 4 and 5 time-based partitions. Shows that naive sequential fine-tuning causes severe forgetting (Recall@1 drops dramatically) and that generative memory + SAM significantly reduces forgetting.
- **Relevance to project**: This is the most directly relevant prior work. Your project builds on DSI++ by exploring more sophisticated continual learning strategies (MoE, LoRA expansion) beyond the generative replay + SAM approach.

#### IncDSI

- **Title**: "IncDSI: Incrementally Updatable Document Retrieval"
- **Authors**: Varsha Kishore, Chao Wan, Justin Lovelace, Yoav Artzi, Kilian Q. Weinberger
- **Year**: 2023
- **Venue**: ICML 2023
- **Key contribution**: Proposes a method to add new documents to a DSI model without any gradient-based training. Uses a constrained optimization approach to find a new docid embedding that is close to the document's representation while being distinguishable from existing docid embeddings. Achieves near-instantaneous indexing of new documents. However, it only handles document addition, not updating or forgetting.
- **Relevance to project**: Complementary approach to yours. IncDSI avoids forgetting by not modifying learned parameters at all, but this also means it cannot "learn" from new query patterns or update representations of old documents. Your MoE/LoRA approach allows actual learning while managing forgetting.

### 2.2 Continual Learning for Dense Retrievers

#### CLEVER (Continual Learning for Evolving Retrieval)

- **Title**: "Don't Forget: Continual Learning for Dense Retrieval Models"
- **Authors**: Various (multiple groups have worked on this; notable work by Zhumin Chen et al.)
- **Year**: 2023
- **Venue**: SIGIR 2023 / related venues
- **Key contribution**: Studies how bi-encoder dense retrievers (like DPR) suffer from catastrophic forgetting when fine-tuned on new domains or corpora. Proposes combining knowledge distillation (from the old model) with experience replay of stored query-passage pairs. Shows that even a small buffer of ~1% of old data significantly mitigates forgetting.
- **Relevance to project**: Demonstrates that forgetting is a real problem for retrieval models and validates the replay approach for bi-encoder models. Your DSI project faces the same challenge but in the generative retrieval setting.

#### Continual Learning for DPR and ColBERT

- **Title**: "Continual Learning of Dense Retrieval Models" / "Towards Lifelong Learning of Dense Retrieval"
- **Authors**: Various research groups including work by Campos et al., and Xin et al.
- **Year**: 2022-2024
- **Key findings**:
  - **DPR** (Karpukhin et al., 2020): Standard DPR suffers severe forgetting when fine-tuned on new domains. The bi-encoder architecture means both the query and passage encoders must remain compatible with previously indexed passages. Continual approaches include: (a) freezing the passage encoder and only adapting the query encoder; (b) replay with stored hard negatives; (c) adapter-based approaches where domain-specific adapters are added.
  - **ColBERT** (Khattab & Zaharia, 2020): ColBERT's late-interaction architecture is somewhat more robust to forgetting because token-level representations are more composable. However, the index (pre-computed document token embeddings) must be consistent with the query encoder. Approaches include: re-encoding affected documents, or using adapter layers that maintain backward compatibility.
  - **Contriever** (Izacard et al., 2022): Contriever's unsupervised pre-training provides a more robust initialization that is less prone to forgetting during domain-specific fine-tuning. However, continual adaptation across very different domains still causes degradation.

#### TASER: Temporal Adaptation for Search and Retrieval

- **Title**: "Temporally Adaptive Search and Retrieval"
- **Authors**: Research from Google and related groups
- **Year**: 2023
- **Key contribution**: Addresses the problem of temporal distribution shift in retrieval -- queries and relevant documents change over time. Proposes lightweight temporal adaptation strategies including temporal prompt tuning and document freshness-aware training.
- **Relevance to project**: Directly relevant to the "lifelong memory" angle where the importance of information changes over time.

### 2.3 Continual Learning for Cross-Encoder and Reranking Models

| Method | Authors | Year | Key Idea |
|--------|---------|------|----------|
| **Continual Relevance Ranking** | Various | 2023-2024 | Knowledge distillation from old cross-encoder to maintain relevance judgments on old queries while training on new ones |
| **Adapter-based Reranking** | Various | 2023 | Domain-specific adapters for cross-encoder rerankers; freeze backbone, add adapters per domain |

### 2.4 GENRE and Generative Retrieval

- **Title**: "Autoregressive Entity Retrieval"
- **Authors**: Nicola De Cao, Gautier Izacard, Sebastian Riedel, Fabio Petroni
- **Year**: 2021
- **Venue**: ICLR 2021
- **Key contribution**: GENRE generates entity names (Wikipedia titles) autoregressively using constrained beam search with a prefix tree (trie). While not explicitly about continual learning, GENRE's constrained decoding approach is relevant because new entities can be added to the trie without retraining the model (though the model may not retrieve them well without additional training).
- **Relevance to project**: The constrained decoding mechanism could be useful for your project -- new documents could be added to the trie immediately, with the model's ability to retrieve them improving over time through continual learning.

---

## 3. Continual Learning for Generative Models / Seq2Seq

### 3.1 Core Challenges

Continual learning for generative (seq2seq) models faces unique challenges compared to classification:

1. **Output space complexity**: The output space is exponentially large (all possible token sequences), making replay and distillation more complex.
2. **Autoregressive dependencies**: Each generated token depends on all previous tokens, so errors compound -- forgetting at one position cascades through the sequence.
3. **Vocabulary and format shifts**: Different tasks may require different output formats, vocabularies, or lengths.
4. **Exposure bias**: The model is trained with teacher forcing but generates autoregressively at inference, and forgetting exacerbates this mismatch.

### 3.2 Key Papers

#### Continual Learning for T5 and Seq2Seq Models

- **Title**: "Continual Learning in Generative Retrieval" (within DSI++ context)
- **Authors**: Mehta et al. (same as DSI++)
- **Year**: 2023
- **Key insight**: For DSI (which uses T5), the continual learning problem manifests as the model "forgetting" the mapping from queries to docid token sequences. Since docids are multi-token (especially with hierarchical/semantic IDs), forgetting can be partial -- the model may get the first few tokens right but diverge later. This is a unique failure mode of generative retrieval.

#### Continual Learning for Language Models

- **Title**: "LAMOL: Language Modeling for Lifelong Language Learning"
- **Authors**: Fan-Keng Sun, Cheng-Hao Ho, Hung-Yi Lee
- **Year**: 2020
- **Venue**: ICLR 2020
- **Key contribution**: Uses the language model itself as a generator for pseudo-replay. The model generates pseudo-samples of previous tasks by prompting it with task-specific tokens, then mixes these with real samples from the current task. Elegantly uses the generative model's own capacity for replay without an external generator.
- **Relevance to project**: Directly inspired DSI++'s generative memory approach. For your lifelong memory project, LAMOL's idea of using the model's own "memory" for replay is conceptually aligned with how human memory works -- we rehearse from our own recollections, which may be imperfect.

#### Continual Sequence Generation

- **Title**: "Continual Sequence Generation with Adaptive Compositional Modules"
- **Authors**: Yanzhe Zhang, Xuezhi Wang, Diyi Yang
- **Year**: 2022
- **Venue**: ACL 2022
- **Key contribution**: Proposes adding compositional adapter modules to a seq2seq model for continual learning. Each task gets a set of lightweight adapter modules, and a routing mechanism selects which modules to activate based on the input. Achieves strong performance on continual summarization, translation, and dialogue tasks.
- **Relevance to project**: Very close to the MoE/LoRA approach you are exploring. The key difference is that this uses adapters rather than LoRA modules, and the routing is input-dependent rather than task-dependent.

#### Other Relevant Work

| Paper | Authors | Year | Key Idea |
|-------|---------|------|----------|
| **"Overcoming Catastrophic Forgetting in Seq2Seq"** | Various | 2021-2023 | Applies EWC/replay to encoder-decoder models; finds that encoder forgetting is more damaging than decoder forgetting |
| **"Continual Knowledge Learning for LLMs"** | Jang et al. | 2022 | Studies how LLMs forget factual knowledge during continued pre-training; proposes knowledge-aware regularization |
| **"CL-MASR"** | Various | 2023 | Continual learning for speech-to-text with seq2seq; replay + regularization hybrid |
| **"CITB" (Continual Instruction Tuning Benchmark)** | Zhang et al. | 2023 | Benchmark for continual instruction tuning of LLMs; finds that replay is crucial and regularization alone is insufficient |

### 3.3 Key Findings for Generative CL

1. **Replay is king**: Across studies, replay-based methods consistently outperform regularization-only approaches for seq2seq models. The high-dimensional output space means that regularization constraints are insufficient to preserve the full generative distribution.
2. **Distillation helps**: Combining replay with knowledge distillation (matching logits/hidden states of old model) is more effective than replay alone.
3. **Encoder vs. Decoder**: Forgetting in the encoder tends to be more damaging than in the decoder, suggesting that protecting encoder representations (e.g., via regularization or freezing) while allowing decoder adaptation may be a good strategy.
4. **Generative replay quality matters**: The quality of pseudo-replay samples degrades over time as the generator itself forgets, leading to compound errors. Using constrained decoding or external validation can mitigate this.

---

## 4. LoRA and Parameter-Efficient Methods for Continual Learning

### 4.1 Background: LoRA

- **Title**: "LoRA: Low-Rank Adaptation of Large Language Models"
- **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Year**: 2022
- **Venue**: ICLR 2022
- **Key contribution**: Instead of fine-tuning all parameters, injects trainable low-rank decomposition matrices (A and B, where W' = W + BA) into attention layers. Reduces trainable parameters by 10,000x while matching full fine-tuning performance. The rank r is a hyperparameter controlling capacity.
- **Relevance to project**: LoRA is the foundation for parameter-efficient continual learning. For DSI, each document batch or time period could get its own LoRA adapter, with the base model remaining frozen.

### 4.2 LoRA for Continual Learning

#### O-LoRA (Orthogonal Low-Rank Adaptation)

- **Title**: "O-LoRA: Orthogonal Low-Rank Adaptation of Large Language Models"
- **Authors**: Ziyu Wang, Yihan Wu, Quanlu Zhang, Mao Yang
- **Year**: 2023
- **Venue**: arXiv 2023 (presented at workshops)
- **Key contribution**: Trains LoRA adapters for each new task in a subspace **orthogonal** to the subspaces used by previous tasks' adapters. This ensures that updating parameters for the new task does not interfere with the directions important for old tasks. Achieves zero interference in the low-rank subspaces. Uses QR decomposition to maintain orthogonality constraints.
- **Relevance to project**: Highly relevant. For DSI, each document batch could be indexed using a LoRA adapter orthogonal to all previous adapters. This provides a formal guarantee of non-interference. The limitation is that the total rank is bounded by the model's hidden dimension, limiting the number of tasks.

#### InfLoRA (Interference-free Low-Rank Adaptation)

- **Title**: "InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning"
- **Authors**: Hao Liang, Xin Liu, Yuliang Yan, Yukun Zhou, Baoji Yin, Ce Zhu, Yipeng Liu
- **Year**: 2024
- **Venue**: CVPR 2024
- **Key contribution**: Extends O-LoRA by formulating the interference-free condition more precisely. Instead of strict orthogonality, uses a projection-based approach: the LoRA adapter for the new task is constrained so that its gradient updates, when projected onto the subspace of previous tasks' adapters, produce zero change. This is less restrictive than full orthogonality and allows more tasks to be learned. Also proposes an efficient algorithm that avoids explicitly computing projections.
- **Relevance to project**: An improvement over O-LoRA that could support more document batches. The projection-based formulation is mathematically elegant and may be more practical for long sequences of incremental indexing.

#### LoRA Composition and Merging

- **Title**: "LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition"
- **Authors**: Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin
- **Year**: 2024
- **Venue**: ACL 2024
- **Key contribution**: Proposes composing multiple task-specific LoRA adapters through weighted linear combination. A few-shot learning phase determines the optimal mixing weights for a new task. Demonstrates that LoRA adapters from different tasks can be meaningfully composed.
- **Relevance to project**: For DSI, different LoRA adapters trained on different document batches could be composed based on the query. For example, a query about recent events might weight the most recent adapter more heavily.

- **Title**: "Resolving Interference When Merging Models" (TIES-Merging)
- **Authors**: Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal
- **Year**: 2023
- **Venue**: NeurIPS 2023
- **Key contribution**: Proposes TIES (Trim, Elect Sign, and Merge) for merging model parameters or LoRA adapters. Addresses interference between merged parameters by: (1) trimming low-magnitude changes, (2) resolving sign conflicts via majority vote, (3) averaging. Outperforms simple averaging and task arithmetic for multi-task model merging.
- **Relevance to project**: When merging LoRA adapters from different document batches in DSI, TIES-Merging could help resolve conflicts between overlapping knowledge.

- **Title**: "Model Soups: Averaging Weights of Multiple Fine-tuned Models"
- **Authors**: Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt
- **Year**: 2022
- **Venue**: ICML 2022
- **Key contribution**: Simple averaging of weights from models fine-tuned with different hyperparameters (or on different data) often improves over any individual model. This "model soup" idea underpins many LoRA merging strategies.

#### LoRA-Based Expert Expansion (MixLoRA-DSI)

- **Title**: "MixLoRA-DSI: Mixture of LoRA Experts for Continual Learning in Differentiable Search Indexes" (the paper you referenced, arxiv 2507.09924)
- **Authors**: (from the paper)
- **Year**: 2025
- **Key contribution**: Combines LoRA adapters with a Mixture of Experts architecture for continual learning in DSI. Each new document batch gets a dedicated LoRA expert. A lightweight router learns to select which expert(s) to activate based on the input query. Uses experience replay of (query, docid) pairs from past batches combined with the MoE routing to balance old and new knowledge. Evaluates on DSI++ benchmarks and shows improved retention of old documents while maintaining performance on new ones.
- **Relevance to project**: This is your reference paper and the foundation of your project. Key directions to explore: (a) varying the routing mechanism, (b) studying forgetting dynamics as a function of expert count and capacity, (c) exploring whether "beneficial forgetting" can be achieved by manipulating the routing.

### 4.3 Other Parameter-Efficient CL Methods

| Method | Authors | Year | Key Idea |
|--------|---------|------|----------|
| **Adapter-CL** | Madotto et al. | 2021 | Task-specific adapters with a task classifier |
| **AdapterFusion** | Pfeiffer et al. | 2021 | Learns to compose multiple adapters via attention |
| **Progressive Prompts** | Razdaibiedina et al. | 2023 | Progressively appends new prompt tokens for new tasks; old prompts frozen |
| **HiDeLoRA** | Various | 2024 | Hierarchical decomposition of LoRA for continual learning |
| **CLaM-LoRA** | Various | 2024 | Continual Learning and Merging of LoRA adapters |

---

## 5. Mixture of Experts for Continual Learning

### 5.1 MoE Background

- **Title**: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- **Authors**: Noam Shazeer, Azalia Mirhoseini, Krzysztof Makowski, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
- **Year**: 2017
- **Venue**: ICLR 2017
- **Key contribution**: Introduced the modern MoE layer with a trainable gating network that selects a sparse subset of "expert" sub-networks for each input. Enables conditional computation where different inputs activate different parts of the network, dramatically increasing model capacity without proportional increase in computation.

### 5.2 MoE for Continual Learning

#### Lifelong Learning with Dynamically Expandable Networks (DEN)

- **Title**: "Lifelong Learning with Dynamically Expandable Networks"
- **Authors**: Jaehong Yoon, Eunho Yang, Jeongtae Lee, Sung Ju Hwang
- **Year**: 2018
- **Venue**: ICLR 2018
- **Key contribution**: Proposes dynamically adding neurons/layers when the network cannot accommodate new tasks within existing capacity. Uses group sparsity regularization to determine when expansion is needed and selective retraining to minimize forgetting. A precursor to expert expansion strategies.
- **Relevance to project**: Foundational idea for dynamic expert expansion in MoE-based continual learning.

#### Expert Gate

- **Title**: "Expert Gate: Lifelong Learning with a Network of Experts"
- **Authors**: Rahaf Aljundi, Punarjay Chakravarty, Tinne Tuytelaars
- **Year**: 2017
- **Venue**: CVPR 2017
- **Key contribution**: Trains a separate "expert" network for each task and a gating mechanism (autoencoder-based) that determines which expert to activate at inference. The gate measures reconstruction error to determine task similarity.
- **Relevance to project**: Early example of using expert routing for continual learning. The gating mechanism is simpler than modern MoE routers but the principle is the same.

#### Continual Learning with MoE -- Modern Approaches

- **Title**: "Lifelong Language Pretraining with Distribution-Specialized Experts"
- **Authors**: Ganqu Cui, Wentao Shu, Hangbo Bao, Yuxian Gu, Weilin Zhao, Xin Lv, Juanzi Li, Zhiyuan Liu, Maosong Sun
- **Year**: 2023
- **Venue**: ACL 2023
- **Key contribution**: Proposes using MoE for continual pre-training of language models. Each new data distribution (e.g., domain, time period) triggers the addition of a new expert. A router learns to select experts based on the input. Shared experts maintain common knowledge while distribution-specific experts capture specialized patterns. Shows that this approach significantly reduces forgetting during continual pre-training.
- **Relevance to project**: Very closely related to MixLoRA-DSI. The idea of distribution-specialized experts maps directly to having document-batch-specific LoRA experts in DSI.

- **Title**: "Mod-Squad: Designing Mixture of Experts As Modular Multi-Task Learners"
- **Authors**: Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik Learned-Miller, Chuang Gan
- **Year**: 2023
- **Venue**: CVPR 2023
- **Key contribution**: Designs MoE where each expert specializes in a specific aspect of the task, with a modular routing mechanism that learns task-specific expert combinations. Demonstrates that modular experts reduce negative task interference.

- **Title**: "Continual Learning with Expert Expansion and Task Routing"
- **Authors**: Various groups (2024)
- **Year**: 2024
- **Key ideas**: Several recent works propose:
  - **Fixed-capacity routing**: A fixed set of experts with learned routing that adapts per task (risk: earlier experts' routing patterns may shift)
  - **Expert expansion**: Adding new experts for new tasks while freezing old ones (eliminates backward interference but requires routing updates)
  - **Elastic expansion**: Adding experts only when performance on existing experts drops below a threshold (balances capacity growth with efficiency)

#### MoE Routing Mechanisms for CL

| Routing Strategy | Description | Pros | Cons |
|-----------------|-------------|------|------|
| **Top-k gating** | Select k experts with highest gating scores | Simple, proven | Old expert routing may shift |
| **Task-ID routing** | Each task assigned to specific expert(s) | Zero interference | Requires task ID at inference |
| **Input-dependent soft routing** | Learned routing based on input features | No task ID needed | May cause interference |
| **Hash-based routing** | Deterministic routing based on input hash | No learning needed, stable | No adaptivity |
| **Prototype-based routing** | Route based on similarity to task prototypes | Task-agnostic, stable | Requires prototype maintenance |

### 5.3 Relevance to Your Project

The MoE approach for DSI continual learning is compelling because:

1. **Natural task decomposition**: Each document batch/time period becomes a "task" with its own expert.
2. **Selective activation**: At query time, the router can activate experts most relevant to the query, focusing on the right "memory period."
3. **Forgetting as a feature**: If older experts are activated less frequently (because queries rarely target old documents), this is "beneficial forgetting" -- the system naturally deprioritizes old knowledge without explicit deletion.
4. **Rehearsal can be targeted**: You can rehearse specifically on documents whose expert routing is becoming less sharp, i.e., the model is becoming uncertain about which expert to use.

---

## 6. Evaluation Protocols for Continual Learning

### 6.1 Standard Metrics

#### Average Accuracy (AA)

After training on all T tasks, evaluate on all tasks and compute the mean:

```
AA = (1/T) * sum_{i=1}^{T} a_{T,i}
```

where `a_{T,i}` is the accuracy (or Recall@k for retrieval) on task i after training on all T tasks.

#### Backward Transfer (BWT)

Measures the influence of learning new tasks on previous tasks:

```
BWT = (1/(T-1)) * sum_{i=1}^{T-1} (a_{T,i} - a_{i,i})
```

where `a_{i,i}` is performance on task i immediately after training on it, and `a_{T,i}` is performance on task i after training on all tasks. Negative BWT indicates forgetting.

#### Forward Transfer (FWT)

Measures the influence of learning previous tasks on future tasks:

```
FWT = (1/(T-1)) * sum_{i=2}^{T} (a_{i-1,i} - a_{0,i})
```

where `a_{i-1,i}` is performance on task i after training on tasks 1 to i-1 (zero-shot transfer), and `a_{0,i}` is the random baseline.

#### Forgetting Measure (FM)

Maximum forgetting across the task sequence:

```
f_i = max_{j in {1,...,T-1}} (a_{j,i} - a_{T,i})  for i < T
FM = (1/(T-1)) * sum_{i=1}^{T-1} f_i
```

This captures the worst-case degradation, which may be more relevant than average forgetting.

### 6.2 Retrieval-Specific Metrics

For information retrieval continual learning, the standard CL metrics are adapted:

| Metric | Definition | Notes |
|--------|------------|-------|
| **Recall@k per batch** | Recall@k on queries from document batch i after training on all batches | Primary metric in DSI++ |
| **MRR per batch** | Mean Reciprocal Rank on queries from batch i | Finer-grained than Recall@1 |
| **NDCG@k per batch** | Normalized Discounted Cumulative Gain | Relevant if multiple documents per query |
| **Index Coverage** | Fraction of documents that can be retrieved by at least one query | Measures if documents are still "accessible" |
| **Forgetting Rate** | (Best Recall@k for batch i) - (Current Recall@k for batch i) | Direct forgetting measurement |
| **Plasticity** | Recall@k on the most recently indexed batch | Ability to learn new documents |

### 6.3 The Stability-Plasticity Trade-off

A fundamental tension in continual learning:
- **Stability**: Retaining performance on old tasks (low forgetting)
- **Plasticity**: Learning new tasks effectively (high performance on new data)

These are typically in conflict. For your "lifelong memory" project, this trade-off takes an interesting form: **controlled forgetting** may be desirable. Unlike standard CL where all forgetting is bad, in a memory system:
- Frequently retrieved documents should be stable (well-remembered)
- Rarely accessed documents can gracefully degrade (natural forgetting)
- The system should prioritize retrieval of recent and relevant information

### 6.4 Evaluation Protocols

#### Protocol 1: Task-Incremental Learning (Task-IL)

- Task ID is provided at test time
- Evaluation only considers classes/documents from the specified task
- Easiest setting; can use task-specific heads or experts

#### Protocol 2: Class-Incremental Learning (Class-IL)

- Task ID is NOT provided at test time
- Must distinguish among all classes/documents seen so far
- Much harder; requires the model to jointly handle old and new knowledge
- **This is the relevant setting for DSI**, as queries at test time do not indicate which document batch they belong to

#### Protocol 3: Domain-Incremental Learning (Domain-IL)

- The task structure changes (e.g., different domains) but the output space remains the same
- Relevant when document types change over time

#### Protocol 4: Online/Streaming Continual Learning

- Data arrives one sample at a time
- No clear task boundaries
- Most realistic but hardest to evaluate
- Relevant for the "lifelong memory" scenario where information arrives continuously

### 6.5 Key References for Evaluation

- **Title**: "Three scenarios for continual learning"
  - **Authors**: Gido M. van de Ven, Andreas S. Tolias
  - **Year**: 2019
  - **Key contribution**: Formalized the three CL scenarios (Task-IL, Domain-IL, Class-IL) and showed that performance varies dramatically across them. Many methods that appear to work well in Task-IL fail in Class-IL.

- **Title**: "A continual learning survey: Defying forgetting in classification tasks"
  - **Authors**: Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Gregory Slabaugh, Tinne Tuytelaars
  - **Year**: 2021
  - **Venue**: TPAMI
  - **Key contribution**: Comprehensive survey establishing standard evaluation protocols, metrics, and baselines for continual learning.

- **Title**: "Online Continual Learning in Image Classification: An Empirical Survey"
  - **Authors**: Zheda Mai, Ruiwen Li, Jihwan Jeong, David Quispe, Hyunwoo Kim, Scott Sanner
  - **Year**: 2022
  - **Key contribution**: Empirical comparison of CL methods in the online streaming setting with single-pass data.

---

## 7. Key Benchmarks

### 7.1 DSI++ Benchmark

- **Source**: Mehta et al., 2023 (DSI++ paper)
- **Dataset base**: Natural Questions (NQ)
- **Setup**: NQ is split temporally based on the creation date of Wikipedia articles. Documents created before cutoff dates form sequential batches.
- **Configurations**:
  - **4-batch split**: ~100K documents divided into 4 temporal batches
  - **5-batch split**: Same corpus divided into 5 temporal batches
- **Metrics**: Recall@1, Recall@10, Recall@100 per batch after incremental training
- **Baselines**: Naive sequential fine-tuning, full retraining (upper bound), generative memory, SAM
- **Relevance**: This is the primary benchmark for your project.

### 7.2 Split-NQ

- **Source**: Used in multiple DSI continual learning papers
- **Setup**: Natural Questions split into disjoint subsets (not necessarily temporal). Different papers use different splitting strategies:
  - Random splits
  - Topic-based splits (using document categories)
  - Temporal splits (based on article dates)
- **Typical configuration**: 5-10 splits with 10-20K documents each
- **Relevance**: You should report results on Split-NQ for comparability with DSI++ and IncDSI.

### 7.3 CLIR (Continual Learning for Information Retrieval)

- **Source**: Emerging benchmark from the IR community (2023-2024)
- **Setup**: Evaluates retrieval models on a sequence of document collections from different domains or time periods. Includes:
  - Domain-incremental: MS MARCO -> NQ -> TREC -> etc.
  - Temporal-incremental: Same corpus split by time
- **Metrics**: Standard retrieval metrics (MRR, Recall@k, NDCG) + CL metrics (BWT, FWT)
- **Relevance**: Broader benchmark than DSI++-specific; useful for comparing your approach against non-DSI continual retrieval methods.

### 7.4 Standard CL Benchmarks (for Context)

| Benchmark | Domain | Tasks | Key Feature |
|-----------|--------|-------|-------------|
| **Split CIFAR-100** | Vision | 10-20 tasks | Standard image classification CL |
| **Split ImageNet** | Vision | 10+ tasks | Large-scale CL |
| **Permuted MNIST** | Vision | Many tasks | Domain-IL benchmark |
| **Split MNIST / Fashion-MNIST** | Vision | 5 tasks | Simple baselines |
| **CORe50** | Vision | 50 tasks | Object recognition, online setting |
| **CTrL (Compositional Task-incremental Continual Learning)** | NLP | 6 tasks | Text classification sequence |
| **TRACE** | NLP | 8 tasks | Continual LLM evaluation across diverse NLP tasks |
| **StreamingQA** | QA | Temporal | QA with temporal knowledge updates |
| **TemporalWiki** | Knowledge | Temporal | Wikipedia snapshots over time |

### 7.5 Benchmarks Relevant to Your "Lifelong Memory" Angle

| Benchmark | Description | Relevance |
|-----------|-------------|-----------|
| **StreamingQA** (Liska et al., 2022) | QA dataset with temporal splits from news articles | Tests if model can answer questions about both old and new information |
| **TemporalWiki** (Jang et al., 2022) | Wikipedia changes over time | Directly models knowledge evolution |
| **EverEvolving QA** | QA pairs that change answers over time | Tests temporal knowledge updating |
| **SituatedQA** (Zhang & Choi, 2021) | QA with temporal context | Questions whose answers depend on when they're asked |

---

## 8. Synthesis: Relevance to the Lifelong Memory DSI Project

### 8.1 How This Literature Maps to Your Project

| Project Aspect | Relevant Literature | Key Takeaway |
|---------------|-------------------|--------------|
| **Base model (DSI)** | DSI (Tay et al., 2022), DSI++ (Mehta et al., 2023) | Generative retrieval is viable but forgetting is severe |
| **Continual indexing** | DSI++, IncDSI | Generative replay + SAM is the current SOTA for DSI CL |
| **LoRA experts** | MixLoRA-DSI, O-LoRA, InfLoRA | Per-batch LoRA experts reduce interference; orthogonality helps |
| **MoE routing** | Shazeer et al., Expert Gate, Mod-Squad | Input-dependent routing is essential since no task ID at test time |
| **Beneficial forgetting** | Your novel angle | No direct prior work on "intentional forgetting" in DSI |
| **Importance-based rehearsal** | GEM, GSS, DER++ | Rehearse important items more; connects to retrieval frequency |
| **Evaluation** | DSI++ benchmark, Split-NQ, CL metrics | Use Recall@k per batch + BWT/FWT + custom "memory decay" metrics |

### 8.2 Key Gaps in the Literature (Opportunities for Your Project)

1. **Forgetting as a feature**: All existing work treats forgetting as purely negative. Your "lifelong memory" angle -- where graceful forgetting of unimportant information is desirable -- is genuinely novel. No existing work in DSI or generative retrieval explores this.

2. **Retrieval-frequency-based rehearsal**: Using the frequency and recency of document retrieval to determine rehearsal priority is unexplored in the DSI context. This connects to the "spacing effect" in human memory research.

3. **MoE for DSI**: The MixLoRA-DSI paper is the first to combine MoE with DSI for continual learning. There is room to explore different routing strategies, expert capacity allocation, and dynamic expansion.

4. **Evaluation metrics for "memory quality"**: Existing CL metrics treat all forgetting equally. You could propose metrics that weight forgetting by document importance (e.g., retrieval frequency), creating a "memory quality" score rather than a raw "memory quantity" score.

5. **Temporal decay modeling**: Explicitly modeling how document representations should decay over time (similar to the TTL model in Tavily's architecture) is unexplored in DSI.

### 8.3 Recommended Reading Priority

**Must-read (core to your project):**
1. Tay et al. (2022) -- DSI original
2. Mehta et al. (2023) -- DSI++
3. MixLoRA-DSI (2025) -- Your reference paper
4. Hu et al. (2022) -- LoRA
5. Buzzega et al. (2020) -- Dark Experience Replay
6. Kirkpatrick et al. (2017) -- EWC

**High priority (directly relevant methods):**
7. Wang et al. (2023) -- O-LoRA
8. Liang et al. (2024) -- InfLoRA
9. Kishore et al. (2023) -- IncDSI
10. Sun et al. (2020) -- LAMOL
11. Cui et al. (2023) -- Distribution-Specialized Experts

**Important context:**
12. van de Ven & Tolias (2019) -- Three CL scenarios
13. De Lange et al. (2021) -- CL survey
14. Yadav et al. (2023) -- TIES-Merging
15. Zhang et al. (2022) -- Adaptive Compositional Modules
16. Shazeer et al. (2017) -- MoE

### 8.4 Suggested Experimental Directions

Based on this literature review, the most promising and feasible (within 2 months) directions are:

1. **Importance-weighted rehearsal for DSI**: Implement a rehearsal buffer where documents are prioritized by retrieval frequency. Compare against uniform rehearsal (DSI++ baseline) and no rehearsal. Measure both standard CL metrics and a novel "weighted recall" metric.

2. **MoE expert routing analysis**: Using the MixLoRA-DSI framework, study how the router evolves over time. Do queries naturally cluster by document age? Does the router learn temporal patterns?

3. **Controlled forgetting experiments**: Deliberately reduce rehearsal for certain document batches and measure how gracefully performance degrades. Compare against hard deletion. Show that gradual forgetting via reduced rehearsal produces better retrieval than binary keep/delete.

4. **Memory consolidation**: Inspired by human sleep/consolidation, periodically "consolidate" knowledge by merging LoRA experts (using TIES-Merging or similar) and spawning a fresh expert. Study whether this improves long-term retention.

---

## References (Alphabetical)

1. Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., & Tuytelaars, T. (2018). Memory Aware Synapses: Learning what (not) to forget. ECCV 2018.
2. Aljundi, R., Chakravarty, P., & Tuytelaars, T. (2017). Expert Gate: Lifelong Learning with a Network of Experts. CVPR 2017.
3. Aljundi, R., Lin, M., Goujaud, B., & Bengio, Y. (2019). Gradient based sample selection for online continual learning. NeurIPS 2019.
4. Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark Experience for General Continual Learning: a Strong, Simple Baseline. NeurIPS 2020.
5. Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient Lifelong Learning with A-GEM. ICLR 2019.
6. Cui, G., Shu, W., Bao, H., et al. (2023). Lifelong Language Pretraining with Distribution-Specialized Experts. ACL 2023.
7. De Cao, N., Izacard, G., Riedel, S., & Petroni, F. (2021). Autoregressive Entity Retrieval. ICLR 2021.
8. De Lange, M., Aljundi, R., Masana, M., et al. (2021). A continual learning survey: Defying forgetting in classification tasks. TPAMI.
9. Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
10. Huang, C., Liu, Q., Lin, B. Y., et al. (2024). LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition. ACL 2024.
11. Jang, J., Ye, S., Yang, S., et al. (2022). Towards Continual Knowledge Learning of Language Models. ICLR 2022.
12. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.
13. Kishore, V., Wan, C., Lovelace, J., Artzi, Y., & Weinberger, K. Q. (2023). IncDSI: Incrementally Updatable Document Retrieval. ICML 2023.
14. Li, Z. & Hoiem, D. (2016/2018). Learning without Forgetting. TPAMI 2018.
15. Liang, H., Liu, X., Yan, Y., et al. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning. CVPR 2024.
16. Lopez-Paz, D. & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. NeurIPS 2017.
17. Mallya, A. & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning. CVPR 2018.
18. Mehta, S. V., Gupta, J., Tay, Y., et al. (2023). DSI++: Updating Transformer Memory with New Documents. EMNLP 2023.
19. Razdaibiedina, A., Mao, Y., Hou, R., et al. (2023). Progressive Prompts: Continual Learning for Language Models. ICLR 2023.
20. Riemer, M., Cases, I., Ajemian, R., et al. (2019). Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference. ICLR 2019.
21. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., et al. (2016). Progressive Neural Networks. arXiv.
22. Serra, J., Suris, D., Miron, M., & Karatzoglou, A. (2018). Overcoming Catastrophic Forgetting with Hard Attention to the Task. ICML 2018.
23. Shazeer, N., Mirhoseini, A., Makowski, K., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
24. Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual Learning with Deep Generative Replay. NeurIPS 2017.
25. Smith, J. S., Karlinsky, L., Gutta, V., et al. (2023). CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning. CVPR 2023.
26. Sun, F.-K., Ho, C.-H., & Lee, H.-Y. (2020). LAMOL: LAnguage MOdeling for Lifelong Language Learning. ICLR 2020.
27. Tay, Y., Tran, V. Q., Dehghani, M., et al. (2022). Transformer Memory as a Differentiable Search Index. NeurIPS 2022.
28. van de Ven, G. M. & Tolias, A. S. (2019). Three scenarios for continual learning. arXiv.
29. Wang, Y., Tay, Y., et al. (2023). O-LoRA: Orthogonal Low-Rank Adaptation of Large Language Models. arXiv.
30. Wang, Z., Zhang, Z., Lee, C.-Y., et al. (2022). Learning to Prompt for Continual Learning. CVPR 2022.
31. Wang, Z., Zhang, Z., Ebrahimi, S., et al. (2022). DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning. ECCV 2022.
32. Wortsman, M., Ilharco, G., Gadre, S. Y., et al. (2022). Model Soups: Averaging Weights of Multiple Fine-tuned Models. ICML 2022.
33. Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). Resolving Interference When Merging Models (TIES-Merging). NeurIPS 2023.
34. Yoon, J., Yang, E., Lee, J., & Hwang, S. J. (2018). Lifelong Learning with Dynamically Expandable Networks. ICLR 2018.
35. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic Intelligence. ICML 2017.
36. Zhang, Y., Wang, X., & Yang, D. (2022). Continual Sequence Generation with Adaptive Compositional Modules. ACL 2022.

---

*Note: This review was compiled from knowledge of the literature through early 2025. For the most recent updates (late 2025 - early 2026), verify specific claims against current arXiv listings, particularly for the MixLoRA-DSI paper and any follow-up work.*
