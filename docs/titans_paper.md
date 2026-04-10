# Titans: Learning to Memorize at Test Time

**arXiv:2501.00663v1 [cs.LG] 31 Dec 2024**

**Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)

---

## Abstract

Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps an attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of a fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines.

---

## 1 Introduction

> "The true art of memory is the art of attention!" -- Samuel Johnson, 1787

Transformers, pure attention-based architectures (Vaswani et al. 2017), have been firmly established as state-of-the-art models in sequence modeling, mainly due to their in-context learning and ability to learn at scale (Kaplan et al. 2020). The primary building blocks of Transformers -- attention modules -- function as associative memory blocks (Bietti et al. 2024), where they learn to store key-value associations and retrieve them by computing pairwise similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer is exclusively conditioned on the direct dependencies of tokens in the current context window. This accurate modeling of dependencies, however, comes with quadratic time and memory complexity in terms of the context length. In complex real-world tasks (e.g., language modeling, video understanding, long-term time series forecasting), the context window can become extremely large, making the applicability of Transformers challenging in these downstream tasks.

To overcome the scalability issue of Transformers, recent studies aim to design different variants of linear Transformers (Kacham, Mirrokni, and Zhong 2024; Katharopoulos et al. 2020; Yang et al. 2024), where softmax is replaced by a kernel function in the attention, resulting in a significant drop in memory consumption. Despite efficiency and the ability to scale to longer context, linear Transformers do not show competitive performance compared to Transformers as the kernel trick makes the model a linear recurrent network, in which the data is compressed into a matrix-valued states. This, however, brings a contradictory fact about linear recurrent (or linear Transformers) models: On one hand, we use these linear models to enhance scalability and efficiency (linear vs. quadratic complexity), whose advantages appear for very long context; On the other hand, a very long context cannot be properly compressed in a small vector-valued or matrix-valued states.

Furthermore, beyond efficiency, most existing architectures -- ranging from Hopfield Networks to LSTMs and Transformers -- face challenges when dealing with generalization, length extrapolation, and/or reasoning, all of which are inseparable parts of many hard real-world tasks. Although these architectures draw inspiration from the human brain, each is missing: (1) a crucial component for learning process -- such as short-term memory, long-term memory, meta-memory, attending to current context, etc.; (2) how these components are interconnected systems that can operate independently; and/or (3) the ability to actively learn from data and memorize the abstraction of past history.

### Memory Perspective

Memory is a fundamental mental process and is an inseparable component of human learning. Without a properly functioning memory system, humans and animals would be restricted to basic reflexes and stereotyped behaviors. Accordingly, memory has been the inspiration for many seminal research in machine learning literature; e.g., Hopfield Networks, LSTMs, and Transformers.

Taking inspiration from the common definitions of memory and learning in neuropsychology literature, most existing architectures consider memory as a neural update caused by an input, and define learning as a process for acquiring effective and useful memory, given an objective. In this perspective:

- **RNNs** can be defined as models with a vector-valued memory module M (also called hidden state) with two main steps: (1) updates the memory using a function f(M_{t-1}, x_t) (with compression); and (2) retrieves the corresponding memory of input using a function g(M_t, x_t).

- **Transformers** can be seen as architectures with a growing memory and two similar steps: (1) updates the memory by appending the key and value to the memory (without compression), and (2) retrieves query vectors' corresponding memory by finding the similarity of query and key vectors.

The main difference between Transformers and linear Transformers is the memory structure as well as the memory updating step, in which linear Transformers compress the historical data into a fixed-size matrix-valued memory while Transformers keep all historical data (within the context length) without any compression.

This perspective motivates five key questions:
- **(Q1)** What constitutes a good structure for the memory?
- **(Q2)** What is a proper memory update mechanism?
- **(Q3)** What is a good memory retrieval process?
- **(Q4)** How to design an efficient architecture that incorporates different interconnected memory modules?
- **(Q5)** Is a deep memory module needed to effectively store/remember long past?

### Contributions and Roadmap

**Neural Memory (Section 3).** A (deep) neural long-term memory that (as a meta in-context model) learns how to memorize/store the data into its parameters at test time. Inspired by human long-term memory system, this memory module is designed so that an event that violates the expectations (being surprising) is more memorable. Surprise is measured by the gradient of the neural network with respect to the input in associative memory loss. A decaying mechanism handles limited memory, generalizing the forgetting mechanism in modern recurrent models. This mechanism is equivalent to optimizing a meta neural network with mini-batch gradient descent, momentum, and weight decay.

**Titans Architectures (Section 4).** A family of deep models with three hyper-heads: (1) Core: short-term memory (attention with limited window size); (2) Long-term Memory: neural long-term memory module; (3) Persistent Memory: learnable data-independent parameters encoding task knowledge. Three variants: memory as (i) a context (MAC), (ii) a layer (MAL), and (iii) a gated branch (MAG).

**Experimental Results (Section 5).** Titans outperform all modern recurrent models and their hybrid variants across language modeling, commonsense reasoning, recall-intensive, needle in haystack, time series forecasting, and DNA modeling tasks. They scale to larger than 2M context window size.

---

## 2 Preliminaries

### 2.1 Backgrounds

#### Attention

Given input x in R^{N x d_in}, causal attention computes output y in R^{N x d_in} based on softmax over input dependent key, value, and query matrices:

```
Q = xW_Q, K = xW_K, V = xW_V

y_i = sum_{j=1}^{i} [exp(Q_i^T K_j / sqrt(d_in)) * V_j] / sum_{l=1}^{i} exp(Q_i^T K_l / sqrt(d_in))
```

#### Efficient Attentions (Linear Attentions)

Softmax is replaced with an alternative kernel function phi(.,.), such that phi(x,y) = phi(x)phi(y). In recurrent format:

```
M_t = M_{t-1} + K_t^T V_t      (Write)
y_t = Q_t M_t                   (Read)
```

#### Modern Linear Models and Their Memory Perspective

General form of recurrent neural network:

```
M_t = f(M_{t-1}, x_t)    -- Write Operation
y_t = g(M_t, x_t)         -- Read Operation
```

The additive nature of linear Transformers' memory write results in memory overflow for long contexts. Two directions address this:

1. **Adding forget mechanism:** Adaptive data-dependent forgetting gates (Mamba, Mamba2, Griffin, xLSTM, LRU)
2. **Improving the write operation:** Delta Rule -- before adding a memory, the model first removes its past value (DeltaNet, Gated DeltaNet)

---

## 3 Learning to Memorize at Test Time

### 3.1 Long-term Memory

**Key idea:** Treat training of a long-term memory as an online learning problem, compressing past information x_1,...,x_{t-1} into the parameters of a neural memory module M_t.

#### Learning Process and Surprise Metric

An event that violates expectations (is surprising) is more memorable. Surprise is defined as the gradient with respect to the input:

```
M_t = M_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
                          ^-- Surprise
```

This can miss information after big surprising moments (gradient becomes small, stuck in flat areas). Improvement breaks surprise into past surprise and momentary surprise:

```
M_t = M_{t-1} + S_t
S_t = eta_t * S_{t-1}        - theta_t * grad(loss(M_{t-1}; x_t))
      ^-- Past Surprise         ^-- Momentary Surprise
```

This is equivalent to **gradient descent with momentum**, where S_t is the momentum element. The term eta_t is a data-dependent surprise decay, and theta_t controls how much momentary surprise is incorporated.

#### Objective (Associative Memory Loss)

```
k_t = x_t W_K,  v_t = x_t W_V

loss(M_{t-1}; x_t) = ||M_{t-1}(k_t) - v_t||^2
```

The memory learns to memorize the mapping between keys and values at test time. Training of the memory is in the inner-loop; W_K and W_V are hyperparameters optimized in the outer-loop.

#### Forgetting Mechanism

Adaptive forgetting for very large sequences:

```
M_t = (1 - alpha_t) * M_{t-1} + S_t
S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
```

Where alpha_t is a data-dependent weight decay that flexibly controls the memory (alpha_t -> 0: keep past; alpha_t -> 1: clear memory). This generalizes gating mechanisms in modern RNNs.

#### Memory Architecture

Simple MLPs with L_M >= 1 layers. Deep memory modules (L_M >= 2) are strictly more expressive than linear models, confirmed experimentally.

#### Retrieving from Memory

Forward pass without weight update:

```
q_t = x_t W_Q
y_t = M*(q_t)    (inference, no weight adjustment)
```

### 3.2 Parallelizing Long-term Memory Training

Training the long-term memory is equivalent to training a meta model with mini-batch gradient descent, momentum, and weight decay. The process is parallelized by:

1. Splitting the sequence into chunks of size b >= 1
2. Computing mini-batch gradient descent within each chunk using matmuls
3. Using parallel associative scan for the momentum term (linear recurrence)

For data-dependent parameters, they can be made functions of their chunk (trading expressivity for speed). When eta and theta are time-invariant inside each chunk, the equation becomes an LTI system computable by global convolution.

### 3.3 Persistent Memory

Learnable but input-independent parameters P = [p_1, p_2, ..., p_{N_p}] appended to the start of the sequence:

```
x_new = [p_1, p_2, ..., p_{N_p}] || x
```

**Three perspectives:**
1. **Memory Perspective:** Input-independent parameters store abstraction of task knowledge
2. **Feedforward Network Perspective:** Equivalent to data-independent attention weights (replaces FFN layers)
3. **Technical Perspective:** Mitigates attention's implicit bias toward initial tokens by redistributing attention weights

---

## 4 How to Incorporate Memory?

### 4.1 Memory as a Context (MAC)

Sequence is chunked into fixed-size segments. For each incoming segment S(t):

1. **Retrieve:** h_t = M*_{t-1}(q_t) -- retrieve past information corresponding to current segment
2. **Concatenate:** S_tilde = [persistent_memory || h_t || S(t)]
3. **Attend:** y_t = Attn(S_tilde) -- full causal attention within the segment
4. **Update memory:** M_t = M_{t-1}(y_t)
5. **Output:** o_t = y_t (x) M*_t(y_t) -- element-wise gating

**Advantages:**
- Attention decides whether long-term memory information is needed given current data
- Attention helps memory store only useful information (better capacity management)
- At test time: persistent memory is fixed, attention does in-context learning, long-term memory continues learning

### 4.2 Gated Memory (MAG)

Two parallel branches combined by gating:

```
x_tilde = [persistent_memory || x]
y = SW-Attn*(x_tilde)        -- sliding window attention (short-term)
o = y (x) M(x_tilde)          -- gated combination with neural memory (long-term)
```

Sliding window attention acts as precise short-term memory; neural memory acts as fading long-term memory.

### 4.3 Memory as a Layer (MAL)

Sequential layer-wise design:

```
x_tilde = [persistent_memory || x]
y = M(x_tilde)               -- neural memory compresses context
o = SW-Attn(y)                -- sliding window attention
```

**Drawback:** Power is limited by each layer independently. A variant without attention (LMM alone) is also evaluated.

### 4.4 Architectural Details

- Residual connections in all blocks
- SiLU activation for query, key, and value projections
- L2-norm for queries and keys
- 1D depthwise-separable convolution after Q, K, V projections
- Normalization and gating with a linear layer before final output

**Theorem 4.1.** Titans are capable of solving problems beyond TC^0, meaning they are theoretically more expressive than Transformers and most modern linear recurrent models in state tracking tasks.

---

## 5 Experiments

### 5.1 Experimental Setup

**Models:** Titans variants (MAC, MAG, MAL, LMM alone) at scales: 170M, 340M, 400M, 760M parameters. Trained on FineWeb-Edu dataset (15B tokens for smaller models, 30B tokens for 760M).

**Baselines:** Transformer++, RetNet, GLA, Mamba, Mamba2, DeltaNet, TTT, Gated DeltaNet, Samba, plus GPT-4, Llama3 with RAG, RecurrentGemma2-9B, Mistral for NIAH tasks.

### 5.2 Language Modeling

Key findings at 760M scale:
- **Titans (MAC):** Best Wiki ppl 18.61, LMB ppl 19.86, Avg reasoning acc 52.51
- **Titans (MAG):** Wiki ppl 19.07, LMB ppl 20.33, Avg reasoning acc 52.50
- **Titans (LMM) alone:** Wiki ppl 20.04, LMB ppl 21.96, Avg reasoning acc 51.56
- **Transformer++:** Wiki ppl 25.21, LMB ppl 27.64, Avg reasoning acc 48.69
- **Gated DeltaNet:** Wiki ppl 22.71, LMB ppl 22.09, Avg reasoning acc 51.08

Neural memory module alone (no attention) achieves the best performance among all non-hybrid models.

### 5.3 Needle in a Haystack (S-NIAH)

At 16K context on S-NIAH-W task:
- **Titans (MAC):** 95.2%
- **Titans (MAG):** 88.2%
- **Titans (MAL):** 90.4%
- **Titans (LMM):** 80.6%
- **TTT:** 0.0%
- **Mamba2:** 0.0%
- **DeltaNet:** 0.0%

Titans maintain high accuracy as context length increases; baselines collapse.

### 5.4 BABILong Benchmark

Titans (MAC) outperforms all baselines including GPT-4 and GPT-4o-mini on reasoning across extremely long documents. Even with ~70x fewer parameters than Llama3.1-8B with RAG, Titans performs better.

### 5.5 Effect of Deep Memory

- Deeper memory (L_M = 1,2,3,4) consistently improves perplexity across all sequence lengths
- Deeper memory is more robust to sequence length, especially with fewer parameters
- Training throughput scales linearly with context length for all depths
- Trade-off: deeper memory = slower training but better performance

### 5.6 Time Series Forecasting

Neural memory module outperforms all baselines (Mamba-based, linear-based, Transformer-based) on ETT, ECL, Traffic, and Weather benchmarks.

### 5.7 DNA Modeling

LMM is competitive with state-of-the-art architectures across GenomicsBenchmarks tasks.

### 5.8 Efficiency

- Neural memory is slightly slower than Mamba2 and Gated DeltaNet (due to deep memory and more expressive transitions)
- Titans (MAL) is faster than baselines due to FlashAttention optimization
- All models scale linearly with context length

### 5.9 Ablation Study

All components positively contribute to performance. Ranked by importance:
1. Weight decay (forgetting mechanism) -- greatest contribution
2. Momentum in surprise measure
3. Convolution
4. Persistent memory

Architecture comparison:
- MAC and MAG have close performance in language modeling/reasoning
- MAC significantly better in long-context NIAH
- Both MAC and MAG outperform MAL

---

## Appendix C: LMM as a Sequence Model (Key Relationships)

### LMM Generalizes Gated DeltaNet

Setting eta_t = 0 in LMM recovers Gated DeltaNet. LMM adds:
- Momentum-based update rule (vs. momentary surprise only)
- Deep memory (vs. linear/matrix-valued)
- Non-linear recurrence (inter-chunk non-linear, intra-chunk linear)

### LMM Generalizes Longhorn

Same loss function but Longhorn uses implicit online learning without forgetting gate. LMM adds:
- Momentum-based rule
- Deep memory
- Non-linear recurrence
- Forget gate

### LMM Generalizes TTT Layer

Key differences:
1. **Forgetting mechanism:** TTT has no way to clear memory for long sequences
2. **Momentum-based update rule:** TTT uses momentary surprise only
3. **Deep memory:** TTT deep modules were not experimentally evaluated

LMM is the first linear recurrent model with momentum-based update rule.

---

## Key References

- Vaswani et al. 2017 -- Attention Is All You Need (Transformers)
- Gu and Dao 2024 -- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Dao and Gu 2024 -- Transformers are SSMs (Mamba2)
- Yang, Kautz, and Hatamizadeh 2024 -- Gated Delta Networks
- Yang, Wang, Zhang, et al. 2024 -- DeltaNet
- Sun et al. 2024 -- TTT: Learning to (learn at test time)
- Schmidhuber 1992 -- Fast Weight Programs
- Katharopoulos et al. 2020 -- Linear Transformers
