# 📚 Complete GPT-2 Implementation Guide

**A Comprehensive Deep-Dive into Every Aspect of this LLM Implementation**

> This document provides line-by-line explanations of data flow, tensor shapes, mathematical operations, and PyTorch implementations throughout the entire project.

---

## 📑 **Master Index**

### Part I: Project Overview
1. [Introduction](#1-introduction)
2. [Project Architecture](#2-project-architecture)
3. [Technology Stack](#3-technology-stack)

### Part II: Data Pipeline Deep Dive
4. [Tokenization Process](#4-tokenization-process)
   - 4.1 [Text to Token IDs](#41-text-to-token-ids)
   - 4.2 [Tiktoken GPT-2 Encoder](#42-tiktoken-gpt-2-encoder)
   - 4.3 [Token Count vs Vocabulary Size](#43-token-count-vs-vocabulary-size)

5. [Dataset Construction](#5-dataset-construction)
   - 5.1 [GPTDatasetV1 Implementation](#51-gptdatasetv1-implementation)
   - 5.2 [Sliding Window Mechanism](#52-sliding-window-mechanism)
   - 5.3 [Input-Target Pair Creation](#53-input-target-pair-creation)

6. [Data Loading Pipeline](#6-data-loading-pipeline)
   - 6.1 [DataLoader Configuration](#61-dataloader-configuration)
   - 6.2 [Batching and Shuffling](#62-batching-and-shuffling)
   - 6.3 [Shape Transformations](#63-shape-transformations)

### Part III: Model Architecture Line-by-Line
7. [Embedding Layer](#7-embedding-layer)
   - 7.1 [Token Embeddings](#71-token-embeddings)
   - 7.2 [Position Embeddings](#72-position-embeddings)
   - 7.3 [Embedding Addition](#73-embedding-addition)

8. [Multi-Head Attention](#8-multi-head-attention)
   - 8.1 [Q, K, V Projections](#81-q-k-v-projections)
   - 8.2 [Head Splitting](#82-head-splitting)
   - 8.3 [Attention Computation](#83-attention-computation)
   - 8.4 [Flash Attention vs Standard](#84-flash-attention-vs-standard)

9. [Feed-Forward Network](#9-feed-forward-network)
   - 9.1 [Expansion Layer](#91-expansion-layer)
   - 9.2 [GELU Activation](#92-gelu-activation)
   - 9.3 [Projection Layer](#93-projection-layer)

10. [Transformer Block](#10-transformer-block)
    - 10.1 [Pre-Norm Architecture](#101-pre-norm-architecture)
    - 10.2 [Residual Connections](#102-residual-connections)
    - 10.3 [Complete Forward Pass](#103-complete-forward-pass)

### Part IV: Training Process
11. [Training Loop Breakdown](#11-training-loop-breakdown)
    - 11.1 [Forward Pass](#111-forward-pass)
    - 11.2 [Loss Calculation](#112-loss-calculation)
    - 11.3 [Backward Pass](#113-backward-pass)
    - 11.4 [Optimizer Step](#114-optimizer-step)

12. [Mixed Precision Training](#12-mixed-precision-training)
    - 12.1 [Autocast Context](#121-autocast-context)
    - 12.2 [Gradient Scaling](#122-gradient-scaling)
    - 12.3 [Memory Optimization](#123-memory-optimization)

13. [Gradient Accumulation](#13-gradient-accumulation)
    - 13.1 [Accumulation Logic](#131-accumulation-logic)
    - 13.2 [Effective Batch Size](#132-effective-batch-size)
    - 13.3 [Weight Update Timing](#133-weight-update-timing)

### Part V: Validation and Generation
14. [Validation Process](#14-validation-process)
    - 14.1 [Evaluation Mode](#141-evaluation-mode)
    - 14.2 [Loss Computation](#142-loss-computation)
    - 14.3 [Metric Tracking](#143-metric-tracking)

15. [Text Generation](#15-text-generation)
    - 15.1 [Autoregressive Sampling](#151-autoregressive-sampling)
    - 15.2 [Temperature Scaling](#152-temperature-scaling)
    - 15.3 [Top-K Filtering](#153-top-k-filtering)

### Part VI: Dataset Collection
16. [HuggingFace Dataset Integration](#16-huggingface-dataset-integration)
    - 16.1 [FineWeb-Edu](#161-fineweb-edu)
    - 16.2 [Cosmopedia-v2](#162-cosmopedia-v2)
    - 16.3 [Streaming and Processing](#163-streaming-and-processing)

### Part VII: Advanced Topics
17. [Model Checkpointing](#17-model-checkpointing)
18. [Flash Attention Implementation](#18-flash-attention-implementation)
19. [Performance Optimization](#19-performance-optimization)

---

# Part I: Project Overview

## 1. Introduction

This implementation provides a **production-ready GPT-2 model** (124M parameters) built entirely from scratch using PyTorch. Every component is explained with:

- **Mathematical formulations**
- **Shape transformations**
- **PyTorch code equivalents**
- **Visual tensor operations**
- **Real example data**

**Key Features:**
- ✅ Flash Attention (2-4x speedup on GPU)
- ✅ Mixed Precision Training (FP16/FP32)
- ✅ Gradient Accumulation (large effective batch sizes)
- ✅ Modular, well-documented code

---

## 2. Project Architecture

```
GPT-2 Model (124M parameters)
│
├── Input Processing
│   ├── Tokenization (tiktoken)
│   ├── Dataset Creation (sliding window)
│   └── Data Loading (batching)
│
├── Model Layers
│   ├── Embedding (Token + Position)
│   ├── 12× Transformer Blocks
│   │   ├── Multi-Head Attention
│   │   ├── Feed-Forward Network
│   │   └── Layer Normalization
│   └── Output Projection
│
├── Training Pipeline
│   ├── Forward Pass
│   ├── Loss Calculation (Cross-Entropy)
│   ├── Backward Pass (Gradients)
│   └── Optimizer (AdamW)
│
└── Generation
    ├── Autoregressive Sampling
    ├── Temperature/Top-K
    └── Decoding
```

---

## 3. Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | PyTorch | 2.9.1+ | Deep learning |
| Tokenizer | tiktoken | Latest | GPT-2 BPE |
| Data | HuggingFace Datasets | Latest | Training data |
| Precision | CUDA AMP | Built-in | Mixed precision |
| Attention | Flash Attention | PyTorch 2.0+ | Speed optimization |

---

# Part II: Data Pipeline Deep Dive

## 4. Tokenization Process

### 4.1 Text to Token IDs

**Goal:** Convert raw text into integer token IDs that the model can process.

**Example:**
```python
# Input
text = "Every effort moves you"

# Step 1: Tokenization
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)

# Output
token_ids = [6109, 3626, 6100, 345]
#            "Every" "effort" "moves" "you"
```

**Shape Transformation:**
```
String (variable length)
    ↓
List[int] (length = 4)
    ↓
[6109, 3626, 6100, 345]
```

---

### 4.2 Tiktoken GPT-2 Encoder

**What is BPE (Byte Pair Encoding)?**

BPE is a compression algorithm that:
1. Starts with individual characters
2. Merges frequently occurring pairs
3. Creates subword units

**Example Encoding:**
```python
# Word: "unhappiness"
# BPE splits into: ["un", "happ", "iness"]

tokenizer.encode("unhappiness")
# [403, 71, 31803]
#  "un" "happ" "iness"
```

**PyTorch Code Equivalent:**
```python
# Pseudocode for BPE encoding
def encode(text):
    # 1. Split into bytes
    bytes_list = text.encode('utf-8')
    
    # 2. Apply learned merges
    tokens = apply_bpe_merges(bytes_list, merge_rules)
    
    # 3. Map to token IDs
    token_ids = [vocab[token] for token in tokens]
    
    return token_ids
```

---

### 4.3 Token Count vs Vocabulary Size

**Critical Distinction:**

| Concept | Definition | Example | Type |
|---------|-----------|---------|------|
| **Token Count** | Number of tokens in YOUR text | 4 | Variable |
| **Vocabulary Size** | Total tokens the model knows | 50,257 | Fixed |

**Visual:**
```
Your text: "Every effort moves you"
           ↓ tokenize
Token IDs: [6109, 3626, 6100, 345]  ← Token Count = 4

Model vocabulary: [token_0, token_1, ..., token_50256]
                                              ↑
                                    Vocabulary Size = 50,257
```

**Why it matters:**
- Token Count: Determines sequence length
- Vocabulary Size: Determines embedding matrix size

---

## 5. Dataset Construction

### 5.1 GPTDatasetV1 Implementation

**Purpose:** Convert continuous text into (input, target) pairs for training.

**Code:**
```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize entire text
        token_ids = tokenizer.encode(txt)  # [token1, token2, ..., tokenN]
        
        # Create sliding windows
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
```

**Line-by-Line Breakdown:**

#### **Line 1:** `token_ids = tokenizer.encode(txt)`

**What it does:**
- Converts entire text to token IDs
- Returns Python list of integers

**Example:**
```python
txt = "Every effort moves you forward"
token_ids = [6109, 3626, 6100, 345, 2651]
#            len = 5 tokens
```

**Shape:**
```
Input: String
Output: List[int] with length = number_of_words
```

---

#### **Line 2:** `for i in range(0, len(token_ids) - max_length, stride):`

**What it does:**
- Creates sliding window positions
- `stride` controls overlap between windows

**Math:**
```
total_tokens = 927,443
max_length = 256
stride = 64

Number of samples = (927,443 - 256) // 64 = 14,487 samples
```

**Visual:**
```
Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]

max_length = 4
stride = 2

Window 1: [1, 2, 3, 4]    (i=0)
Window 2:    [3, 4, 5, 6] (i=2, overlaps!)
Window 3:       [5, 6, 7, 8] (i=4)
```

---

#### **Line 3-4:** `input_chunk` and `target_chunk`

**What it does:**
- `input_chunk`: Current sequence
- `target_chunk`: Same sequence shifted by 1 (what to predict)

**Example:**
```python
i = 0
max_length = 4

input_chunk = token_ids[0:4] = [6109, 3626, 6100, 345]
target_chunk = token_ids[1:5] = [3626, 6100, 345, 2651]
```

**Visual Alignment:**
```
Input:  [6109,  3626,  6100,  345]
Target:        [3626,  6100,  345,  2651]
         ↓      ↓      ↓      ↓
Predict "effort" "moves" "you" "forward"
  given  "Every" "effort" "moves" "you"
```

**PyTorch Tensor Creation:**
```python
self.input_ids.append(torch.tensor(input_chunk))
# Creates: tensor([6109, 3626, 6100, 345])
# dtype: torch.int64
# shape: [4]
```

---

### 5.2 Sliding Window Mechanism

**Why use sliding windows?**

1. **Limited context:** Model can only see `max_length` tokens
2. **Data efficiency:** Creates multiple training samples from one text
3. **Structured learning:** Each sample teaches next-word prediction

**Complete Example:**

```python
text = "I HAD always thought Jack Gisburn rather a cheap genius"
tokens = [40, 367, 2885, 1464, 3619, 402, 271, 10899, 257, 7026, 33849]
#         11 tokens total

max_length = 4
stride = 2

# Sample 1 (i=0)
input:  [40, 367, 2885, 1464]     # "I HAD always thought"
target: [367, 2885, 1464, 3619]   # "HAD always thought Jack"

# Sample 2 (i=2)
input:  [2885, 1464, 3619, 402]   # "always thought Jack Gisburn"
target: [1464, 3619, 402, 271]    # "thought Jack Gisburn rather"

# Sample 3 (i=4)
input:  [3619, 402, 271, 10899]   # "Jack Gisburn rather a"
target: [402, 271, 10899, 257]    # "Gisburn rather a cheap"

# ...and so on
```

**Total samples created:**
```
(11 - 4) // 2 = 3 samples
```

---

### 5.3 Input-Target Pair Creation

**Goal:** Teach the model to predict the next token.

**Mathematical Formulation:**

For position `j` in a sequence:
```
P(token_j+1 | token_1, token_2, ..., token_j)
```

**Training Objective:**

Given input `[w1, w2, w3, w4]`, predict `[w2, w3, w4, w5]`

**Loss Calculation:**

```python
# For each position:
Position 0: Predict w2 given [w1]
Position 1: Predict w3 given [w1, w2]
Position 2: Predict w4 given [w1, w2, w3]
Position 3: Predict w5 given [w1, w2, w3, w4]

# Cross-entropy loss averages over all positions
```

---

## 6. Data Loading Pipeline

### 6.1 DataLoader Configuration

**Code:**
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,                # GPTDatasetV1 instance
    batch_size=2,          # Process 2 samples together
    shuffle=True,          # Randomize order each epoch
    drop_last=False        # Keep incomplete final batch
)
```

**Line-by-Line:**

#### **`batch_size=2`**

**What it does:** Combines 2 samples into one batch

**Before batching (from dataset):**
```python
Sample 1: tensor([40, 367, 2885, 1464])      # shape: [4]
Sample 2: tensor([2885, 1464, 3619, 402])    # shape: [4]
```

**After batching (from dataloader):**
```python
Batch: tensor([[40, 367, 2885, 1464],
               [2885, 1464, 3619, 402]])     # shape: [2, 4]
```

**PyTorch Operation:**
```python
# Internally, DataLoader does:
batch = torch.stack([sample1, sample2], dim=0)
# Creates new dimension at position 0 (batch dimension)
```

---

#### **`shuffle=True`**

**What it does:** Randomizes sample order each epoch

**Epoch 1:**
```
[sample_5, sample_2, sample_8, sample_1, sample_3, ...]
```

**Epoch 2 (different order):**
```
[sample_3, sample_9, sample_1, sample_5, sample_2, ...]
```

**Why?** Prevents overfitting to sequential patterns

---

#### **`drop_last=False`**

**What it does:** Keeps the last batch even if incomplete

**Example:**
```
Total samples: 100
Batch size: 8

Batches: 12 full batches (96 samples) + 1 partial batch (4 samples)

drop_last=False: Use all 100 samples (13 batches)
drop_last=True: Discard last 4 (12 batches only)
```

---

### 6.2 Batching and Shuffling

**Complete Data Flow:**

```python
# Before DataLoader
GPTDatasetV1.__getitem__(0) → tensor([40, 367, ...])    # [max_length]
GPTDatasetV1.__getitem__(1) → tensor([2885, 1464, ...]) # [max_length]

# DataLoader batching
for inputs, targets in dataloader:
    print(inputs.shape)   # torch.Size([batch_size, max_length])
    print(targets.shape)  # torch.Size([batch_size, max_length])
```

**Example with real config:**
```python
batch_size = 2
max_length = 256

# One batch from dataloader
inputs.shape = [2, 256]
targets.shape = [2, 256]

# What's inside
inputs = tensor([
    [token_ids for sample 1],  # 256 token IDs
    [token_ids for sample 2]   # 256 token IDs
])
```

---

### 6.3 Shape Transformations

**Complete Pipeline:**

```
Raw Text (string)
    ↓ tokenizer.encode()
Token IDs (list) [N tokens]
    ↓ GPTDatasetV1 (sliding window)
Individual Samples (tensors) [max_length]
    ↓ DataLoader (batching)
Batched Samples (tensors) [batch_size, max_length]
    ↓ Model Embedding
Embedded Tensors (float) [batch_size, max_length, emb_dim]
```

**Example with numbers:**
```
"Every effort moves" (14 characters)
    ↓
[6109, 3626, 6100] (3 tokens)
    ↓ sliding window with max_length=256
tensor([6109, 3626, 6100, ...]   ) (1 sample, padded to 256)
    ↓ batching with batch_size=2
tensor([[6109, 3626, ...],
        [10899, 257, ...]])        (shape: [2, 256])
    ↓ embedding with emb_dim=768
tensor([[[0.01, -0.78, ...],     (shape: [2, 256, 768])
         [0.45, 0.12, ...]],
        ...])
```

---

# Part III: Model Architecture Line-by-Line

## 7. Embedding Layer

### 7.1 Token Embeddings

**Purpose:** Convert integer token IDs to dense float vectors

**Code:**
```python
self.tok_emb = nn.Embedding(vocab_size, emb_dim)
# vocab_size = 50,257
# emb_dim = 768

tok_embeds = self.tok_emb(in_idx)
```

**What happens inside:**

#### **Input:**
```python
in_idx = tensor([[6109, 3626, 6100]])
# shape: [1, 3]
# dtype: int64
```

#### **Embedding Matrix:**
```python
self.tok_emb.weight.shape = [50257, 768]
# 50,257 tokens × 768 dimensions

# It's a lookup table:
# Row 0: embedding for token 0
# Row 1: embedding for token 1
# ...
# Row 6109: embedding for token 6109 ("Every")
# ...
# Row 50256: embedding for token 50256
```

#### **Lookup Operation:**
```python
# For each token ID, get its corresponding row
tok_embeds[0, 0, :] = self.tok_emb.weight[6109, :]  # "Every"
tok_embeds[0, 1, :] = self.tok_emb.weight[3626, :]  # "effort"
tok_embeds[0, 2, :] = self.tok_emb.weight[6100, :]  # "moves"
```

#### **Output:**
```python
tok_embeds.shape = [1, 3, 768]
#                   ↑  ↑  ↑
#                   │  │  └── 768-dim vector for each token
#                   │  └────── 3 tokens
#                   └─────────── 1 batch

# Example values (random initialization):
tok_embeds = tensor([[[  0.0123, -0.7821,  0.2344, ...,  0.4512],  # "Every"
                      [  0.4512,  0.1234, -0.5678, ...,  0.9876],  # "effort"
                      [ -0.1523,  0.8765,  0.3241, ...,  0.6754]]])# "moves"
```

**PyTorch Equivalent:**
```python
# Manual implementation
def embedding_lookup(token_ids, embedding_matrix):
    batch_size, seq_len = token_ids.shape
    emb_dim = embedding_matrix.shape[1]
    
    output = torch.zeros(batch_size, seq_len, emb_dim)
    
    for b in range(batch_size):
        for s in range(seq_len):
            token_id = token_ids[b, s]
            output[b, s, :] = embedding_matrix[token_id, :]
    
    return output
```

---

### 7.2 Position Embeddings

**Purpose:** Add positional information (word order matters!)

**Why needed?**
```
"dog bites man" ≠ "man bites dog"
```

Transformers have no inherent notion of order, so we add position embeddings.

**Code:**
```python
self.pos_emb = nn.Embedding(context_length, emb_dim)
# context_length = 256 (max sequence length we support)
# emb_dim = 768

pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
```

**Line-by-Line:**

#### **`torch.arange(seq_len)`**

Creates position indices:
```python
seq_len = 3
positions = torch.arange(3)
# tensor([0, 1, 2])
```

#### **`self.pos_emb(positions)`**

Looks up embeddings for each position:
```python
pos_embeds.shape = [3, 768]
#                   ↑  ↑
#                   │  └── 768-dim vector
#                   └────── 3 positions

# Example values:
pos_embeds = tensor([[0.0512, -0.3421,  0.8765, ...],  # Position 0
                     [0.2341,  0.5678, -0.1234, ...],  # Position 1
                     [0.6754, -0.9876,  0.3241, ...]])  # Position 2
```

**Adding Batch Dimension:**
```python
# pos_embeds is [seq_len, emb_dim]
# tok_embeds is [batch, seq_len, emb_dim]

# Broadcasting: pos_embeds automatically expands to match
```

---

### 7.3 Embedding Addition

**Combine token and position information:**

```python
x = tok_embeds + pos_embeds
```

**Shape Analysis:**

```python
tok_embeds.shape = [1, 3, 768]  # [batch, seq, emb]
pos_embeds.shape = [   3, 768]  # [seq, emb]

# Broadcasting rules:
# pos_embeds expands to [1, 3, 768]
# Then element-wise addition

x.shape = [1, 3, 768]
```

**Visual Example:**

```
Token Embedding for "Every" at position 0:
  [0.01, -0.78, 0.23, ...]

Position Embedding for position 0:
  [0.05, -0.34, 0.88, ...]

Combined:
  [0.06, -1.12, 1.11, ...]  ← Element-wise sum
```

**Complete Example:**

```python
# Input
in_idx = tensor([[6109, 3626, 6100]])  # "Every effort moves"

# Step 1: Token embeddings
tok_emb = self.tok_emb(in_idx)
# Shape: [1, 3, 768]

# Step 2: Position embeddings
positions = torch.arange(3)  # [0, 1, 2]
pos_emb = self.pos_emb(positions)
# Shape: [3, 768]

# Step 3: Add
x = tok_emb + pos_emb
# Shape: [1, 3, 768]

# Now each token has:
# - Semantic meaning (from token embedding)
# - Position information (from position embedding)
```

---

## 8. Multi-Head Attention

### 8.1 Q, K, V Projections

**Purpose:** Transform input into Query, Key, Value representations

**Code:**
```python
self.W_query = nn.Linear(d_in, d_out, bias=False)
self.W_key = nn.Linear(d_in, d_out, bias=False)
self.W_value = nn.Linear(d_in, d_out, bias=False)

queries = self.W_query(x)
keys = self.W_key(x)
values = self.W_value(x)
```

**Input:**
```python
x.shape = [batch, seq_len, emb_dim]
        = [1, 3, 768]
```

**Linear Transformation:**

Each `nn.Linear(768, 768)` performs:
```
output = input @ weight.T
```

**Weight Matrix:**
```python
self.W_query.weight.shape = [768, 768]

# Matrix multiplication:
queries = x @ W_query.T
        = [1, 3, 768] @ [768, 768]
        = [1, 3, 768]
```

**Visual:**

```
Input x (position 0):
  [0.06, -1.12, 1.11, ..., 0.45]  (768 values)
      ↓ multiply by W_query
Query (position 0):
  [0.23, 0.45, -0.67, ..., 0.89]  (768 values, different!)
```

**All Three Projections:**
```python
queries.shape = [1, 3, 768]  # What am I looking for?
keys.shape = [1, 3, 768]     # What do I contain?
values.shape = [1, 3, 768]   # What information do I have?
```

---

### 8.2 Head Splitting

**Purpose:** Divide attention across multiple "heads" for parallel processing

**Code:**
```python
# Config
num_heads = 12
head_dim = 768 // 12 = 64

# Reshape
queries = queries.view(batch, seq_len, num_heads, head_dim)
# Shape: [1, 3, 12, 64]

queries = queries.transpose(1, 2)
# Shape: [1, 12, 3, 64]
```

**Step-by-Step:**

#### **Before .view():**
```python
queries.shape = [1, 3, 768]

# Conceptually:
queries[0, 0, :] = [q0, q1, q2, ..., q767]  # All 768 dims for position 0
```

#### **After .view():**
```python
queries.shape = [1, 3, 12, 64]
#                ↑  ↑  ↑   ↑
#                │  │  │   └── 64 dims per head
#                │  │  └────── 12 heads
#                │  └─────────── 3 positions
#                └────────────── 1 batch

# The 768 dims are split into 12 groups of 64:
queries[0, 0, 0, :] = [q0, q1, ..., q63]      # Head 0, position 0
queries[0, 0, 1, :] = [q64, q65, ..., q127]   # Head 1, position 0
...
queries[0, 0, 11, :] = [q704, ..., q767]     # Head 11, position 0
```

#### **After .transpose(1, 2):**
```python
queries.shape = [1, 12, 3, 64]
#                ↑  ↑   ↑  ↑
#                │  │   │  └── 64 dims per head
#                │  │   └────── 3 positions
#                │  └─────────── 12 heads (moved to dim 1!)
#                └────────────── 1 batch

# Why? So we can compute attention for each head independently
```

**Visual:**

```
Original: [batch, seq, 768]
             ↓ view
Split: [batch, seq, 12, 64]
             ↓ transpose(1,2)
Ready: [batch, 12, seq, 64]

Now dimension 1 represents heads:
batch[0, head_0, :, :] → attention for head 0
batch[0, head_1, :, :] → attention for head 1
...
```

---

### 8.3 Attention Computation

**Standard Attention Formula:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Code:**
```python
# queries.shape = [1, 12, 3, 64]
# keys.shape = [1, 12, 3, 64]

# Step 1: Compute scores
attn_scores = queries @ keys.transpose(2, 3)
# Shape: [1, 12, 3, 3]

# Step 2: Scale
attn_scores = attn_scores / math.sqrt(64)

# Step 3: Apply causal mask
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
attn_scores.masked_fill_(mask, -torch.inf)

# Step 4: Softmax
attn_weights = torch.softmax(attn_scores, dim=-1)
# Shape: [1, 12, 3, 3]

# Step 5: Apply to values
context = attn_weights @ values
# Shape: [1, 12, 3, 64]
```

**Line-by-Line Breakdown:**

#### **Step 1: `queries @ keys.transpose(2, 3)`**

**Matrix multiplication:**
```python
queries.shape = [1, 12, 3, 64]
keys.T.shape = [1, 12, 64, 3]  # After transpose

result = [1, 12, 3, 3]
```

**What it computes:**

For each head, create a 3×3 matrix of dot products:

```
           Key_0  Key_1  Key_2
Query_0  [score  score  score]
Query_1  [score  score  score]
Query_2  [score  score  score]
```

**Example (head 0):**
```python
# Position 0's query compared to all keys:
scores[0, 0, 0, :] = [
    dot(Q_0, K_0),  # How much position 0 attends to itself
    dot(Q_0, K_1),  # How much position 0 attends to position 1
    dot(Q_0, K_2)   # How much position 0 attends to position 2
]

# If queries and keys are similar, dot product is high
```

**Numerical Example:**
```python
Q_0 = [0.5, 0.3, 0.2, ...]  (64 dims)
K_0 = [0.5, 0.3, 0.2, ...]  (64 dims, same!)

dot(Q_0, K_0) = 0.5*0.5 + 0.3*0.3 + 0.2*0.2 + ...
              ≈ 18.5  (high score - very similar!)

K_1 = [-0.2, 0.8, -0.5, ...]  (64 dims, different)
dot(Q_0, K_1) = 0.5*(-0.2) + 0.3*0.8 + ...
              ≈ 2.1  (lower score - less similar)
```

---

#### **Step 2: Scaling by `√d_k`**

**Why?**

Without scaling, dot products grow with dimension size:
- 64-dim vectors → typical dot product ~30
- 768-dim vectors → typical dot product ~300

Large values → softmax becomes too sharp → training instability

**Formula:**
```python
d_k = 64  # head dimension
scaled_scores = attn_scores / math.sqrt(64)
              = attn_scores / 8.0
```

**Example:**
```python
Before: [18.5, 2.1, -5.3]
After:  [2.31, 0.26, -0.66]  # Much smaller range
```

---

#### **Step 3: Causal Masking**

**Purpose:** Prevent looking ahead at future tokens (autoregressive)

**Mask Creation:**
```python
mask = torch.triu(torch.ones(3, 3), diagonal=1)
# tensor([[0., 1., 1.],
#         [0., 0., 1.],
#         [0., 0., 0.]])

mask_bool = mask.bool()
# True means "mask out" (set to -inf)
```

**Visual:**
```
Position 0 can see: [0]           ← only itself
Position 1 can see: [0, 1]        ← positions 0 and 1
Position 2 can see: [0, 1, 2]     ← all positions
```

**Applying mask:**
```python
# Before masking
attn_scores[0, 0, :, :] = [
    [2.31,  0.26, -0.66],  # Position 0
    [1.45,  2.13,  0.89],  # Position 1
    [0.33, -0.55,  1.78]   # Position 2
]

# After masking (set future positions to -inf)
attn_scores[0, 0, :, :] = [
    [2.31,  -inf,  -inf],  # Position 0: can't see 1 or 2
    [1.45,  2.13,  -inf],  # Position 1: can't see 2
    [0.33, -0.55,  1.78]   # Position 2: can see all
]
```

---

#### **Step 4: Softmax**

**Purpose:** Convert scores to probabilities (sum to 1)

**Formula:**
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**Example (position 0):**
```python
scores = [2.31, -inf, -inf]

# exp() of each:
exp_scores = [exp(2.31), exp(-inf), exp(-inf)]
           = [10.07, 0.0, 0.0]

# Normalize:
probs = [10.07/10.07, 0.0/10.07, 0.0/10.07]
      = [1.0, 0.0, 0.0]  # 100% attention to itself!
```

**Example (position 1):**
```python
scores = [1.45, 2.13, -inf]
exp_scores = [4.26, 8.41, 0.0]
total = 12.67

probs = [4.26/12.67, 8.41/12.67, 0.0]
      = [0.34, 0.66, 0.0]  # 34% to pos 0, 66% to itself
```

**Shape:**
```python
attn_weights.shape = [1, 12, 3, 3]
# For each head, each position, we have a probability distribution
```

---

#### **Step 5: Apply to Values**

**Purpose:** Weighted combination of value vectors

**Code:**
```python
context = attn_weights @ values
# [1, 12, 3, 3] @ [1, 12, 3, 64] = [1, 12, 3, 64]
```

**What it does:**

For each position, create a weighted sum of all value vectors:

**Example (position 1, head 0):**
```python
# Attention weights:
weights = [0.34, 0.66, 0.0]

# Value vectors:
V_0 = [0.5, 0.3, ..., 0.2]  (64 dims)
V_1 = [0.1, 0.7, ..., 0.4]  (64 dims)
V_2 = [0.2, 0.5, ..., 0.1]  (64 dims)

# Weighted sum:
context_1 = 0.34 * V_0 + 0.66 * V_1 + 0.0 * V_2
          = [0.237, 0.564, ..., 0.332]  (64 dims)
```

**Intuition:**
- Position 1 pays 66% attention to itself and 34% to position 0
- Its output is a blend weighted by these percentages

---

### 8.4 Flash Attention vs Standard

**Standard Attention (above):**
```python
# Explicit steps
scores = Q @ K.T
scores = scores / sqrt(d_k)
scores.masked_fill_(mask, -inf)
weights = softmax(scores)
output = weights @ V
```

**Flash Attention (optimized):**
```python
output = torch.nn.functional.scaled_dot_product_attention(
    queries, keys, values,
    attn_mask=None,
    dropout_p=0.1,
    is_causal=True  # Handles masking automatically
)
```

**Differences:**

| Aspect | Standard | Flash |
|--------|----------|-------|
| Speed | Baseline | 2-4x faster |
| Memory | Stores attn matrix | Doesn't store matrix |
| Implementation | Manual steps | Single function call |
| Masking | Manual `masked_fill_` | Automatic with `is_causal=True` |
| Device | CPU/GPU | GPU optimized (CUDA kernels) |

**Output:**
```python
# Both produce identical results:
output.shape = [1, 12, 3, 64]
```

---

## 9. Feed-Forward Network

### 9.1 Expansion Layer

**Purpose:** Non-linear transformation with dimension expansion

**Code:**
```python
self.fc1 = nn.Linear(emb_dim, 4 * emb_dim)
# 768 → 3072

hidden = self.fc1(x)
```

**Input:**
```python
x.shape = [1, 3, 768]
```

**Matrix Multiplication:**
```python
weight.shape = [3072, 768]
bias.shape = [3072]

hidden = x @ weight.T + bias
       = [1, 3, 768] @ [768, 3072] + [3072]
       = [1, 3, 3072]
```

**What happens:**

Each 768-dim token vector becomes 3072-dim:

```
Input (position 0):
  [0.06, -1.12, ..., 0.45]  (768 values)
      ↓ linear transformation
Hidden (position 0):
  [0.23, 0.45, ..., -0.12]  (3072 values!)
```

**Why expand?**

More capacity for complex transformations. Think of it as:
- Input: Compressed representation
- Hidden: Expanded "working space"
- Output: Compressed back

---

### 9.2 GELU Activation

**Purpose:** Non-linear activation (allows learning complex patterns)

**Formula:**
```
GELU(x) = x * Φ(x)
where Φ(x) is the CDF of standard normal distribution

Approximation:
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Code:**
```python
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))

activated = gelu(hidden)
```

**What it does:**

Element-wise transformation:

```python
# Input
hidden = tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, ...]])

# Output
activated = tensor([[-0.046, -0.159, 0.0, 0.841, 1.954, ...]])
```

**Visual:**

```
GELU is smooth and slightly different from ReLU:

     ^
   2 |        _____
     |      /
   1 |    /
     |   /
   0 |——————————
     | /
  -1 |/___________
     |
     +————————————> x
    -2  -1  0  1  2

Key properties:
- Smooth (differentiable everywhere)
- Non-zero for negative values (unlike ReLU)
- Approximately linear for large positive x
```

**Shape:**
```python
hidden.shape = [1, 3, 3072]
activated.shape = [1, 3, 3072]  # Same! Element-wise operation
```

---

### 9.3 Projection Layer

**Purpose:** Project back to original dimension

**Code:**
```python
self.fc2 = nn.Linear(4 * emb_dim, emb_dim)
# 3072 → 768

output = self.fc2(activated)
```

**Transformation:**
```python
activated.shape = [1, 3, 3072]
output.shape = [1, 3, 768]

# Each 3072-dim vector → 768-dim vector
```

**Complete FFN Flow:**

```
Input: [1, 3, 768]
   ↓ fc1 (expand)
Hidden: [1, 3, 3072]
   ↓ GELU (activate)
Activated: [1, 3, 3072]
   ↓ fc2 (project)
Output: [1, 3, 768]
```

**Why this architecture?**

1. **Expansion provides capacity:** More neurons = can learn more complex functions
2. **Non-linearity enables learning:** Without GELU, two linear layers = one linear layer
3. **Projection maintains dimensions:** Output matches input for residual connection

**PyTorch Implementation:**
```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.gelu = GELU()
        self.fc2 = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
    
    def forward(self, x):
        # x: [batch, seq, 768]
        x = self.fc1(x)    # [batch, seq, 3072]
        x = self.gelu(x)   # [batch, seq, 3072]
        x = self.fc2(x)    # [batch, seq, 768]
        return x
```

---

## 10. Transformer Block

### 10.1 Pre-Norm Architecture

**Standard Transformer (Post-Norm):**
```python
x = x + Attention(x)
x = LayerNorm(x)
x = x + FFN(x)
x = LayerNorm(x)
```

**GPT-2 uses Pre-Norm:**
```python
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Why Pre-Norm?**

1. **Training stability:** Normalizes before unstable operations
2. **Gradient flow:** Better backpropagation through deep networks
3. **Faster convergence:** Empirically works better for LLMs

**Code:**
```python
def forward(self, x):
    # Attention block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.dropout(x)
    x = x + shortcut  # Residual
    
    # FFN block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.dropout(x)
    x = x + shortcut  # Residual
    
    return x
```

---

### 10.2 Residual Connections

**Purpose:** Allow gradients to flow directly through the network

**Problem without residuals:**
```
Input → Layer1 → Layer2 → ... → Layer12 → Output

During backprop, gradients must flow through all 12 layers
→ Vanishing gradients! (gradients → 0)
```

**With residuals:**
```
Input ──┐
        ├→ Layer1 ──┐
        └───────────┘  ← Shortcut!
                │
                ├→ Layer2 ──┐
                └───────────┘
                      │
                      ...
```

**Math:**
```python
# Without residual
output = f(x)

# With residual
output = x + f(x)

# Gradient flows through both paths:
d_output/d_x = 1 + d_f/d_x
             ↑ Always has this term!
```

**Example:**

```python
# Input
x = tensor([[[ 0.06, -1.12,  1.11, ...]]]) # [1, 3, 768]

# After attention
attn_out = tensor([[[-0.05,  0.23, -0.15, ...]]])

# Residual connection
output = x + attn_out
       = [[[ 0.01, -0.89,  0.96, ...]]]
# Notice: similar to input (residual helps!)
```

---

### 10.3 Complete Forward Pass

**Full code with shapes:**

```python
def forward(self, x):
    # Input: [batch, seq, 768]
    
    # === ATTENTION BLOCK ===
    shortcut = x                    # [batch, seq, 768]
    x = self.norm1(x)              # [batch, seq, 768] - normalize
    x = self.att(x)                # [batch, seq, 768] - attend
    x = self.dropout(x)            # [batch, seq, 768] - regularize
    x = x + shortcut               # [batch, seq, 768] - residual
    
    # === FFN BLOCK ===
    shortcut = x                    # [batch, seq, 768]
    x = self.norm2(x)              # [batch, seq, 768] - normalize
    x = self.ff(x)                 # [batch, seq, 768] - transform
    x = self.dropout(x)            # [batch, seq, 768] - regularize
    x = x + shortcut               # [batch, seq, 768] - residual
    
    return x                        # [batch, seq, 768]
```

**Example with numbers:**

```python
# Input (after embeddings)
x_init = tensor([[[0.06, -1.12, 1.11, ...]]])  # [1, 3, 768]

# --- Attention Block ---
# Norm
x_norm1 = norm(x_init) = [[[ 0.05, -0.98, 0.93, ...]]]

# Attention (12 heads processing in parallel)
x_att = attention(x_norm1) = [[[-0.05, 0.23, -0.15, ...]]]

# Dropout (randomly zero some values)
x_drop1 = dropout(x_att) = [[[-0.05, 0.0, -0.15, ...]]]  # 2nd value dropped!

# Residual
x_res1 = x_init + x_drop1 = [[[0.01, -1.12, 0.96, ...]]]

# --- FFN Block ---
# Norm
x_norm2 = norm(x_res1) = [[[0.01, -0.99, 0.85, ...]]]

# FFN (expand → activate → project)
x_ffn = ffn(x_norm2) = [[[0.12, -0.34, 0.56, ...]]]

# Dropout
x_drop2 = dropout(x_ffn) = [[[0.12, -0.34, 0.0, ...]]]  # 3rd value dropped!

# Residual
x_final = x_res1 + x_drop2 = [[[0.13, -1.46, 0.96, ...]]]

# Return
return x_final  # This goes to the next transformer block!
```

---

# Will continue in next response due to length...

---

This is Part 1 of the comprehensive guide. Should I continue with:
- Part IV: Training Process (detailed training loop)
- Part V: Validation and Generation
- Part VI: Dataset Collection
- Part VII: Advanced Topics

The document is structured to be ~20,000+ lines when complete. Would you like me to continue building it section by section?
