"""
Detailed explanation of Line 73: tok_embeds = tok_emb(inputs)
This demonstrates how embedding layers work with tensors of token IDs
"""

import torch
import torch.nn as nn

# ============================================================
# STEP 1: What does `inputs` look like?
# ============================================================

# After this line from orchestration.py:
# inputs, targets = next(data_iter)

# `inputs` is a PyTorch tensor with shape [batch_size, sequence_length]
# Let's simulate with simpler numbers for clarity:

batch_size = 2      # Instead of 8 (smaller for demonstration)
seq_length = 5      # Instead of 1024 (smaller for demonstration)

# Example token IDs (these are the actual integers from the tokenizer)
inputs = torch.tensor([
    [40, 367, 2885, 1464, 1807],   # First sequence (5 token IDs)
    [3619, 402, 271, 10899, 2138]  # Second sequence (5 token IDs)
])

print("=" * 60)
print("STEP 1: What is `inputs`?")
print("=" * 60)
print(f"Type: {type(inputs)}")
print(f"Shape: {inputs.shape}")  # [2, 5] = [batch_size, seq_length]
print(f"Data type: {inputs.dtype}")  # torch.int64 (integers!)
print(f"\nActual values:\n{inputs}")
print("\nThink of it as:")
print("  - 2 sentences (batch)")
print("  - Each sentence has 5 words (tokens)")
print("  - Each word is represented by an ID number")

# ============================================================
# STEP 2: What is `tok_emb`?
# ============================================================

vocab_size = 50257  # GPT-2 vocabulary size
emb_dim = 768       # Embedding dimension (size of vector per token)

# Create the embedding layer (from line 69 in orchestration.py)
tok_emb = nn.Embedding(vocab_size, emb_dim)

print("\n" + "=" * 60)
print("STEP 2: What is `tok_emb`?")
print("=" * 60)
print(f"Type: {type(tok_emb)}")
print(f"Embedding layer parameters:")
print(f"  - vocab_size: {vocab_size} (number of unique tokens)")
print(f"  - emb_dim: {emb_dim} (size of each embedding vector)")
print(f"\nThis is essentially a LOOKUP TABLE with:")
print(f"  - {vocab_size} rows (one for each possible token)")
print(f"  - {emb_dim} columns (the embedding vector)")
print(f"\nTotal learnable parameters: {vocab_size * emb_dim:,}")

# ============================================================
# STEP 3: How does tok_emb(inputs) work?
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: How does tok_emb(inputs) work?")
print("=" * 60)

# When you call tok_emb(inputs), PyTorch does the following:
# For each token ID in inputs, it looks up the corresponding row 
# in the embedding table and returns that vector

tok_embeds = tok_emb(inputs)

print(f"\nInput shape:  {inputs.shape}")      # [2, 5]
print(f"Output shape: {tok_embeds.shape}")   # [2, 5, 768]

print("\nWhat happened?")
print("  Each token ID was replaced by its 768-dimensional vector!")
print("\n  Before: inputs[0, 0] = 40 (just a number)")
print(f"  After:  tok_embeds[0, 0] = [... 768 numbers ...]")
print(f"\n  First embedding vector for token ID 40:")
print(f"  {tok_embeds[0, 0, :10]}...")  # Show first 10 dimensions
print(f"  (... and 758 more numbers)")

# ============================================================
# STEP 4: Visual representation
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: VISUAL REPRESENTATION")
print("=" * 60)

print("""
BEFORE (inputs):
┌─────────────────────────────┐
│ Batch 0: [40, 367, 2885, ...│  ← Just integers (token IDs)
│ Batch 1: [3619, 402, 271,.. │
└─────────────────────────────┘
Shape: [2, 5]

                ↓
         tok_emb(inputs)
                ↓

AFTER (tok_embeds):
┌──────────────────────────────────────┐
│ Batch 0:                             │
│   Token 40   → [0.12, -0.45, ...(768)]│  ← Real-valued vector
│   Token 367  → [0.89, 0.32, ...(768)] │
│   Token 2885 → [-0.21, 1.03, ...(768)]│
│   ...                                 │
│ Batch 1:                             │
│   Token 3619 → [0.44, -0.88, ...(768)]│
│   Token 402  → [1.12, 0.05, ...(768)] │
│   ...                                 │
└──────────────────────────────────────┘
Shape: [2, 5, 768]
""")

# ============================================================
# STEP 5: Answer to your specific questions
# ============================================================

print("=" * 60)
print("ANSWERS TO YOUR QUESTIONS")
print("=" * 60)

print("""
Q1: "Does neural network take list of numbers?"
A1: No, it takes PyTorch TENSORS (multi-dimensional arrays).
    Lists must be converted to tensors first.

Q2: "Or inputs is a tensor?"
A2: YES! `inputs` is a tensor of shape [batch_size, seq_length]
    containing integer token IDs.

Q3: "Tensor containing what? Just list of token IDs?"
A3: YES! It contains ONLY token IDs (integers from 0 to 50256).
    BEFORE embedding: [40, 367, 2885] (integers)
    AFTER embedding:  [[0.12, -0.45, ...], [0.89, 0.32, ...], ...]
                      (vectors of real numbers)

Q4: "How does line 73 work?"
A4: tok_emb(inputs) performs a LOOKUP operation:
    1. Take each token ID from inputs
    2. Look it up in the embedding table
    3. Return the corresponding 768-dimensional vector
    4. Stack all vectors together
""")

# ============================================================
# BONUS: Show the actual transformation step by step
# ============================================================

print("\n" + "=" * 60)
print("BONUS: Step-by-step transformation for one token")
print("=" * 60)

token_id = 40
print(f"Token ID: {token_id}")
print(f"Embedding vector for token {token_id}:")
print(f"  Shape: {tok_emb.weight[token_id].shape}")  # [768]
print(f"  First 10 values: {tok_emb.weight[token_id][:10]}")
print(f"\nThis is the SAME vector that appears at tok_embeds[0, 0]")
print(f"Verification: {torch.allclose(tok_emb.weight[token_id], tok_embeds[0, 0])}")
