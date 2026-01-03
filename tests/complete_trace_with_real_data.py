"""
COMPLETE STEP-BY-STEP TRACE WITH REAL DATA
Starting from: dataloader = create_dataloader_v1(...)

Shows actual data samples and shapes at each step.
"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# STEP 0: PREPARATION - Create a sample text file
# ============================================================================
print("=" * 80)
print("STEP 0: INITIAL TEXT FILE")
print("=" * 80)

# Simulating "the-verdict.txt" with actual beginning of the story
raw_text = """I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera."""

print(f"raw_text (string):")
print(f"  Length: {len(raw_text)} characters")
print(f"  Type: {type(raw_text)}")
print(f"\nFirst 100 characters:")
print(f"  '{raw_text[:100]}...'")
print(f"\nShape: Just a string (no shape yet)")

# ============================================================================
# STEP 1: INSIDE create_dataloader_v1() - Line 35
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Initialize Tokenizer (Line 35)")
print("=" * 80)

tokenizer = tiktoken.get_encoding("gpt2")

print(f"tokenizer = tiktoken.get_encoding('gpt2')")
print(f"  Type: {type(tokenizer)}")
print(f"  Vocabulary size: {tokenizer.n_vocab}")
print(f"\nThis tokenizer can convert text ↔ token IDs")

# ============================================================================
# STEP 2: INSIDE GPTDatasetV1.__init__() - Line 13
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Tokenize Entire Text (Line 13)")
print("=" * 80)

token_ids = tokenizer.encode(raw_text)

print(f"token_ids = tokenizer.encode(raw_text)")
print(f"  Type: {type(token_ids)}")
print(f"  Length: {len(token_ids)} tokens")
print(f"  Data type: list of integers")
print(f"\nFirst 20 token IDs:")
print(f"  {token_ids[:20]}")
print(f"\nLet's decode them back to see what they represent:")
for i, token_id in enumerate(token_ids[:10]):
    decoded = tokenizer.decode([token_id])
    print(f"  token_ids[{i}] = {token_id:5d} → '{decoded}'")

print(f"\nShape: 1D list of {len(token_ids)} integers")
print(f"Visual: [40, 367, 2885, 1464, ...] ← Just a flat list of numbers")

# ============================================================================
# STEP 3: SLIDING WINDOW - Lines 16-20
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Create Input-Target Pairs with Sliding Window (Lines 16-20)")
print("=" * 80)

# Smaller parameters for demonstration
max_length = 8   # Instead of 1024
stride = 8       # Instead of 1024
batch_size = 2   # Instead of 8

print(f"Parameters:")
print(f"  max_length = {max_length} (context window size)")
print(f"  stride = {stride} (how much to shift the window)")
print(f"  len(token_ids) = {len(token_ids)}")

input_ids = []
target_ids = []

print(f"\nSliding window iterations:")
for i in range(0, len(token_ids) - max_length, stride):
    input_chunk = token_ids[i:i + max_length]
    target_chunk = token_ids[i + 1: i + max_length + 1]
    
    input_ids.append(torch.tensor(input_chunk))
    target_ids.append(torch.tensor(target_chunk))
    
    print(f"\nIteration {len(input_ids)} (i={i}):")
    print(f"  input_chunk  = token_ids[{i}:{i + max_length}]")
    print(f"    → {input_chunk}")
    print(f"  target_chunk = token_ids[{i + 1}:{i + max_length + 1}]")
    print(f"    → {target_chunk}")
    print(f"  Notice: target is input shifted by 1 position!")
    
    # Show visual alignment
    print(f"\n  Visual alignment:")
    print(f"    Input:  {input_chunk}")
    print(f"    Target: {target_chunk}")
    print(f"            ↑ Each target is the next token to predict")

print(f"\nResult after sliding window:")
print(f"  input_ids: List of {len(input_ids)} tensors")
print(f"  target_ids: List of {len(target_ids)} tensors")
print(f"  Each tensor shape: {input_ids[0].shape} = [{max_length}]")

# ============================================================================
# STEP 4: DATASET CREATED
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Dataset Created (GPTDatasetV1)")
print("=" * 80)

# Simulate the dataset
class GPTDatasetV1(Dataset):
    def __init__(self, input_ids, target_ids):
        self.input_ids = input_ids
        self.target_ids = target_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

dataset = GPTDatasetV1(input_ids, target_ids)

print(f"dataset = GPTDatasetV1(...)")
print(f"  Length: {len(dataset)} samples")
print(f"  Each sample: (input_tensor, target_tensor)")
print(f"\nExample: dataset[0]")
sample_input, sample_target = dataset[0]
print(f"  Input:  {sample_input}")
print(f"  Target: {sample_target}")
print(f"  Shapes: {sample_input.shape}, {sample_target.shape}")

# ============================================================================
# STEP 5: DATALOADER CREATED - Lines 41-44
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DataLoader Created (Lines 41-44)")
print("=" * 80)

dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=False,  # False for reproducible output
    drop_last=True
)

print(f"dataloader = DataLoader(dataset, batch_size={batch_size}, ...)")
print(f"  Number of batches: {len(dataloader)}")
print(f"  Batch size: {batch_size}")
print(f"\nDataLoader groups individual samples into batches")

# ============================================================================
# STEP 6: GET FIRST BATCH - Lines 65-66
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Fetch First Batch (Lines 65-66)")
print("=" * 80)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(f"inputs, targets = next(data_iter)")
print(f"\ninputs:")
print(f"  Type: {type(inputs)}")
print(f"  Shape: {inputs.shape}  ← [batch_size, max_length]")
print(f"  dtype: {inputs.dtype}")
print(f"\n  Actual values:")
print(f"{inputs}")
print(f"\n  Visual representation:")
print(f"  ┌────────────────────────────────┐")
print(f"  │ Batch 0: {inputs[0].tolist()}")
print(f"  │ Batch 1: {inputs[1].tolist()}")
print(f"  └────────────────────────────────┘")
print(f"  Shape: [{batch_size} batches, {max_length} tokens each]")

print(f"\ntargets:")
print(f"  Type: {type(targets)}")
print(f"  Shape: {targets.shape}  ← [batch_size, max_length]")
print(f"  dtype: {targets.dtype}")
print(f"\n  Actual values:")
print(f"{targets}")

# ============================================================================
# STEP 7: TOKEN EMBEDDINGS - Line 69 & 73
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Create Token Embedding Layer (Line 69)")
print("=" * 80)

vocab_size = 50257
emb_dim = 768

tok_emb = torch.nn.Embedding(vocab_size, emb_dim)

print(f"tok_emb = torch.nn.Embedding({vocab_size}, {emb_dim})")
print(f"  Type: {type(tok_emb)}")
print(f"  Weight shape: {tok_emb.weight.shape}")
print(f"  Total parameters: {vocab_size * emb_dim:,}")

print(f"\n⚠️ IMPORTANT: Are these random numbers?")
print(f"  YES! Initially, the embedding weights are RANDOMLY initialized!")
print(f"  These random values are LEARNED during training.")
print(f"\n  Example: Embedding for token 40:")
print(f"    {tok_emb.weight[40][:10]}...")
print(f"    (showing first 10 of 768 dimensions)")
print(f"\n  This is a LOOKUP TABLE:")
print(f"    ┌────────┬──────────────────────────────────┐")
print(f"    │Token ID│ Embedding Vector (768-dim)       │")
print(f"    ├────────┼──────────────────────────────────┤")
print(f"    │   0    │ [random floats...]               │")
print(f"    │   1    │ [random floats...]               │")
print(f"    │  ...   │ ...                              │")
print(f"    │  {inputs[0,0].item()}   │ [will be looked up next] │")
print(f"    │  ...   │ ...                              │")
print(f"    │ 50256  │ [random floats...]               │")
print(f"    └────────┴──────────────────────────────────┘")

print("\n" + "=" * 80)
print("STEP 8: Apply Token Embedding (Line 73)")
print("=" * 80)

tok_embeds = tok_emb(inputs)

print(f"tok_embeds = tok_emb(inputs)")
print(f"\nBEFORE (inputs):")
print(f"  Shape: {inputs.shape}")
print(f"  Type: {inputs.dtype}")
print(f"  Sample: {inputs[0, :5]}")  # First 5 tokens of first batch
print(f"  These are INTEGERS (token IDs)")

print(f"\nAFTER (tok_embeds):")
print(f"  Shape: {tok_embeds.shape}  ← [batch_size, max_length, emb_dim]")
print(f"  Type: {tok_embeds.dtype}")
print(f"  Sample: First token of first batch (first 10 dimensions):")
print(f"    {tok_embeds[0, 0, :10]}")
print(f"  These are FLOATS (embedding vectors)")

print(f"\n  Transformation visualization:")
print(f"  ┌─────────────────────────────────────────────────────────┐")
print(f"  │ BEFORE: inputs[0, 0] = {inputs[0, 0].item():5d}                        │")
print(f"  │         (just an integer)                               │")
print(f"  │                                                         │")
print(f"  │                         ↓                               │")
print(f"  │                  Embedding Lookup                       │")
print(f"  │                         ↓                               │")
print(f"  │                                                         │")
print(f"  │ AFTER: tok_embeds[0, 0] = [{tok_embeds[0, 0, 0].item():.4f}, ...] │")
print(f"  │        (768 floating point numbers)                     │")
print(f"  └─────────────────────────────────────────────────────────┘")

# ============================================================================
# STEP 9: POSITIONAL EMBEDDINGS - Line 70 & 75-76
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Add Positional Embeddings (Lines 70, 75-76)")
print("=" * 80)

context_length = max_length
pos_emb = torch.nn.Embedding(context_length, emb_dim)

print(f"pos_emb = torch.nn.Embedding({context_length}, {emb_dim})")
print(f"  This encodes the POSITION of each token in the sequence")

batch_size_actual, seq_len = inputs.shape
pos_indices = torch.arange(seq_len)

print(f"\npos_indices = torch.arange({seq_len})")
print(f"  {pos_indices}")
print(f"  Shape: {pos_indices.shape}")
print(f"  These are position numbers: [0, 1, 2, 3, ..., {seq_len-1}]")

pos_embeds = pos_emb(pos_indices)

print(f"\npos_embeds = pos_emb(pos_indices)")
print(f"  Shape: {pos_embeds.shape}  ← [seq_len, emb_dim]")
print(f"  Sample (position 0, first 10 dims): {pos_embeds[0, :10]}")

# ============================================================================
# STEP 10: COMBINE TOKEN + POSITIONAL EMBEDDINGS - Line 79
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: Combine Token + Positional Embeddings (Line 79)")
print("=" * 80)

input_vectors = tok_embeds + pos_embeds

print(f"input_vectors = tok_embeds + pos_embeds")
print(f"\n  tok_embeds.shape:  {tok_embeds.shape}")
print(f"  pos_embeds.shape:  {pos_embeds.shape}")
print(f"  input_vectors.shape: {input_vectors.shape}")

print(f"\n  Broadcasting happens automatically!")
print(f"  pos_embeds [{seq_len}, {emb_dim}] is broadcast to [{batch_size_actual}, {seq_len}, {emb_dim}]")

print(f"\n  Visual representation:")
print(f"  ┌──────────────────────────────────────────────────────┐")
print(f"  │ Token Embedding (what the word means)               │")
print(f"  │   [{batch_size_actual}, {seq_len}, {emb_dim}]                           │")
print(f"  │                     +                                │")
print(f"  │ Position Embedding (where the word is)              │")
print(f"  │   [{seq_len}, {emb_dim}] → broadcasts to [{batch_size_actual}, {seq_len}, {emb_dim}]    │")
print(f"  │                     =                                │")
print(f"  │ Final Input (meaning + position)                    │")
print(f"  │   [{batch_size_actual}, {seq_len}, {emb_dim}]                           │")
print(f"  └──────────────────────────────────────────────────────┘")

print(f"\n  Sample: First token of first batch (first 10 dims):")
print(f"    Token emb:  {tok_embeds[0, 0, :10]}")
print(f"    Pos emb:    {pos_embeds[0, :10]}")
print(f"    Combined:   {input_vectors[0, 0, :10]}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY: COMPLETE DATA FLOW")
print("=" * 80)

print(f"""
1. raw_text (string)
   ↓ tokenizer.encode()
   
2. token_ids: list of {len(token_ids)} integers
   {token_ids[:10]}...
   ↓ sliding window (max_length={max_length}, stride={stride})
   
3. input_ids, target_ids: {len(input_ids)} pairs of tensors
   Each shape: [{max_length}]
   ↓ DataLoader(batch_size={batch_size})
   
4. inputs, targets: batched tensors
   Shape: [{batch_size}, {max_length}]
   {inputs}
   ↓ tok_emb(inputs) - LOOKUP TABLE with RANDOM INITIAL VALUES
   
5. tok_embeds: embedded tokens
   Shape: [{batch_size}, {max_length}, {emb_dim}]
   ↓ add pos_embeds
   
6. input_vectors: final input to transformer
   Shape: [{batch_size}, {max_length}, {emb_dim}]
   
✅ Ready to feed into the transformer layers!
""")

print("=" * 80)
print("ANSWER: Are embedding weights random?")
print("=" * 80)
print("""
YES! torch.nn.Embedding initializes weights RANDOMLY at creation.

During training:
  - The model learns BETTER embeddings
  - Similar words get similar vectors
  - The random values become meaningful representations

Example:
  Before training: "king" = [0.23, -0.45, 0.12, ...]  (random)
  After training:  "king" = [0.89, 0.72, -0.13, ...]  (learned)
                   "queen" = [0.85, 0.68, -0.10, ...]  (similar!)
""")
