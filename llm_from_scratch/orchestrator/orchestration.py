import torch
import tiktoken

# Fix imports: Add parent directory to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_from_scratch.Dataset.loader import create_dataloader_v1
from llm_from_scratch.GPT2Model.gpt2 import GPTModel
from llm_from_scratch.Trainer.trainer import train_model_simple

# 1. SETUP & CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    # Model architecture
    "vocab_size": 50257,    # GPT-2 Vocabulary size
    "context_length": 256,  # Maximum sequence length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of Transformer blocks
    "drop_rate": 0.1,       # Dropout percentage
    "qkv_bias": False,      # Query-Key-Value bias
    
    # Attention settings
    "use_flash": True,      # Use Flash Attention (faster, requires PyTorch 2.0+, GPU recommended)
                            # Set to False for CPU or older PyTorch versions
    
    # Dataloader settings
    "batch_size": 2,        # Number of samples per batch
    "stride": 64,           # Sliding window stride for creating samples
    "drop_last": False,     # Whether to drop incomplete batches
    
    # Training settings
    "train_ratio": 0.90,    # Train/validation split ratio
}

# 2. DATA PREPARATION & SPLIT
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split data into train and validation sets
split_idx = int(cfg["train_ratio"] * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

# Initialize Loaders using config parameters
train_loader = create_dataloader_v1(
    train_data, 
    batch_size=cfg["batch_size"], 
    max_length=cfg["context_length"], 
    stride=cfg["stride"], 
    drop_last=cfg["drop_last"]
)
val_loader = create_dataloader_v1(
    val_data, 
    batch_size=cfg["batch_size"], 
    max_length=cfg["context_length"], 
    stride=cfg["stride"], 
    drop_last=cfg["drop_last"]
)

# 3. INITIALIZE MODEL & OPTIMIZER
model = GPTModel(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# 4. GENERATION UTILITY
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Autoregressive generation: Predicts one token at a time, 
    appends it to input, and repeats.
    """
    for _ in range(max_new_tokens):
        # 1. Truncate context to max allowed by model
        idx_cond = idx[:, -context_size:]
        
        # 2. Get predictions (Shape: [Batch, Seq, Vocab])
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 3. Slicing: Only care about the LAST token prediction
        logits = logits[:, -1, :] # New Shape: [Batch, Vocab]
        
        # 4. Convert scores to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # 5. Greedy Search: Pick the #1 most likely word
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) # Shape: [Batch, 1]
        
        # 6. Append to sequence
        idx = torch.cat((idx, idx_next), dim=1) 
        
    return idx

# 5. THE "PLAY" (FORWARD PASS TEST)
print("--- Starting Forward Pass Test ---")
data_iter = iter(train_loader)
inputs, targets = next(data_iter)
inputs, targets = inputs.to(device), targets.to(device)

# Initial Forward Pass
logits = model(inputs) 
print(f"Input Shape:  {inputs.shape}")  # [2, 256]
print(f"Logits Shape: {logits.shape}") # [2, 256, 50257]

# 6. INITIAL GENERATION (GIBBERISH CHECK)
print("\n--- Initial Generation (Untrained) ---")
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Every effort moves"
encoded = tokenizer.encode(start_context)
idx = torch.tensor(encoded).unsqueeze(0).to(device) # Add batch dimension

model.eval() # Switch to eval mode (disable dropout)
out = generate_text_simple(model, idx, max_new_tokens=10, context_size=cfg["context_length"])
print(f"Input text: {start_context}")
print(f"Output:     {tokenizer.decode(out.squeeze(0).tolist())}")
model.train() # Switch back to training mode

# # 7. START TRAINING
# print("\n--- Starting Training Loop ---")
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     optimizer=optimizer,
#     device=device,
#     num_epochs=10,
#     eval_freq=5,
#     eval_iter=1,
#     start_context=start_context,
#     tokenizer=tokenizer
# )