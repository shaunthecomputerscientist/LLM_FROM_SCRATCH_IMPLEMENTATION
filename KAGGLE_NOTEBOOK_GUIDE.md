# 🚀 Kaggle-Ready GPT-2 Training Notebook

This notebook can run standalone on Kaggle! Just upload this `.ipynb` file and run all cells.

## Setup Instructions for Kaggle

### Option 1: Upload this notebook directly to Kaggle
1. Go to Kaggle.com
2. Create new notebook
3. Upload this `llm.ipynb`
4. Run all cells

### Option 2: Clone from your GitHub
1. Fork/clone repository to your GitHub
2. In Kaggle notebook, run the setup cell below

---

## Cell 1: Setup - Clone Repository from GitHub

```python
# Run this cell FIRST on Kaggle to download all code from GitHub
import os
import sys

# Check if we're on Kaggle
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("🔍 Running on Kaggle - cloning repository...")
    
    # Clone the repository
    !git clone https://github.com/shaunthecomputerscientist/LLM_FROM_SCRATCH_IMPLEMENTATION.git
    
    # Change to repo directory
    os.chdir('LLM_FROM_SCRATCH_IMPLEMENTATION')
    
    # Add to Python path
    sys.path.insert(0, '/kaggle/working/LLM_FROM_SCRATCH_IMPLEMENTATION')
    
    # Install requirements
    !pip install -q -r requirements.txt
    
    print("✅ Repository cloned and ready!")
    print(f"📂 Current directory: {os.getcwd()}")
    
else:
    print("💻 Running locally - using local files")
    # Add local path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd()))
```

**Alternative: Download specific files only (lightweight)**

```python
# If you only need core files, use this instead
import urllib.request
import os

base_url = "https://raw.githubusercontent.com/shaunthecomputerscientist/LLM_FROM_SCRATCH_IMPLEMENTATION/main/"

files_to_download = [
    "llm_from_scratch/GPT2Model/gpt2.py",
    "llm_from_scratch/CMHA/cmha.py",
    "llm_from_scratch/TransformerBlock/transformer_block.py",
    "llm_from_scratch/FFN/ffn.py",
    "llm_from_scratch/GELU/GELU.py",
    "llm_from_scratch/LayerNorm/layernorm.py",
    "llm_from_scratch/Dataset/loader.py",
    "llm_from_scratch/Trainer/trainer.py",
]

for file_path in files_to_download:
    url = base_url + file_path
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    urllib.request.urlretrieve(url, file_path)
    print(f"✓ Downloaded: {file_path}")
```

---

## Cell 2: Imports

```python
import torch
import tiktoken
import numpy as np
from llm_from_scratch.GPT2Model.gpt2 import GPTModel
from llm_from_scratch.Dataset.loader import create_dataloader_v1
from llm_from_scratch.Trainer.trainer import (
    train_model_simple, 
    generate_text_simple,
    calc_loss_batch
)

print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

---

## Cell 3: Configuration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    # Model architecture
    "vocab_size": 50257,
    "context_length": 256,      # Reduced for Kaggle
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 6,              # Reduced for faster training
    "drop_rate": 0.1,
    "qkv_bias": False,
    "use_flash": torch.cuda.is_available(),  # Auto-detect GPU
    
    # Training
    "batch_size": 4,
    "stride": 64,
    "train_ratio": 0.90,
}

print("Configuration:")
for k, v in cfg.items():
    print(f"  {k}: {v}")
```

---

## Cell 4: Load/Create Dataset

```python
from datasets import load_dataset

# Option 1: Use existing text file (if available)
try:
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"✓ Loaded local dataset: {len(raw_text):,} characters")
    
except FileNotFoundError:
    # Option 2: Download from HuggingFace
    print("📥 Downloading dataset from HuggingFace...")
    
    # Small sample for quick testing
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train",
        streaming=True
    )
    
    # Take first 100 documents
    raw_text = ""
    for i, doc in enumerate(dataset):
        if i >= 100:
            break
        raw_text += doc['text'] + "\n\n"
    
    # Save for future use
    with open("the-verdict.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    print(f"✓ Downloaded dataset: {len(raw_text):,} characters")

# Create train/val split
split_idx = int(cfg["train_ratio"] * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

train_loader = create_dataloader_v1(
    train_data, 
    batch_size=cfg["batch_size"],
    max_length=cfg["context_length"],
    stride=cfg["stride"]
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=cfg["batch_size"], 
    max_length=cfg["context_length"],
    stride=cfg["stride"]
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

---

## Cell 5: Initialize Model

```python
model = GPTModel(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
```

---

## Cell 6: Test Forward Pass

```python
# Test with one batch
data_iter = iter(train_loader)
inputs, targets = next(data_iter)
inputs, targets = inputs.to(device), targets.to(device)

print(f"Input shape: {inputs.shape}")
print(f"Target shape: {targets.shape}")

# Forward pass
with torch.no_grad():
    logits = model(inputs)
    
print(f"Logits shape: {logits.shape}")
print(f"Expected: [{cfg['batch_size']}, {cfg['context_length']}, {cfg['vocab_size']}]")

# Calculate loss
loss = calc_loss_batch(inputs, targets, model, device)
print(f"Initial loss: {loss.item():.4f}")
```

---

## Cell 7: Train Model

```python
# Training configuration
num_epochs = 5
eval_freq = 10
eval_iter = 5

print("🚂 Starting training...")
print(f"Epochs: {num_epochs}")
print(f"Device: {device}")
print(f"Mixed Precision: {cfg.get('use_flash', False)}")

# Train
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=eval_freq,
    eval_iter=eval_iter,
    start_context="Every effort moves",
    tokenizer=tiktoken.get_encoding("gpt2"),
    memory_efficient=torch.cuda.is_available(),
    accumulation_steps=4
)

print("\n✅ Training complete!")
```

---

## Cell 8: Generate Text

```python
tokenizer = tiktoken.get_encoding("gpt2")

prompts = [
    "Every effort moves",
    "I had always thought",
    "The artist pondered"
]

model.eval()
print("🎨 Generating text...\n")

for prompt in prompts:
    encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    output = generate_text_simple(
        model=model,
        idx=encoded,
        max_new_tokens=50,
        context_size=cfg["context_length"],
        temperature=0.7,
        top_k=10
    )
    
    generated = tokenizer.decode(output.squeeze(0).tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
    print("-" * 80 + "\n")
```

---

## Cell 9: Save Model

```python
# Save checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': cfg,
    'train_losses': train_losses,
    'val_losses': val_losses
}

torch.save(checkpoint, 'gpt2_checkpoint.pt')
print("✅ Model saved to: gpt2_checkpoint.pt")

# Save just the model weights
torch.save(model.state_dict(), 'gpt2_weights.pt')
print("✅ Weights saved to: gpt2_weights.pt")
```

---

## Cell 10: Plot Training Curves

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Evaluation Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot 2: Tokens Seen
plt.subplot(1, 2, 2)
plt.plot(tokens_seen, train_losses, label='Train Loss')
plt.xlabel('Tokens Seen')
plt.ylabel('Loss')
plt.title('Loss vs Tokens Processed')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Training curves saved to: training_curves.png")
```

---

## 🎯 Quick Start Summary

**On Kaggle:**
1. Upload this notebook
2. Enable GPU (Settings → Accelerator → GPU)
3. Run all cells
4. Download trained model from output folder

**On Colab:**
1. Same as Kaggle
2. Runtime → Change runtime type → GPU

**Locally:**
1. Clone repository
2. Run: `jupyter notebook llm.ipynb`

---

## 📚 Resources

- **GitHub:** https://github.com/shaunthecomputerscientist/LLM_FROM_SCRATCH_IMPLEMENTATION
- **Documentation:** See `docs/COMPLETE_PROJECT_GUIDE.md`
- **README:** See repository README.md
