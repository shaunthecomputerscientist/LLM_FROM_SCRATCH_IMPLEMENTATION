import torch
import torch.nn as nn

import sys
from pathlib import Path
from torch.utils.checkpoint import checkpoint
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_from_scratch.TransformerBlock.transformer_block import TransformerBlock
from llm_from_scratch.LayerNorm.layernorm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 1. Embedding Layers (Moved inside the model)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # 2. The Stack of Transformer Blocks
        # This creates 'n_layers' (e.g., 12) identical blocks in a sequence
        # 1. CHANGE: Use nn.ModuleList instead of nn.Sequential
        # ModuleList allows us to iterate and apply logic to each block individually
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.gradient_checkpointing = cfg.get("gradient_checkpointing", False)        
        # 3. Final Stabilization
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # 4. The Output Head (Language Model Head)
        # Maps the 768-dim vector back to the 50,257-dim vocab scores
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        # Step 1: Embeddings (Your 'Processing' logic from orchestration.py)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds 
        x = self.drop_emb(x)
        
        # Step 2: Pass through all 12 Transformer Blocks
        # x = self.trf_blocks(x)
        # 3. CHANGE: Manual loop through blocks with Checkpointing logic
        for block in self.trf_blocks:
            if self.gradient_checkpointing and self.training:
                # This 'forgets' the activations and recomputes them during backward()
                # 'use_reentrant=False' is the safer, modern default
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Step 3: Final Norm and Output
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits