import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_from_scratch.CMHA.cmha import MultiHeadAttention
from llm_from_scratch.FFN.ffn import FeedForward
from llm_from_scratch.LayerNorm.layernorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. Multi-Head (Causal) Attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            use_flash=cfg.get("use_flash", True)  # Default to True if not specified
        )
        
        # 2. Feed-Forward Network
        self.ff = FeedForward(cfg)
        
        # 3. Layer Normalization (one for each major sub-layer)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # 4. Dropout for the shortcut/residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # --- PART A: ATTENTION SUB-LAYER ---
        # 1. Save the input as a 'shortcut' (Residual Connection)
        shortcut = x
        
        # 2. Normalize and then apply Attention
        x = self.norm1(x)
        x = self.att(x)
        
        # 3. Apply Dropout and ADD the shortcut back
        x = self.drop_shortcut(x)
        x = x + shortcut 

        # --- PART B: FEED-FORWARD SUB-LAYER ---
        # 1. Save the current state as a 'shortcut'
        shortcut = x
        
        # 2. Normalize and then apply Feed-Forward
        x = self.norm2(x)
        x = self.ff(x)
        
        # 3. Apply Dropout and ADD the shortcut back
        x = self.drop_shortcut(x)
        x = x + shortcut 

        return x