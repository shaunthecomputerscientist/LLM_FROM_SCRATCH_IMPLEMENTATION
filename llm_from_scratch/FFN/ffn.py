import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_from_scratch.GELU.GELU import GELU

class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) used in each Transformer block.
    
    Architecture:
        Linear (expand 4x) → GELU → Linear (project back)
        
    This is applied to EACH token independently (position-wise).
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration dictionary with:
                - emb_dim: Embedding dimension (e.g., 768 for GPT-2)
        """
        super().__init__()
        self.layers = nn.Sequential(
            # Layer 1: Expand the dimension (usually 4x larger)
            # Input:  [batch_size, sequence_length, emb_dim]
            # Output: [batch_size, sequence_length, 4 * emb_dim]
            # Example: [8, 1024, 768] → [8, 1024, 3072]
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            
            # Layer 2: Apply GELU non-linearity (element-wise, no shape change)
            # Input:  [batch_size, sequence_length, 4 * emb_dim]
            # Output: [batch_size, sequence_length, 4 * emb_dim] (same shape)
            # Example: [8, 1024, 3072] → [8, 1024, 3072]
            GELU(),
            
            # Layer 3: Project back down to the original embedding size
            # Input:  [batch_size, sequence_length, 4 * emb_dim]
            # Output: [batch_size, sequence_length, emb_dim]
            # Example: [8, 1024, 3072] → [8, 1024, 768]
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Apply feed-forward network to input.
        
        Args:
            x: Input tensor
               Shape: [batch_size, sequence_length, emb_dim]
               - batch_size: Number of sequences in batch (e.g., 8)
               - sequence_length: Number of tokens per sequence (e.g., 1024)
                                 (YES! sequence_length = context_length)
               - emb_dim: Embedding dimension (e.g., 768)
               
               Example shape: [8, 1024, 768]
        
        Returns:
            Output tensor with SAME shape as input
            Shape: [batch_size, sequence_length, emb_dim]
            Example: [8, 1024, 768]
        
        Shape transformations step-by-step:
            Input:           [8, 1024, 768]
            ↓ Linear layer 1
            After expand:    [8, 1024, 3072]  ← 4x expansion
            ↓ GELU
            After GELU:      [8, 1024, 3072]  ← No shape change
            ↓ Linear layer 2  
            Output:          [8, 1024, 768]   ← Back to original
            
        Note: 
            - sequence_length = context_length = max_length from dataloader
            - emb_dim = d_model (common notation in papers)
            - This is applied to EACH token position independently (position-wise FFN)
        """
        return self.layers(x)