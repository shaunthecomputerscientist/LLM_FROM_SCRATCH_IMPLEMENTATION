import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization - normalizes across the feature dimension (emb_dim).
    
    Unlike Batch Normalization (normalizes across batch), Layer Norm normalizes
    each token's embedding independently. This makes it better for variable-length
    sequences and transformers.
    """
    def __init__(self, emb_dim):
        """
        Args:
            emb_dim: Embedding dimension (e.g., 768 for GPT-2)
        """
        super().__init__()
        self.eps = 1e-5  # Prevents division by zero
        
        # Learnable parameters (initialized to identity transformation)
        # Scale (gamma) and Shift (beta) let the model learn optimal normalization
        self.scale = nn.Parameter(torch.ones(emb_dim))   # Shape: [768]
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Shape: [768]

    def forward(self, x):
        """
        Apply Layer Normalization.
        
        Args:
            x: Input tensor
               Shape: [batch_size, sequence_length, emb_dim]
               Example: [8, 1024, 768]
               
        Returns:
            Normalized tensor with SAME shape as input
            Shape: [batch_size, sequence_length, emb_dim]
            Example: [8, 1024, 768]
            
        DETAILED EXPLANATION OF dim=-1 and keepdim:
        
        ┌──────────────────────────────────────────────────────────────┐
        │ Input x shape: [8, 1024, 768]                                │
        │                 ↑    ↑    ↑                                  │
        │               dim=0 dim=1 dim=2 (or dim=-1)                  │
        └──────────────────────────────────────────────────────────────┘
        
        dim=-1 means "last dimension" = dimension 2 = emb_dim (768)
        
        WHY dim=-1?
        We want to normalize EACH token's 768 features independently.
        Each token vector should have mean=0 and variance=1 across its features.
        
        Example for ONE token:
            Before: [0.5, -2.3, 1.7, ..., 0.9]  ← 768 numbers
            After:  [0.1, -0.8, 0.4, ..., 0.2]  ← normalized (mean≈0, var≈1)
        """
        
        # Step 1: Calculate mean across emb_dim (last dimension)
        mean = x.mean(dim=-1, keepdim=True)
        # Input x:        [8, 1024, 768]
        # Output mean:    [8, 1024, 1]     ← keepdim=True keeps the dimension!
        #
        # If keepdim=False, shape would be: [8, 1024] (dimension removed)
        # 
        # Each mean[i, j, 0] is the average of x[i, j, :] (all 768 features)
        # Example: mean[0, 0, 0] = average of [x[0,0,0], x[0,0,1], ..., x[0,0,767]]
        
        # Step 2: Calculate variance across emb_dim (last dimension)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Input x:        [8, 1024, 768]
        # Output var:     [8, 1024, 1]     ← keepdim=True keeps the dimension!
        #
        # Each var[i, j, 0] is the variance of x[i, j, :] (all 768 features)
        
        # Step 3: Standardize (subtract mean, divide by std)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # x:              [8, 1024, 768]
        # mean:           [8, 1024, 1]     ← broadcasts to [8, 1024, 768]
        # var:            [8, 1024, 1]     ← broadcasts to [8, 1024, 768]
        # norm_x:         [8, 1024, 768]   ← Same shape as input
        #
        # Broadcasting example:
        #   x[0, 0, :] = [0.5, -2.3, 1.7, ...]  (768 values)
        #   mean[0, 0, 0] = 0.2  → broadcasts to [0.2, 0.2, 0.2, ...] (768 times)
        #   Result: [0.5-0.2, -2.3-0.2, 1.7-0.2, ...] = [0.3, -2.5, 1.5, ...]
        
        # Step 4: Scale and shift (learned during training)
        output = self.scale * norm_x + self.shift
        # norm_x:         [8, 1024, 768]
        # self.scale:     [768]           ← broadcasts to [8, 1024, 768]
        # self.shift:     [768]           ← broadcasts to [8, 1024, 768]
        # output:         [8, 1024, 768]  ← Same shape as input
        
        return output




# ════════════════════════════════════════════════════════════════
# VISUAL EXAMPLE: What happens to a SINGLE token
# ════════════════════════════════════════════════════════════════
"""
Let's say we have token at position [0, 0] with 768 features:

BEFORE LayerNorm:
x[0, 0, :] = [0.5, -2.3, 1.7, 0.9, -0.4, ...]  (768 numbers)
  → mean = 0.2
  → variance = 1.5

AFTER Standardization (step 3):
norm_x[0, 0, :] = [(0.5-0.2)/√1.5, (-2.3-0.2)/√1.5, ...]
                = [0.24, -2.04, 1.22, ...]
  → Now has mean ≈ 0 and variance ≈ 1

AFTER Scale & Shift (step 4):
output[0, 0, :] = scale * norm_x + shift
  → Model learns optimal scale and shift during training

This happens INDEPENDENTLY for each of the 8×1024 = 8,192 tokens!
"""

# ════════════════════════════════════════════════════════════════
# WHY keepdim=True?
# ════════════════════════════════════════════════════════════════
"""
Compare keepdim=True vs keepdim=False:

Input x shape: [8, 1024, 768]

WITH keepdim=True:
  mean shape: [8, 1024, 1]
  Can broadcast: x - mean works! 
    [8, 1024, 768] - [8, 1024, 1] = [8, 1024, 768] ✓

WITHOUT keepdim=False:
  mean shape: [8, 1024]
  Cannot broadcast easily: x - mean needs reshaping!
    [8, 1024, 768] - [8, 1024] = ERROR! ✗

keepdim=True makes broadcasting automatic and clean!
"""