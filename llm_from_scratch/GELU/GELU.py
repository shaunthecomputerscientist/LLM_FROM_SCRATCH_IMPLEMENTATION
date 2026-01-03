import torch
import torch.nn as nn

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    GELU is a smooth, non-linear activation that works better than ReLU for transformers.
    It's used in GPT-2, GPT-3, and BERT.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Apply GELU activation element-wise.
        
        Args:
            x: Input tensor of any shape
               Typical shape: [batch_size, sequence_length, emb_dim]
               Example: [8, 1024, 768] or [8, 1024, 3072] (in FFN)
        
        Returns:
            Output tensor with SAME shape as input
            Shape: [batch_size, sequence_length, emb_dim]
            
        Note: GELU does NOT change the shape - it's an element-wise operation!
              Just like ReLU, it applies the function to each number independently.
        
        Shape transformation:
            Input:  [8, 1024, 768]
            Output: [8, 1024, 768]  ← Same shape!
        """
        # This is the 'GELU' approximation formula used in GPT-2/GPT-3
        # Applies element-wise: every number is transformed independently
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))