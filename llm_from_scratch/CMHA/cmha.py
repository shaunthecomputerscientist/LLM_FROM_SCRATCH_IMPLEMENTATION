import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, use_flash=True):
        """
        Multi-Head Attention with optional Flash Attention support.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
            context_length: Maximum sequence length
            dropout: Dropout probability
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in Q, K, V projections
            use_flash: If True, uses PyTorch's Flash Attention (faster, GPU only)
                      If False, uses standard attention implementation
        """
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 
        self.use_flash = use_flash

        # Projections for Query, Key, and Value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Merges head outputs
        self.dropout = nn.Dropout(dropout)
        
        # Only create mask if NOT using flash (Flash handles masking internally)
        if not self.use_flash:
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. Linear projections
        keys = self.W_key(x) 
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2. Split into multi-head format: [Batch, Heads, Tokens, Head_Dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash:
            # ═══════════════════════════════════════════════════════════
            # FLASH ATTENTION PATH (PyTorch 2.0+, GPU recommended)
            # ═══════════════════════════════════════════════════════════
            # Uses optimized CUDA kernels for faster computation
            # Automatically handles causal masking with is_causal=True
            context_vec = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=None,  # Flash handles masking internally
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True  # Automatically applies causal mask
            )
        else:
            # ═══════════════════════════════════════════════════════════
            # STANDARD ATTENTION PATH (Compatible with all devices)
            # ═══════════════════════════════════════════════════════════
            # 3. Calculate Scores (Dot Product)
            attn_scores = queries @ keys.transpose(2, 3) 

            # 4. Apply the Causal Mask
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            
            # 5. Softmax with scaling
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 6. Apply weights to Values
            context_vec = attn_weights @ values

        # 7. Re-merge heads (same for both paths)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        
        # 8. Final Projection
        context_vec = self.out_proj(context_vec)

        return context_vec