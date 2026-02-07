# The Coding Challenge: Multi-Head Self-Attention (MHSA)

**The Task:**
Implement the Multi-Head Self-Attention mechanism from scratch using only basic PyTorch operations (no `nn.MultiheadAttention`).

**Requirements:**
1. *Input:* A tensor of shape $(B, N, D)$ (Batch, Sequence Length, Embedding Dimension).
2. *Linear Projections:* Create $Q$ (Query), $K$ (Key), and $V$ (Value) using linear layers.
3. *Heads:* Split the embedding dimension into $H$ heads.
4. *Scaled Dot-Product:* Calculate attention scores using the formula:$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
5. *Output:* Concatenate the heads back together and pass them through a final linear projection to return a tensor of shape $(B, N, D)$.

**Constraints:**
- Use `.transpose()` and `.reshape()` to handle the multi-head splitting efficiently (avoid loops).
- Include a Dropout layer after the softmax for regularization.

## Starter Code Template

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, D = x.shape
        # Your logic here
        pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def multihead_split(self, x):
        B, N, D = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        B, N, D = x.shape
        # Your logic here
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x) # each of shape: B, N, D
        q, k, v = self.multihead_split(q), self.multihead_split(k), self.multihead_split(v) # each of shape: B, num_head, N, head_D

        attn = (q @ k.transpose(-1, -2)) / self.scale # B, num_head, N, N
        attn = self.dropout(attn.softmax(dim=-1)) # B, num_head, N, N
    
        out = attn @ v # B, num_head, N, head_D
        out = out.transpose(1, 2) # B, N, num_head, head_D
        out = out.reshape(B, N, D)
        return self.out_proj(out)
```

# Bonus Challenge
Complete the implementation below. Pay special attention to:
- *The Scaling:* Integrating the $\sqrt{d_k}$ factor to prevent gradient vanishing.
- *The Masking (Optional but good):* How would you apply a mask (like a causal mask or padding mask) before the softmax?
- *The Matrix Multiplication:* Using torch.matmul or the @ operator across the last two dimensions.

## Starter Code Template
```Python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Args:
        query: (Batch, Heads, Seq_Len, d_k)
        key:   (Batch, Heads, Seq_Len, d_k)
        value: (Batch, Heads, Seq_Len, d_v)
        mask:  Optional mask tensor
        
    Returns:
        output: (Batch, Heads, Seq_Len, d_v)
        attention_weights: The softmax weights
    """
    # YOUR CODE HERE
    pass
```

## Attempt \#1
```Python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Args:
        query: (Batch, Heads, Seq_Len, d_k)
        key:   (Batch, Heads, Seq_Len, d_k)
        value: (Batch, Heads, Seq_Len, d_v)
        mask:  Optional mask tensor
        
    Returns:
        output: (Batch, Heads, Seq_Len, d_v)
        attention_weights: The softmax weights
    """
    B, H, S, D = query.shape

    scale = 1 / torch.sqrt(D)
    attn = query @ key.transpose(-1, -2) * scale
    if mask:
        attn = attn.masked_fill(mask==0, float('-inf'))
    attn = attn.softmax(dim=-1)

    return attn @ value, attn
```