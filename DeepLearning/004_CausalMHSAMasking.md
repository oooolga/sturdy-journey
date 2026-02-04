# The Coding Challenge: Causal MHSA Masking
Follow-up to [MultiheadSelfAttention](./003_MultiheadSelfAttention.md) question: if we are using this for World Modeling where the sequence $N$ is very long (e.g., 10,000 frames), this $N \times N$ attention matrix becomes a massive bottleneck. What is one way to modify this code to make the attention 'causal' so that frame $t$ can't look at frame $t+1$?

## Solution
You would create a lower triangular mask (using torch.tril) and fill the upper triangle of the attn scores with -inf before the softmax.

## Implementation
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

        # Causal Masking
        mask = torch.tril(torch.ones((N, N), device=x.device)).view(1, 1, N, N)
        attn = attn.masked_fill(mask==0, float('-inf'))

        attn = self.dropout(attn.softmax(dim=-1)) # B, num_head, N, N
        out = attn @ v # B, num_head, N, head_D
        out = out.transpose(1, 2) # B, N, num_head, head_D
        out = out.reshape(B, N, D)
        return self.out_proj(out)
```