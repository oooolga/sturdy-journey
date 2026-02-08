# The Coding Challenge: Factorized Attention

**The Task:** Implement a `FactorizedAttention` module that takes a video tensor $(B, T, C, H, W)$ and performs:
1. Spatial Attention: Attention within each frame (pixels attend to pixels in the same $T$).
2. Temporal Attention: Attention across time (each pixel at $(h, w)$ only attends to the same $(h, w)$ across all $T$).

**Goal:** Reduce the complexity from $O((T \cdot H \cdot W)^2)$ to $O((H \cdot W)^2 + T^2)$.

## Implementation Template

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedSpatioTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # 1. SPATIAL ATTENTION
        # YOUR TASK: Reshape x so that pixels in each frame attend to each other
        # Hint: (B*T, H*W, C)
        
        # 2. TEMPORAL ATTENTION
        # YOUR TASK: Reshape x so that each pixel attends to itself across time
        # Hint: (B*H*W, T, C)
        
        pass
```

## Attempt \#1

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedSpatioTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # 1. SPATIAL ATTENTION
        # YOUR TASK: Reshape x so that pixels in each frame attend to each other
        # Hint: (B*T, H*W, C)
        s_in = x.permute(0, 1, 3, 4, 2).contiguous().reshape(B*T, H*W, C)
        s_out, _ = self.spatial_attn(s_in, s_in, s_in) # B*T, H*W, C
        
        # 2. TEMPORAL ATTENTION
        # YOUR TASK: Reshape x so that each pixel attends to itself across time
        # Hint: (B*H*W, T, C)
        t_in = s_out.reshape(B, T, -1, C).permute(0, 2, 1, 3)
        t_out = self.temporal_attn(t_in, t_in, t_in)

        return t_out.reshape(B,H,W,T,C).permute(0,3,4,1,2)
```