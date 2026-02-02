# The Coding Challenge: Spatio-Temporal Window Attention

**The Task:** Implement a Spatio-Temporal Window Attention module in PyTorch.

**Requirements:** 
1. *Input:* A 5D tensor of shape $(B, C, T, H, W)$ (Batch, Channels, Time, Height, Width).
2. *Partitioning:* Divide the video into non-overlapping "3D tubes" (windows) of size $(wT, wH, wW)$.
3. *Attention:* Perform standard scaled dot-product self-attention **only within each tube**.
4. *Output:* Reshape the result back to the original input shape $(B, C, T, H, W)$.

**Constraints:**
- You must use einops for tensor manipulation if you are familiar with it, or standard view/permute operations.
- Assume the input dimensions are perfectly divisible by the window sizes.
- The function should be efficient (avoiding explicit loops where possible).

## Starter Code Template
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            window_size (tuple): (wT, wH, wW) dimensions of the 3D window.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        # Your logic starts here
        pass
```

## Attempt \#1
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            window_size (tuple): (wT, wH, wW) dimensions of the 3D window.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        # Your logic starts here
        wT, wH, wW = self.window_size
        T_wT, H_wH, W_wW = T//wT, H//wH, W//wW
        num_cubes = T_wT * H_wH * W_wW
        L = wT*wH*wW
        dim_head = self.dim // self.num_heads

        x = x.reshape(B, self.dim, T_wT, wT, H_wH, wH, W_wW, wW).permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(B*num_cubes, L, self.dim)

        qkv = self.qkv(x) # B*num_cubes, L, 3C
        qkv = qkv.reshape(B*num_cubes, L, 3, self.num_heads, self.dim//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # B*num_cubes, num_heads, L, dim_head

        attn = (q @ k.transpose(-1, -2)) * self.scale # B*num_cubes, num_heads, L, L
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B*num_cubes, L, self.dim) # B*num_cubes, L, dim

        out = self.proj(out) # B*num_cubes, L, dim
        out = out.reshape(B, T_wT, H_wH, W_wW, wT, wH, wW, self.dim)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().reshape(B, self.dim, T, H, W)
        return out
```

**Gemini Feedback:** Nailed it. This version is interview-ready and technically sound.