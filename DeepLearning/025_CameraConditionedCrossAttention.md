# The Challenge: Implement Camera-Conditioned Cross-Attention
Imagine you are building a video model where you want to control the camera trajectory. You have your video features (the "latent" frames) and a set of camera embeddings (vectors representing pitch, yaw, roll, and translation).

**The Task:** Implement a `CameraCrossAttention` module. This module should allow the video latent features to "look at" the camera instructions to adjust the pixel flow.

**Requirements:**
1. Input:
    - *Latents*: A tensor of shape $(B, T, C, H, W)$.
    - *Camera Embeds*: A tensor of shape $(B, T, D_{cam})$ (one camera vector per frame).
2. Spatial Invariance: The camera instruction for frame $t$ should be applied to all pixels $(H, W)$ in that same frame $t$
3. Mechanism: 
    - Queries ($Q$): Derived from the video latents.
    - Keys ($K$) & Values ($V$): Derived from the camera embeddings.
4. Efficiency: Use `F.scaled_dot_product_attention`.

## Implementation Template
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraCrossAttention(nn.Module):
    def __init__(self, latent_dim, cam_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        
        self.to_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.to_k = nn.Linear(cam_dim, latent_dim, bias=False)
        self.to_v = nn.Linear(cam_dim, latent_dim, bias=False)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, cam_features):
        """
        x: (B, T, C, H, W) - Video Latents
        cam_features: (B, T, D_cam) - Camera Parameters
        """
        B, T, C, H, W = x.shape
        
        # 1. YOUR TASK: Reshape x so that each pixel is a "token"
        # x_flat should be (B, T * H * W, C) or (B * T, H * W, C)
        # Note: If camera control is frame-specific, think about the best grouping.
        
        # 2. YOUR TASK: Prepare cam_features as K and V.
        # Since cam_features are (B, T, D_cam), you need to decide if 
        # pixels in frame t should only attend to camera t, or all cameras.
        
        # 3. Apply Multi-Head Cross-Attention
        
        pass
```

## Attempt \#1

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraCrossAttention(nn.Module):
    def __init__(self, latent_dim, cam_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        
        self.to_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.to_k = nn.Linear(cam_dim, latent_dim, bias=False)
        self.to_v = nn.Linear(cam_dim, latent_dim, bias=False)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, cam_features):
        """
        x: (B, T, C, H, W) - Video Latents
        cam_features: (B, T, D_cam) - Camera Parameters
        """
        B, T, C, H, W = x.shape
        
        # 1. YOUR TASK: Reshape x so that each pixel is a "token"
        # x_flat should be (B, T * H * W, C) or (B * T, H * W, C)
        # Note: If camera control is frame-specific, think about the best grouping.
        x_flat = x.reshape(B*T, C, H*W).transpose(-1, -2).contiguous()
        q = self.to_q(x_flat) # B*T, H*W, C
        
        # 2. YOUR TASK: Prepare cam_features as K and V.
        # Since cam_features are (B, T, D_cam), you need to decide if 
        # pixels in frame t should only attend to camera t, or all cameras.
        cam_features = cam_features.reshape(B*T, 1, -1)
        k, v = self.to_k(cam_features), self.to_v(cam_features)
        
        # 3. Apply Multi-Head Cross-Attention
        def multihead_reshape(x):
            # x.shape = B, L, D
            B, L, D = x.shape
            return x.reshape(B, L, self.num_heads, D//self.num_heads).transpose(1, 2).contiguous()
        
        q, k, v = map(multihead_reshape, (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v) # B*T, num_heads, H*W, head_dim

        out = out.transpose(1, 2).contiguous().reshape(B, T, H, W, C)
        out = self.proj(out)

        return out.permute(0, 1, 4, 2, 3)
```