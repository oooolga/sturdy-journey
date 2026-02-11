# The Problem: "Integrated 3D Positional Encoding"

Standard positional encoding (like the one used in NeRF) encodes each coordinate independently. However, in modern generative 3D models, we often use Integrated Positional Encoding (IPE) to handle "aliasing" (blurriness). For this interview question, we will focus on a multi-scale 3D encoding.

**The Task:** Write a PyTorch function `get_3d_positional_encoding(coords, num_bands)` that transforms a set of 3D points into a high-dimensional feature vector.

**Requirements:**
1. *Input:* A tensor coords of shape $(B, N, 3)$ representing $N$ points in 3D space.
2. *Harmonics:* For each coordinate $(x, y, z)$, you must compute $L$ harmonic bands:$$\sin(2^L \pi \cdot x), \cos(2^L \pi \cdot x)$$
3. *Dimensionality:* The output should be of shape $(B, N, 3 + 3 \times 2 \times L)$. This includes the original coordinates plus the sine and cosine of each band.
4. *Efficiency:* You must implement this using broadcasting to avoid explicit loops over the coordinates or bands.

## Implementation Template
```Python
import torch
import math

def get_3d_positional_encoding(coords: torch.Tensor, num_bands: int) -> torch.Tensor:
    """
    Args:
        coords: (B, N, 3) tensor of xyz coordinates
        num_bands: Number of frequency bands (L)
    Returns:
        (B, N, D) encoded features
    """
    # Your code here
    pass
```

## Attempt \#1
```Python
import torch
import math

def get_3d_positional_encoding(coords: torch.Tensor, num_bands: int) -> torch.Tensor:
    """
    Args:
        coords: (B, N, 3) tensor of xyz coordinates
        num_bands: Number of frequency bands (L)
    Returns:
        (B, N, D) encoded features
    """
    # Your code here
    def get_harmonic_bands(x, L):
        # x.shape = B, N, 1
        if len(x.shape)==2:
            x = x[::, None]

        L_arange = torch.arange(L)
        buff = (2**L_arange)*torch.pi
        return torch.cat([
            torch.sin(buff[None, None,:]*x),
            torch.cos(buff[None, None,:]*x)
        ], dim=-1) # B, N, 2*L

    x_bands = get_harmonic_bands(coords[::, 0], num_bands)
    y_bands = get_harmonic_bands(coords[::, 1], num_bands)
    z_bands = get_harmonic_bands(coords[::, 2], num_bands)

    return torch.cat([
        coords,
        x_bands,
        y_bands,
        z_bands
    ], dim=-1)
```

## Gemini Implementation
```Py
import torch

def get_3d_positional_encoding(coords: torch.Tensor, num_bands: int) -> torch.Tensor:
    # coords shape: [B, N, 3]
    B, N, _ = coords.shape
    device = coords.device
    
    # 1. Create exponential frequencies: [2^0, 2^1, ..., 2^{L-1}]
    # Shape: [num_bands]
    frequencies = 2.0 ** torch.arange(num_bands, device=device)
    
    # 2. Scale by PI as per the formula
    frequencies = frequencies * torch.pi # [L]
    
    # 3. Outer product between coords and frequencies
    # [B, N, 3, 1] * [1, 1, 1, L] -> [B, N, 3, L]
    scaled_coords = coords.unsqueeze(-1) * frequencies
    
    # 4. Compute sin and cos
    # We flatten the last two dims (3 and L) into 3*L
    sin_feats = torch.sin(scaled_coords).reshape(B, N, -1)
    cos_feats = torch.cos(scaled_coords).reshape(B, N, -1)
    
    # 5. Concatenate everything: Original + Sin + Cos
    # Final Dim: 3 + (3*L) + (3*L)
    return torch.cat([coords, sin_feats, cos_feats], dim=-1)
```