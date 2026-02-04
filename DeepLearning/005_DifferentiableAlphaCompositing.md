# The Coding Challenge: Differentiable Alpha-Compositing

**The Task:** Implement the front-to-back alpha blending function in PyTorch. This function takes a sequence of colors and opacities (already sorted by depth) and computes the final pixel color.

**Requirements:**
1. Input: 
    - colors: A tensor of shape $(B, N, 3)$ representing the RGB colors of $N$ Gaussians covering a pixel.
    - opacities: A tensor of shape $(B, N, 1)$ representing the opacity ($\alpha$) of each Gaussian
2. Equation: The color $C$ is calculated as:$$C = \sum_{i=1}^{N} c_i \alpha_i T_i, \quad \text{where } T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$(Where $T_i$ is the "transmittance"—the probability that the ray has not been hit by any Gaussian before index $i$).
3. Efficiency: Implement this using `torch.cumprod` to avoid explicit Python loops, ensuring the operation is differentiable and fast.

## Starter Code Template
```Python
import torch
import torch.nn as nn

def accumulate_alpha_blending(colors, opacities):
    """
    Args:
        colors: (B, N, 3) - RGB values
        opacities: (B, N, 1) - Alpha values (0 to 1)
    Returns:
        final_color: (B, 3)
    """
    # B: Batch (or number of pixels)
    # N: Number of Gaussians sorted by depth
    
    # 1. Compute (1 - alpha) for each Gaussian
    # 2. Compute the cumulative product to get Transmittance (Ti)
    # 3. Apply the blending formula
    pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn

def accumulate_alpha_blending(colors, opacities):
    """
    Args:
        colors: (B, N, 3) - RGB values
        opacities: (B, N, 1) - Alpha values (0 to 1)
    Returns:
        final_color: (B, 3)
    """
    # B: Batch (or number of pixels)
    # N: Number of Gaussians sorted by depth
    
    B, N, _ = colors.shape

    # 1. Compute (1 - alpha) for each Gaussian
    non_opacities = 1.0-opacities # B, N, 1

    # 2. Compute the cumulative product to get Transmittance (Ti)
    temp = torch.ones_like(non_opacities[:, :1, :])
    temp = torch.cat((temp, non_opacities[:, :-1, :]), dim=1)

    transmittance = torch.cumprod(temp, dim=1)

    # 3. Apply the blending formula
    return torch.sum(colors * opacities * transmittance, dim=1)
```