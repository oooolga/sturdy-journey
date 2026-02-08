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

# Similar Challenge: The 3DGS Rasterization Pipeline

In a coding interview for a Graphics or Vision Engineer role, you might be asked to explain or implement the logic behind the "Splatting" process.

**The Scenario:** You have a set of 3D Gaussians defined by their center $\mu$, covariance $\Sigma$, opacity $\alpha$, and color $c$ (often via Spherical Harmonics). You need to project these onto a 2D image plane.

**The Math:** To project a 3D Gaussian to 2D, we use the Jacobian of the projective transformation $J$. The 2D covariance matrix $\Sigma'$ is approximated as:$$\Sigma' = J \cdot W \cdot \Sigma \cdot W^T \cdot J^T$$where $W$ is the viewing transformation (world-to-camera).

## The Coding Challenge: The Tiled Alpha-Blending Step
Once the Gaussians are projected to 2D and sorted by depth, we calculate the final color of a pixel using Point-Based Alpha Blending (similar to Volumetric Rendering in NeRF).

**Your Task:** Write a function compute_pixel_color that calculates the final color $C$ of a pixel by iterating through $N$ sorted Gaussians that overlap with that pixel.

**The Formula:** $$C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

## Starter Code
```Python
import torch

def compute_pixel_color(colors, opacities, densities):
    # 1. Compute effective alpha for each Gaussian at this pixel
    alphas = opacities * densities # Shape: (N,)
    
    # 2. Compute Transmittance (T_i): probability that light reaches this point
    # T_i = Product of (1 - alpha_j) for all j < i
    # We use a trick: cumprod of (1 - alphas), then shift by 1
    oneminus_alpha = 1.0 - alphas + 1e-7 # Small epsilon for stability
    
    # cumprod gives: [(1-a0), (1-a0)(1-a1), (1-a0)(1-a1)(1-a2), ...]
    cumprod_probs = torch.cumprod(oneminus_alpha, dim=0)
    
    # Roll/Shift to get T_i: [1, (1-a0), (1-a0)(1-a1), ...]
    # This ensures the first Gaussian has a transmittance of 1.0
    transmittance = torch.ones_like(cumprod_probs)
    transmittance[1:] = cumprod_probs[:-1]
    
    # 3. Final Color: Sum of (Color * Alpha * Transmittance)
    # weights shape: (N,)
    weights = alphas * transmittance
    
    # result shape: (3,)
    return torch.sum(colors * weights[:, None], dim=0)
```