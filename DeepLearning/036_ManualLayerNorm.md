# The Problem: "Manual LayerNorm Implementation"

**Task:** Write a PyTorch `nn.Module` called `ManualLayerNorm` that implements the Layer Normalization operation for a 2D input (Batch, Features).

**The Equation:** For a vector $x$ of dimension $D$:$$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta$$
Where:
- $\mathrm{E}[x]$ and $\mathrm{Var}[x]$ are the mean and variance calculated across the feature dimension for each individual sample in the batch.
- $\gamma$ (scale) and $\beta$ (shift) are learnable parameters initialized to 1s and 0s respectively.
- $\epsilon$ is a small constant for numerical stability.

## Implementation Template
```Python
import torch
import torch.nn as nn
import torch.nn.Functional as F

class ManualLayerNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Initialize gamma and beta as learnable parameters
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, feature_dim]
        # Your code here
        pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn
import torch.nn.Functional as F

class ManualLayerNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Initialize gamma and beta as learnable parameters
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, feature_dim]
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x-x_mean)/torch.sqrt(x_var + self.eps)
        return x_norm * self.gamma + self.beta
```