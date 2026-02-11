# The Challenge: "Time-Conditioned Flow Layer"

In Flow Matching (FM) for 4D reconstruction, the model doesn't just predict a static 3D shape; it predicts a velocity field $v_t(x)$ that changes over time $t$. A common mistake is failing to properly "inject" the time scalar into the neural network's feature map.

**Task:** Write a PyTorch `nn.Module` called `FlowCondLayer` that implements a time-conditioned linear layer. The layer should:
1. Take two inputs: a batch of 3D features $x \in \mathbb{R}^{B \times D}$ and a batch of time scalars $t \in \mathbb{R}^{B}$.
2. Convert the scalar $t$ into a Sinusoidal Positional Embedding of dimension $D$.
3. Use this time embedding to scale and shift the feature $x$ (Adaptive Layer Norm style) before passing it through a standard linear projection.

**Mathematical Requirements:** The scaling and shifting should follow the formula:$$y = \text{Linear}(x \cdot (1 + \gamma(t)) + \beta(t))$$ Where $\gamma(t)$ and $\beta(t)$ are MLPs that project the time embedding.

## Implementation Template
```Py
import torch
import torch.nn as nn
import math

class FlowCondLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Define your time embedding projection, gamma, beta, and linear layers here
        
    def get_time_embedding(self, t: torch.Tensor):
        # Implement sinusoidal embedding for t
        # PE(t, 2i) = sin(t / 10000^(2i/dim))
        pass

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x shape: [B, D]
        # t shape: [B]
        pass
```

## Solution
```Python
import torch
import torch.nn as nn
import math

class FlowCondLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Scale (gamma) and Shift (beta) predictors
        self.ada_norm = nn.Linear(dim, 2 * dim)
        self.out_layer = nn.Linear(dim, dim)
        
    def get_time_embedding(self, t: torch.Tensor):
        half_dim = self.dim // 2
        # Calculate frequencies in log space for stability
        # freq = exp(-log(10000) * i / half_dim)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device).float() / half_dim
        )
        
        # outer product: [B, 1] * [1, half_dim] -> [B, half_dim]
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1. Embed time
        t_em = self.get_time_embedding(t) # [B, D]
        t_em = self.time_mlp(t_em)       # [B, D]
        
        # 2. Predict gamma and beta in one pass
        conditioning = self.ada_norm(t_em) # [B, 2*D]
        gamma, beta = conditioning.chunk(2, dim=-1) # [B, D], [B, D]
        
        # 3. Apply AdaLN-style modulation and project
        # y = Linear( x * (1 + gamma) + beta )
        return self.out_layer(x * (1 + gamma) + beta)
```