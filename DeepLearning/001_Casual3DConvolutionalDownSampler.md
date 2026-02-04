# The Coding Challenge: Causal 3D Convolutional Downsampler

**The Task:** Implement a Causal 3D Convolutional Block in PyTorch. This is a specific type of convolution used in video modeling (like in Sora or Magvit-v2) to ensure that the representation of a frame at time $t$ does not depend on future frames ($t+1$).

**Requirements:**
1. Input: A 5D tensor $(B, C_{in}, T, H, W)$.
2. Causal Padding: Implement manual padding in the temporal dimension so that a kernel of size $k$ only looks at the current and past frames.
3. Downsampling: The block should reduce spatial dimensions $(H, W)$ by a factor of 2 while keeping the temporal dimension $T$ the same.
4. **Efficiency:** Use `nn.Conv3d` as your base, but manage the padding logic yourself.

## Start Code Template
```Python
import torch
import torch.nn as nn

class CausalDownsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        """
        Args:
            kernel_size (tuple): (kT, kH, kW)
        """
        super().__init__()
        self.kT, self.kH, self.kW = kernel_size
        
        # We want spatial downsampling (stride 2 on H, W) 
        # but temporal preservation (stride 1 on T)
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=(1, 2, 2),
            padding=0 # We will handle padding manually
        )

    def forward(self, x):
        # B, C, T, H, W = x.shape
        # 1. Apply Causal Padding to T
        # 2. Apply Symmetric/Reflective Padding to H, W
        # 3. Apply Convolution
        pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDownsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        """
        Args:
            kernel_size (tuple): (kT, kH, kW)
        """
        super().__init__()
        self.kT, self.kH, self.kW = kernel_size
        
        # We want spatial downsampling (stride 2 on H, W) 
        # but temporal preservation (stride 1 on T)
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=(1, 2, 2),
            padding=0 # We will handle padding manually
        )

    def forward(self, x):
        # B, C, T, H, W = x.shape
        # 1. Apply Causal Padding to T
        T_padding = self.kT-1, 0
        # 2. Apply Symmetric/Reflective Padding to H, W
        H_padding = self.kH//2, self.kH//2
        W_padding = self.kW//2, self.kW//2

        x = torch.pad(x, (*W_padding, *H_padding, *T_padding))
        # 3. Apply Convolution
        return self.conv(x)
```

**Gemini Feedback:** This is almost perfect. Your understanding of causal padding—padding only the "past" and not the "future"—is spot on. In a coding interview, this would likely be accepted as a correct solution, though there are two minor "industry-standard" tweaks you should know to make it bulletproof.
1. The `torch.pad` vs `F.pad` NuanceWhile `torch.pad` works in many versions, the standard PyTorch API for functional padding is `torch.nn.functional.pad` (which you already imported as `F`).
2. The Symmetric Spatial PaddingIn your code, you use //2. This is correct for odd kernel sizes (like $3 \times 3$). However, if an interviewer gives you an even kernel size (like $4 \times 4$ for patch-based downsampling), a symmetric //2 on both sides might result in an output shape that is off-by-one.