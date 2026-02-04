# The Coding Challenge: Depthwise Separable Convolution

**The Task:** Implement a Depthwise Separable Convolution block from scratch using basic PyTorch layers.

**Requirements:**
1. *The Concept:* A Depthwise Separable Convolution breaks a standard convolution into two distinct steps:
    - Depthwise Convolution: A single convolutional filter is applied to each input channel independently.
    - Pointwise Convolution: A $1 \times 1$ convolution is used to mix the information across channels.
2. *Input:* A 4D tensor of shape $(B, C_{in}, H, W)$.
3. *Parameters:* Your block should accept in_channels, out_channels, kernel_size, and stride.
4. *Comparison:* In a comment, briefly state the ratio of parameters between this and a standard convolution.

## Starter Code Template
```Python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # Your logic here
        # Hint: Look into the 'groups' parameter in nn.Conv2d
        pass

    def forward(self, x):
        # Apply the depthwise step, then the pointwise step
        pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # Your logic here
        # Hint: Look into the 'groups' parameter in nn.Conv2d
        self.depth_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding=kernel_size//2
        )
        
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        # Apply the depthwise step, then the pointwise step
        return self.pointwise_conv(self.depth_conv(x))
```