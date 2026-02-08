# The Challenge: Manual 2D Convolution

Implement a 2D Convolution layer in PyTorch without using `torch.nn.Conv2d` or `torch.nn.functional.conv2d`.

**The Goal:** Implement the forward pass using the `im2col` (image-to-column) strategy. This is the industry-standard way to turn a convolution into a single large Matrix Multiplication (GEMM), which is how NVIDIA’s cuDNN actually executes them on a GPU.

**Requirements:**
1. **Parameters:** Support `in_channels`, `out_channels`, `kernel_size`, and `stride`. (Ignore padding/dilation for now to keep it clean).

2. **The Strategy:**
    - Extract patches from the input image.
    - Flatten each patch into a column.
    - Perform a single `torch.matmul` with the flattened kernels.
    - Reshape the result back to the output spatial dimensions.

3. **No Loops:** Try to avoid nesting for loops for spatial dimensions (height/width); use `torch.nn.functional.unfold` to handle the patch extraction.

## Starter Code Template
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Initialize weights: (out_channels, in_channels, k, k)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        B, C, H, W = x.shape
        K = self.kernel_size
        
        # 1. Calculate output dimensions
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        
        # YOUR TASK: 
        # 1. Use F.unfold to get the patches.
        # 2. Reshape weights for matrix multiplication.
        # 3. Compute the convolution via MatMul.
        # 4. Add bias and reshape to (B, out_channels, H_out, W_out).
        patches = F.unfold(x, kernel_size=K, stride=self.stride).transpose(-1, -2).contiguous() # B, num_patches, C*K*K
        weight = self.weight.reshape(self.out_channels, -1).transpose(0, 1) # C*K*K, out_channels
        ret = patches @ weight + self.bias # B, num_patches, out_channels
        return ret.transpose(-1, -2).contiguous().view(B, self.out_channels, H_out, W_out)
```

## Gemini's Solution
```Py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Standard Weight Shape: (out_channels, in_channels, k, k)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size
        
        # 1. Calculate output dimensions
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        
        # 2. Extract patches (im2col)
        # Shape: (B, C*K*K, L) where L = H_out * W_out
        patches = F.unfold(x, kernel_size=K, stride=self.stride)
        
        # 3. Reshape weight for MatMul
        # Shape: (out_channels, C*K*K)
        weight_flat = self.weight.view(self.out_channels, -1)
        
        # 4. Compute Convolution via GEMM (General Matrix Multiply)
        # (out_channels, C*K*K) @ (B, C*K*K, L) -> (B, out_channels, L)
        # We use matmul which handles the batch dimension automatically
        out = torch.matmul(weight_flat, patches)
        
        # 5. Add bias
        # bias shape: (out_channels) -> needs to be (1, out_channels, 1) for broadcasting
        out = out + self.bias.view(1, -1, 1)
        
        # 6. Reshape back to spatial dimensions
        return out.view(B, self.out_channels, H_out, W_out)
```