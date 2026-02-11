# The Problem: "6D Rotation Representation"

Many state-of-the-art models (like PyTorch3D and various Flow Matching papers) avoid Quaternions and instead use a 6D Rotation Representation. This representation is continuous and much easier for a Neural Network to learn.

**Task:** Implement a function rotation_6d_to_matrix(d6: torch.Tensor) that converts a 6D vector into a full $3 \times 3$ rotation matrix.

**The Algorithm:** Given a 6D input $[a_1, a_2]$ (where $a_1, a_2 \in \mathbb{R}^3$):
1. Normalize $a_1$ to get the first column of the rotation matrix ($b_1$).
2. Use the Gram-Schmidt process to find a vector $b_2$ that is orthogonal to $b_1$ and normalized.
3. Compute the third column $b_3$ using the cross product ($b_1 \times b_2$).
4. Stack $[b_1, b_2, b_3]$ to form the $3 \times 3$ matrix.

## Implementation Template
```Python
import torch

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Args:
        d6: Tensor of shape (B, 6)
    Returns:
        Rotation matrices of shape (B, 3, 3)
    """
    # Your code here
    pass
```

## Attempt \#1
```Python
import torch
import torch.nn.functional as F

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    # 1. Split into two 3D vectors: [B, 3] each
    a1, a2 = d6[:, :3], d6[:, 3:]
    
    # 2. Normalize a1 to get b1
    # Use keepdim=True so we can divide [B, 3] by [B, 1]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    
    # 3. Dot product of a2 and b1: [B, 1]
    # Sum across the vector dimension to get the scalar projection
    dot_product = torch.sum(b1 * a2, dim=-1, keepdim=True)
    
    # 4. Remove b1 component from a2 and normalize to get b2
    u2 = a2 - dot_product * b1
    b2 = F.normalize(u2, dim=-1, eps=1e-6)
    
    # 5. Cross product for the final column
    b3 = torch.cross(b1, b2, dim=-1)
    
    # 6. Stack into (B, 3, 3)
    # This stacks columns. If you want rows, stack and then transpose.
    return torch.stack([b1, b2, b3], dim=-1)
```