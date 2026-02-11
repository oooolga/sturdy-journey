# The Challenge: Rotational Consistency Check

In 4D reconstruction, you often have a camera moving around a dynamic object. To ensure your model isn't "drifting," you need to verify that a sequence of relative camera rotations correctly returns to the identity matrix.Task:Write a function is_path_closed(rotations, tol=1e-6) that takes a list of $3 \times 3$ rotation matrices (as PyTorch Tensors) representing relative transforms between consecutive frames ($R_{0 \to 1}, R_{1 \to 2}, ... R_{n \to 0}$).

Return `True` if the accumulated rotation represents a closed loop (the identity matrix $I$), and `False` otherwise.

## Implementation Template
```Python
import torch

def is_path_closed(rotations: list[torch.Tensor], tol: float = 1e-6) -> bool:
    # Your code here
    pass

    # Example: A sequence that should return True
    # R_total = R_n * ... * R_1 * R_0 = I
```

## Attempt \#1
```Python
import torch

def is_path_closed(rotations: list[torch.Tensor], tol: float = 1e-6) -> bool:
    # 1. Start with the Identity matrix (3x3)
    # Ensure it's on the same device/dtype as our input
    device = rotations[0].device
    dtype = rotations[0].dtype
    r_total = torch.eye(3, device=device, dtype=dtype)

    # 2. Accumulate rotations
    # Note: Matrix multiplication is associative, but order matters for 
    # extrinsic vs intrinsic transforms. Usually: R_total = R_n @ ... @ R_0
    for r_i in rotations:
        r_total = r_i @ r_total

    # 3. Create target Identity
    target = torch.eye(3, device=device, dtype=dtype)

    # 4. Robust Comparison: Frobenius Norm
    # torch.dist calculates the p-norm of (input - target)
    diff = torch.dist(r_total, target, p=2) 
    
    return diff < tol
```