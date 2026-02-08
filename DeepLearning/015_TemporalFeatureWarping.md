# The Coding Challenge: Temporal Feature Warping

**The Goal:** Write a function that takes a pixel in Frame 1 and projects it into Frame 2 using the camera intrinsics and the relative transformation between the two camera poses.

Given Constants & Inputs
- *Intrinsics Matrix ($K$):* A $3 \times 3$ matrix representing the camera's focal length and optical center.
- *Extrinsics ($T_{1 \to 2}$):* A $4 \times 4$ relative transformation matrix (rotation and translation) that moves a point from Frame 1's coordinate system to Frame 2's coordinate system.
- *Frame 1 Data:* A pixel coordinate $\mathbf{p}_1 = [u_1, v_1, 1]^T$ and its scalar depth value $d_1$.

## The Mathematical Pipeline
To solve this, you must implement the following transformation chain:
1. *Back-projection:* Convert the 2D pixel $(u_1, v_1)$ to a 3D point $\mathbf{P}_1$ in Frame 1's camera coordinates using $K^{-1}$.$$\mathbf{P}_1 = d_1 \cdot K^{-1} \mathbf{p}_1$$
2. *Coordinate Transformation:* Apply the relative transform to move the point to Frame 2's coordinate system.$$\mathbf{P}_2 = R \mathbf{P}_1 + \mathbf{t}$$
3. Forward-projection: Project the 3D point $\mathbf{P}_2$ back onto the 2D image plane of Frame 2.$$\mathbf{p}_2 \simeq K \mathbf{P}_2$$

## Starter Code
(PyTorch Style)Fill in the missing logic in the function below:

```Python
import torch

def warp_pixel(u1, v1, d1, K, T_1_to_2):
    """
    Args:
        u1, v1: Pixel coordinates in Frame 1
        d1: Depth of the pixel in Frame 1
        K: Camera intrinsics (3x3)
        T_1_to_2: Relative pose (4x4)
    Returns:
        (u2, v2): Projected pixel coordinates in Frame 2
    """
    # 1. Create homogeneous pixel coordinate p1
    p1 = torch.tensor([u1, v1, 1.0]).reshape(3, 1)
    
    # 2. Back-project to 3D in Frame 1 (P1)
    # TODO: P1 = ...
    
    # 3. Transform P1 to P2 using T_1_to_2
    # Hint: Use the R (3x3) and t (3x1) from the 4x4 matrix
    # TODO: P2 = ...
    
    # 4. Project P2 to Frame 2 pixel coordinates (p2)
    # TODO: p2 = ...
    
    # 5. Normalize p2 by its depth (the z-component)
    u2 = p2[0] / p2[2]
    v2 = p2[1] / p2[2]
    
    return u2.item(), v2.item()
```

## Attempt \#1

```Python
import torch

def warp_pixel(u1, v1, d1, K, T_1_to_2):
    """
    Args:
        u1, v1: Pixel coordinates in Frame 1
        d1: Depth of the pixel in Frame 1
        K: Camera intrinsics (3x3)
        T_1_to_2: Relative pose (4x4)
    Returns:
        (u2, v2): Projected pixel coordinates in Frame 2
    """
    # 1. Create homogeneous pixel coordinate p1
    p1 = torch.tensor([u1, v1, 1.0], device=K.device).reshape(3, 1)
    
    # 2. Back-project to 3D in Frame 1 (P1)
    p1_cam = torch.linalg.inv(K) @ p1 * d1

    # 3. Transform P1 to P2 using T_1_to_2
    # Hint: Use the R (3x3) and t (3x1) from the 4x4 matrix
    p1_cam_hom = torch.cat([p1_cam, torch.ones(1,).to(K.device)], dim=0)
    p2_cam_hom = T_1_to_2 @ p1_cam_hom
    p2_cam = p2_cam_hom[:3]
    
    # 4. Project P2 to Frame 2 pixel coordinates (p2)
    p2 = K @ p2_cam
    
    # 5. Normalize p2 by its depth (the z-component)
    u2 = p2[0] / p2[2]
    v2 = p2[1] / p2[2]
    
    return u2.item(), v2.item()
```