# The Challenge: Depth to World-Space Point Cloud

**Scenario:** You are given a depth map $D$ (where each pixel value represents distance from the camera), a $3 \times 3$ Intrinsic matrix $K$, and a $4 \times 4$ Extrinsic matrix $T_{world \to cam}$ (World-to-Camera).

**Your Task:** Write a function depth_to_world_coords that converts a specific pixel coordinate $(u, v)$ and its depth $d$ into its $(x, y, z)$ coordinates in World Space.

**The Math Refresher:** 
1. *Pixel to Camera Space ($z$ is depth):*
$$x_c = \frac{(u - c_x) \cdot d}{f_x}$$
$$y_c = \frac{(v - c_y) \cdot d}{f_y}$$
$$z_c = d$$
2. *Camera Space to World Space:* $$P_{world} = T_{cam \to world} \cdot P_{camera}$$ (Remember that $T_{cam \to world}$ is the inverse of your extrinsic matrix $T_{world \to cam}$.)

## Starter Code
```Python
import torch

def depth_to_world_coords(u, v, d, K, T_world_to_cam):
    """
    Args:
        u, v: Pixel coordinates (integers).
        d: Depth value at that pixel (float).
        K: 3x3 Intrinsic matrix.
        T_world_to_cam: 4x4 Extrinsic matrix (World -> Camera).
        
    Returns:
        p_world: (3,) tensor representing (x, y, z) in World Space.
    """
    # YOUR CODE HERE
    pass
```

## Attempt \#1
```Python
import torch

def depth_to_world_coords(u, v, d, K, T_world_to_cam):
    # 1. Pixel to Camera Space (3D)
    # K_inv @ [u, v, 1] gives the ray [x/z, y/z, 1]
    pixel_homo = torch.tensor([u, v, 1.0], device=K.device)
    p_cam_3d = (torch.linalg.inv(K) @ pixel_homo) * d
    
    # 2. Camera to World Space (4x4 Matrix Multiplication)
    # Append 1.0 to make it homogeneous [x, y, z, 1]
    p_cam_4d = torch.cat([p_cam_3d, torch.ones(1, device=K.device)])
    
    # Invert Extrinsic (World->Cam becomes Cam->World)
    T_cam_to_world = torch.linalg.inv(T_world_to_cam)
    
    # Perform transform and slice back to 3D (first 3 elements)
    p_world = (T_cam_to_world @ p_cam_4d)[:3]
    
    return p_world
```