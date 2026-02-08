# The Challenge: The Vectorized Warper

**Objective:** Implement a differentiable warping module that takes a source image and its depth map, and "re-renders" it from the perspective of a second camera.

1. *The Inputs*:
    - `src_img`: $[B, C, H, W]$ — The feature map or image from Frame 1.
    - `depth_1`: $[B, 1, H, W]$ — The depth map of Frame 1.
    - `K`: $[B, 3, 3]$ — Camera intrinsics.
    - `T_1_to_2`: $[B, 4, 4]$ — The relative pose from Frame 1 to Frame 2.
2. *The Task*:
Complete the `VectorizedWarp` class. You must:
    1. Generate a coordinate grid for the image.
    2. Unproject the entire grid to 3D space using the depth map.
    3. Transform the 3D cloud to the second camera's frame.
    4. Project back to 2D and normalize to the $[-1, 1]$ range required by PyTorch's `grid_sample`.

## Starter Code
```Python
import torch
import torch.nn.functional as F

class VectorizedWarp(torch.nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.H, self.W = height, width
        # Pre-compute the pixel grid to save FLOPs
        # TODO: Create a grid of [u, v, 1] coordinates of shape [3, H*W]
        # Hint: use torch.meshgrid and torch.stack
        self.register_buffer("grid", self._make_grid())

    def _make_grid(self):
        # Your logic here to create the base pixel grid
        pass

    def forward(self, src_img, depth_1, K, T_1_to_2):
        B = src_img.shape[0]
        K_inv = torch.linalg.inv(K)
        
        # Step 1: Back-project Frame 1 pixels to 3D Camera Coords (P1)
        # P1 = K_inv @ pixels * depth
        # TODO: Implement vectorized back-projection
        
        # Step 2: Transform P1 to Frame 2 (P2)
        # P2 = R @ P1 + t
        # TODO: Extract R, t and transform
        
        # Step 3: Project P2 to Frame 2 Pixel Coords
        # TODO: Apply K and divide by Z
        
        # Step 4: Normalize to [-1, 1] for grid_sample
        # grid_sample expects coordinates in (x, y) format mapped to [-1, 1]
        # TODO: Normalize u2, v2
        
        # Step 5: Sample the image
        # sampled_img = F.grid_sample(src_img, sampling_grid, align_corners=True)
        return sampled_img
``` 

## Solution
```Python
import torch
import torch.nn.functional as F

class VectorizedWarp(torch.nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.H, self.W = height, width
        self.register_buffer("grid", self._make_grid())

    def _make_grid(self):
        # Your logic here to create the base pixel grid
        u, v = torch.meshgrid(torch.arange(self.H),
                              torch.arange(self.W),
                              indexing='xy')
        grid = torch.stack([u.float(), v.float(),
                            torch.ones_like(u)], dim=0)
        return grid.reshape(3, -1)

    def forward(self, src_img, depth_1, K, T_1_to_2):
        B = src_img.shape[0]
        K_inv = torch.linalg.inv(K)
        depth_1 = depth_1.reshape(B, -1) # B, H*W

        # Step 1: Back-project to Frame 1 Camera Coords (P1)
        # grid: [3, HW], K_inv: [B, 3, 3] -> P1: [B, 3, HW]
        P1 = K_inv @ self.grid
        P1 = P1 * depth_1.view(B, 1, -1)
        
        # Step 2: Transform P1 to Frame 2 (P2)
        # P2 = R @ P1 + t
        R = T_1_to_2[:, :3, :3]
        t = T_1_to_2[:, :3, 3:4]
        P2 = R @ P1 + t # [B, 3, HW]
        
        # Step 3: Project P2 to Frame 2 Pixel Coords
        p2_homo = K @ P2 # [B, 3, HW]
        u2 = p2_homo[:, 0] / (p2_homo[:, 2] + 1e-7)
        v2 = p2_homo[:, 1] / (p2_homo[:, 2] + 1e-7)
        
        # Step 4: Normalize to [-1, 1] for grid_sample
        # grid_sample expects (x, y) which is (u, v)
        u2_norm = (2 * u2 / (self.W - 1)) - 1
        v2_norm = (2 * v2 / (self.H - 1)) - 1
        
        # Reshape to [B, H, W, 2]
        sampling_grid = torch.stack([u2_norm, v2_norm], dim=-1).reshape(B, self.H, self.W, 2)
        
        # Step 5: Sample the image
        sampled_img = F.grid_sample(src_img, sampling_grid, 
                                    mode='bilinear', 
                                    padding_mode='zeros', 
                                    align_corners=True)
        
        # Bonus: Create a validity mask for pixels that moved out of frame
        mask = (u2_norm.abs() <= 1) & (v2_norm.abs() <= 1)
        mask = mask.reshape(B, 1, self.H, self.W).float()
        
        return sampled_img, mask
```