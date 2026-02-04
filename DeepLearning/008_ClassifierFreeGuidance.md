# The Challenge: Implementing the CFG Inference Step
**Scenario:** You are building the inference pipeline for a Diffusion model. You have a model that takes two inputs: the noisy latent $z_t$ and a conditioning vector $c$.

**Your Task:** Write a function `compute_cfg_output` that calculates the guided noise prediction. Your implementation must handle the "null-conditioning" internally and apply the guidance scale formula.

**Constraints/Context:**
- The `model(z_t, c)` function returns the predicted noise.
- Assume `null_cond` is a pre-defined constant representing the embedding for an empty prompt.
- The guidance scale $w$ is a float.

## Starter Code
```Python
import torch

def compute_cfg_output(model, z_t, cond, null_cond, guidance_scale):
    """
    Args:
        model: The diffusion model (U-Net or Transformer).
        z_t: The current noisy latent at timestep t.
        cond: The text/label conditioning embedding.
        null_cond: The 'unconditional' or 'empty' embedding.
        guidance_scale: The CFG weight (w).
        
    Returns:
        The guided noise prediction (epsilon_hat).
    """
    # YOUR CODE HERE
    pass
```

## Attempt \#1
```Python
import torch

def compute_cfg_output(model, z_t, cond, null_cond, guidance_scale):
    """
    Args:
        model: The diffusion model (U-Net or Transformer).
        z_t: The current noisy latent at timestep t.
        cond: The text/label conditioning embedding.
        null_cond: The 'unconditional' or 'empty' embedding.
        guidance_scale: The CFG weight (w).
        
    Returns:
        The guided noise prediction (epsilon_hat).
    """
    # YOUR CODE HERE
    noise_uncond = model(z_t, null_cond)
    noise_cond = model(z_t, cond)

    return noise_uncond + guidance_scale*(noise_cond-noise_uncond)
```