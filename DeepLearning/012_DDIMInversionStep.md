# The Challenge: Implementing the Inversion Step

**The Logic:**
To move from $x_t$ to $x_{t+1}$:
1. *Predict $x_0$:* Same as the denoising step, use the current $x_t$ and the current $\bar{\alpha}_t$ to estimate the clean image.
2. *Step Forward:* Instead of scaling $x_0$ and the noise by the previous (smaller) alpha, you scale them by the next (larger) alpha.

## Starter Code

```Python
import torch

def ddim_inversion_step(x_t, predicted_noise, alpha_bar_t, alpha_bar_t_next):
    """
    Args:
        x_t: The image/latent at current timestep t.
        predicted_noise: The epsilon predicted by the model at (x_t, t).
        alpha_bar_t: Cumulative alpha at current timestep t.
        alpha_bar_t_next: Cumulative alpha at the NEXT timestep (t+1).
    
    Returns:
        The noisier image/latent x_{t+1}.
    """
    # YOUR CODE HERE
    pass
```

## Attempt \#1

```Python
import torch

def ddim_inversion_step(x_t, predicted_noise, alpha_bar_t, alpha_bar_t_next):
    """
    Args:
        x_t: The image/latent at current timestep t.
        predicted_noise: The epsilon predicted by the model at (x_t, t).
        alpha_bar_t: Cumulative alpha at current timestep t.
        alpha_bar_t_next: Cumulative alpha at the NEXT timestep (t+1).
    
    Returns:
        The noisier image/latent x_{t+1}.
    """
    x_0_pred = (x_t - torch.sqrt(1-alpha_bar_t)*predicted_noise)/torch.sqrt(alpha_bar_t)
    inverse_dir = torch.sqrt(1-alpha_bar_t_next)*predicted_noise
    return torch.sqrt(alpha_bar_t_next)*x_0_pred+inverse_dir
```