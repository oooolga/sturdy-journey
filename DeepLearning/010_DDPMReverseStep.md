# The Challenge: Implementing the DDPM Reverse Step

**Scenario:** You are implementing the sampling loop for a DDPM (Denoising Diffusion Probabilistic Model). You have the noisy image $x_t$ and the noise predicted by your model $\epsilon_\theta$. Your goal is to compute $x_{t-1}$.

**The Math:** The formula for the reverse step is:$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
Where: $z \sim \mathcal{N}(0, \mathbf{I})$ if $t > 0$, and $z = 0$ if $t = 0$. $\sigma_t$ is typically set to $\sqrt{\beta_t}$ (or a variant like $\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t}$). For this challenge, use $\sqrt{\beta_t}$.

## Starter Code
```Python
import torch

def ddpm_step(x_t, predicted_noise, alpha_t, alpha_bar_t, beta_t, t):
    """
    Args:
        x_t: Current noisy image (N, C, H, W).
        predicted_noise: The epsilon predicted by the UNet.
        alpha_t: Scalar or (N, 1, 1, 1) tensor for the current timestep.
        alpha_bar_t: Cumulative product of alphas at timestep t.
        beta_t: Variance at timestep t.
        t: The current integer timestep (to check if it's the last step).

    Returns:
        The denoised image x_{t-1}.
    """
    # YOUR CODE HERE
    pass
```

## Attempt \# 1
```Python
import torch

def ddpm_step(x_t, predicted_noise, alpha_t, alpha_bar_t, beta_t, t):
    """
    Args:
        x_t: Current noisy image (N, C, H, W).
        predicted_noise: The epsilon predicted by the UNet.
        alpha_t: Scalar or (N, 1, 1, 1) tensor for the current timestep.
        alpha_bar_t: Cumulative product of alphas at timestep t.
        beta_t: Variance at timestep t.
        t: The current integer timestep (to check if it's the last step).

    Returns:
        The denoised image x_{t-1}.
    """
    # YOUR CODE HERE
    noise_term = 0.0 if t == 0 else beta_t**0.5 * torch.randn_like(x_t)
    return (x_t - (1-alpha_t)/(1-alpha_bar_t)**0.5*predicted_noise)/(alpha_t)**0.5 + noise_term
```