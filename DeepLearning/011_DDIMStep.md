# The Challenge: The DDIM Step

**The Math:**
The DDIM update rule (with $\sigma=0$ for full determinism) is:
1. *Predict $x_0$:* First, estimate the clean image from the current noisy $x_t$:$$x_{0, \text{pred}} = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} $$
2. *Direction pointing to $x_t$:* Calculate the "re-noised" component that points back toward the current noise level:$$\text{dir\_xt} = \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$
3. *Final Update:* $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_{0, \text{pred}} + \text{dir\_xt}$$

## Your Task
Write a function ddim_step that implements this deterministic update.
```Python
import torch

def ddim_step(x_t, predicted_noise, alpha_bar_t, alpha_bar_t_prev):
    """
    Args:
        x_t: Current noisy image (N, C, H, W).
        predicted_noise: The epsilon predicted by the model.
        alpha_bar_t: Cumulative alpha at current timestep t.
        alpha_bar_t_prev: Cumulative alpha at the previous timestep (t-1).
                          Note: This allows for "jump-steps" (e.g., t=100, t_prev=50).

    Returns:
        The deterministic denoised image x_{t-1}.
    """
    # YOUR CODE HERE
    pass
```

## Attempt \#1

```Python
import torch

def ddim_step(x_t, predicted_noise, alpha_bar_t, alpha_bar_t_prev):
    """
    Args:
        x_t: Current noisy image (N, C, H, W).
        predicted_noise: The epsilon predicted by the model.
        alpha_bar_t: Cumulative alpha at current timestep t.
        alpha_bar_t_prev: Cumulative alpha at the previous timestep (t-1).
                          Note: This allows for "jump-steps" (e.g., t=100, t_prev=50).

    Returns:
        The deterministic denoised image x_{t-1}.
    """
    # YOUR CODE HERE
    dir_xt = (1-alpha_bar_t_prev)**0.5 * predictd_noise
    x_0_pred = (x_t-((1-alpha_bar_t)**0.5 * predictd_noise))/(alpha_bar_t**0.5)
    return (alpha_bar_t_prev ** 0.5) * x_0_pred + dir_xt
```