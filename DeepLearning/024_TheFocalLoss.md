# The Challenge: Implement Focal Loss

In many CV tasks (like detecting small objects in a video frame), the "background" class overwhelmingly outnumbers the "foreground" objects. Standard Cross-Entropy fails because the easy negatives dominate the gradient.

Focal Loss addresses this by adding a modulating factor $(1 - p_t)^\gamma$.

**The Task:** Implement a FocalLoss module in PyTorch.

**The Formula:** $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

**Requirements:**
1. *Inputs:*
    - logits: Unnormalized scores of shape $(N, C)$.
    - targets: Ground truth indices of shape $(N)$.
2. *Parameters:*
    - alpha: A weighting factor for the classes (use a scalar for simplicity, or a tensor for per-class).
    - gamma: The "focusing" parameter (usually set to 2.0).
3. Stability: You must use `log_softmax` to ensure numerical stability rather than computing probabilities directly.
4. Reduction: Return the mean loss across the batch.

# Implementation Template
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        log_pt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # If alpha is a tensor, we gather the specific weight for each target
        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.gather(0, targets)
        else:
            at = self.alpha

        # The focal loss modulating factor
        loss = - at * (1 - pt)**self.gamma * log_pt
        
        return loss.mean()
```