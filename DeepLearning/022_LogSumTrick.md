
3 The Challenge: Manual Stable Softmax Cross-Entropy

Implement a function `manual_cross_entropy(logits, targets)` that is numerically stable and behaves exactly like `torch.nn.functional.cross_entropy`.

**Requirements:**
1. Input: 
    1. *logits:* A tensor of shape $(N, C)$ representing unnormalized scores for $N$ samples across $C$ classes.
    2. *targets:* A tensor of shape $(N)$ containing the ground-truth class indices (long).
    3. *Numerical Stability:* You must implement the Log-Sum-Exp trick to prevent overflow.
    4. *No High-Level Fused Functions:* You are not allowed to use `F.cross_entropy`, `F.log_softmax`, or `torch.logsumexp`. You must use basic operations like `torch.exp`, `torch.log`, `torch.sum`, and `torch.gather`.
    5. *Output:* A single scalar representing the mean loss across the batch.
    
## Implementation Template
```Py
import torch

def manual_cross_entropy(logits, targets):
    """
    Args:
        logits: (N, C) 
        targets: (N,) - Class indices
    Returns:
        loss: scalar
    """
    # 1. Get the max of each row for stability (m)
    # 2. Compute Log-Sum-Exp: m + log(sum(exp(logits - m)))
    # 3. Compute Log-Softmax: logits - LSE
    # 4. Extract the log-probabilities of the target classes
    # 5. Return the negative mean
    pass

# --- Test Case ---
# logits = torch.tensor([[100.0, 101.0, 102.0], [1.0, 2.0, 3.0]])
# targets = torch.tensor([2, 0])
# Expected output should match F.cross_entropy(logits, targets)
```

## Gemini Solution
```Py
def manual_cross_entropy(logits, targets):
    N, C = logits.shape
    
    # 1. Get the max of each row (and keep the values only)
    # Using keepdim=True is vital for the broadcasting in step 2
    m, _ = torch.max(logits, dim=1, keepdim=True) 

    # 2. Compute Log-Sum-Exp
    # Note: sum() reduces the dimension, so we need to match it back to (N, 1)
    # or ensure broadcasting works correctly.
    lse = m + torch.log(torch.sum(torch.exp(logits - m), dim=1, keepdim=True))

    # 3. Compute Log-Softmax
    log_probs = logits - lse 

    # 4. Extract target log-probs using advanced indexing
    # We want log_probs[0, targets[0]], log_probs[1, targets[1]], etc.
    target_log_probs = log_probs[torch.arange(N), targets]
    
    # 5. Return the negative mean (Negative Log Likelihood)
    return -target_log_probs.mean()

# --- Test Case ---
# logits = torch.tensor([[100.0, 101.0, 102.0], [1.0, 2.0, 3.0]])
# targets = torch.tensor([2, 0])
# Expected output should match F.cross_entropy(logits, targets)
```
