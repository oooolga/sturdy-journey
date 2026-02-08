# The Challenge: Implement ManualBatchNorm2d
Implement a 2D Batch Normalization layer in PyTorch without using `torch.nn.BatchNorm2d`.

**Requirements:**
1. **Parameters:** Support `num_features`, `eps` (for numerical stability), and `momentum` (for updating running stats).
2. **Training vs. Eval:** 
    - During Training, calculate the mean and variance of the current batch. Update the `running_mean` and `running_var` using the momentum formula.
    - During Inference: Use the saved `running_mean` and `running_var`.
3. **Affine Transform:** Implement the learnable parameters $\gamma$ (weight) and $\beta$ (bias).
4. **Spatial Reduction:** Remember that for 2D images $(B, C, H, W)$, you must calculate the mean and variance across the Batch, Height, and Width dimensions (everything except the channel dimension).

## Implementation Template
```Python
import torch
import torch.nn as nn

class ManualBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 1. Learnable parameters (Gamma and Beta)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 2. Non-trainable buffers for inference (Running Stats)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        
        if self.training:
            # YOUR TASK:
            # 1. Calculate mean and var for the current batch (across B, H, W)
            # 2. Update running_mean and running_var using momentum:
            #    running = (1 - momentum) * running + momentum * batch_stat
            # 3. Normalize: (x - mean) / sqrt(var + eps)
            pass
        else:
            # 4. Normalize using running stats
            pass

        # 5. Apply Gamma and Beta (Scale and Shift)
        # 6. Return the result
        pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn

class ManualBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 1. Learnable parameters (Gamma and Beta)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 2. Non-trainable buffers for inference (Running Stats)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        
        B, C, H, W = x.shape
        if self.training:
            # YOUR TASK:
            # 1. Calculate mean and var for the current batch (across B, H, W)
            dims = (0, 2, 3)
            batch_mean = x.mean(dim=dims, keepdim=True) # dim=(c,)
            batch_var = x.var(dim=dims, keepdim=True, unbiased=False) # 1/n variance, dim=(c,)

            # 2. Update running_mean and running_var using momentum:
            #    running = (1 - momentum) * running + momentum * batch_stat
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()

            # Compute unbiased var
            n = B*H*W
            unbiased_var = batch_var.squeeze() * n / (n-1)
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * unbiased_var

            # 3. Normalize: (x - mean) / sqrt(var + eps)
            x_normalized = (x-batch_mean) / torch.sqrt(batch_var+self.eps)
        else:
            # 4. Normalize using running stats
            x_normalized = (x-self.running_mean[None, :, None, None]) / torch.sqrt(self.running_var[None, :, None, None]+self.eps)

        # 5. Apply Gamma and Beta (Scale and Shift)
        x_normalized = self.gamma[None, :, None, None] * x_normalized + self.beta[None, :, None, None]
        # 6. Return the result
        return x_normalized
```

## Gemini Solution
```Python
def forward(self, x):
    B, C, H, W = x.shape
    dims = (0, 2, 3)
    
    if self.training:
        batch_mean = x.mean(dim=dims, keepdim=True)
        batch_var = x.var(dim=dims, keepdim=True, unbiased=False)

        # Update buffers correctly
        with torch.no_grad():
            # .copy_ ensures the data is updated in the existing buffer
            self.running_mean.copy_((1 - self.momentum) * self.running_mean + 
                                    self.momentum * batch_mean.squeeze())
            
            n = B * H * W
            unbiased_var = batch_var.squeeze() * (n / (n - 1))
            self.running_var.copy_((1 - self.momentum) * self.running_var + 
                                    self.momentum * unbiased_var)

        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
    else:
        # Using your concise None indexing
        mean = self.running_mean[None, :, None, None]
        var = self.running_var[None, :, None, None]
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

    return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]
```