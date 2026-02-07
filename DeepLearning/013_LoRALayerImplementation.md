# The Challenge: Implementing the LoRA Layer

**Scenario:** You are tasked with converting a standard `nn.Linear` layer into a LoRA-enabled layer. Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times k}$, you will keep $W$ frozen and learn two smaller matrices, $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$, where $r$ (the rank) is much smaller than $d$ or $k$.

**The Math:** The modified forward pass is:$$h = Wx + \frac{\alpha}{r}(BA)x$$
Where:
- $W$ is the frozen pre-trained weight.
- $A$ is initialized with Gaussian noise.
- $B$ is initialized with zeros (to ensure the training starts with the original model's behavior).
- $\alpha$ is a scaling constant.

## Starter Code
```Python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=1):
        super().__init__()
        self.base_layer = base_layer  # This is the frozen nn.Linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Determine dimensions from the base_layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # YOUR CODE HERE: Define lora_A and lora_B
        
        # YOUR CODE HERE: Freeze the base_layer
        
        # YOUR CODE HERE: Initialize weights
        
    def forward(self, x):
        # YOUR CODE HERE: Implement the forward pass
        pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=1):
        super().__init__()
        self.base_layer = base_layer  # This is the frozen nn.Linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Determine dimensions from the base_layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # YOUR CODE HERE: Define lora_A and lora_B
        self.lora_A = nn.Parameter(torch.randn(in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, out_features))
        
        # YOUR CODE HERE: Freeze the base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # YOUR CODE HERE: Initialize weights

        
    def forward(self, x):
        # YOUR CODE HERE: Implement the forward pass
        x = self.base_layer(x) + self.scaling * (self.lora_A @ self.lora_B) * x
        return x
```

## The Critique
1. **The Matrix Multiplication Bug:** In your code: `(self.lora_A @ self.lora_B) * x`

    - `self.lora_A @ self.lora_B results` in a matrix of size `(in_features, out_features)`.
    - `x` usually has the shape `(Batch, Seq_Len, in_features)`.
    - You cannot perform element-wise multiplication (`*`) between a weight matrix and an input activation in this way. You need to perform a linear transformation (matrix multiplication).
2. **Computational Efficiency (The "LoRA Secret"):** 
    - The whole point of LoRA is to avoid the $O(d \times k)$ cost of the large matrix. 
    - Your code calculates $W_{new} = (A \cdot B)$, which is a large matrix, and then applies it.
    - **The Better Way:** Calculate $B(Ax)$.
    - **Complexity:** If $x$ is $(1 \times d)$, then $(x \cdot A) \cdot B$ costs $O(d \cdot r + r \cdot k)$, whereas $(A \cdot B)$ costs $O(d \cdot r \cdot k)$. Since $r$ is tiny (e.g., 4 or 8), the associative property of matrix multiplication saves you massive amounts of memory and time.

## The Corrected Implementation
```Python 
def forward(self, x):
    # 1. Original frozen path
    result = self.base_layer(x)
    
    # 2. LoRA path: x -> A -> B
    # Efficient order: (x @ A) @ B
    # x is (B, S, in_dim), lora_A is (in_dim, r), lora_B is (r, out_dim)
    lora_out = (x @ self.lora_A) @ self.lora_B
    
    # 3. Scale and add
    return result + lora_out * self.scaling
```

## Attempt \#2
```Python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=1):
        super().__init__()
        self.base_layer = base_layer  # This is the frozen nn.Linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Determine dimensions from the base_layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # YOUR CODE HERE: Define lora_A and lora_B
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)

        # YOUR CODE HERE: Freeze the base_layer
        self.base_layer.requires_grad_(False)
        
        # YOUR CODE HERE: Initialize weights
        nn.init.normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # YOUR CODE HERE: Implement the forward pass
        return self.base_layer(x) + self.scaling*self.lora_B(self.lora_A(x))
```

## Follow-up Question
Since $h = Wx + (BA)x$ is equivalent to $h = (W + BA)x$, how would you "merge" these weights into a single `nn.Linear` layer after training is finished?

**Solution**
```Python
@torch.no_grad()
def merge_lora(self):
    # Calculate the delta weight: (out, rank) @ (rank, in) -> (out, in)
    # Note: B.weight is (out, rank), A.weight is (rank, in)
    weight_delta = self.lora_B.weight @ self.lora_A.weight
    
    # Update base layer in-place
    self.base_layer.weight += weight_delta * self.scaling
```

Why do we use `@torch.no_grad()` when merging?

*Answer:* Since we are manually modifying the weight tensor, we want to ensure PyTorch doesn't try to build a computational graph for this operation, which would waste memory and potentially interfere with the state if you were still in a training loop.