# The Challenge: LoRA-Integrated Multi-Head Self-Attention
**The Task:** Implement a `LoRAMultiHeadSelfAttention` module in PyTorch that integrates Low-Rank Adaptation (LoRA) directly into the $W_q$ and $W_v$ projections.

**Requirements:** 
1. *Frozen Base:* The original projection weights must be frozen, while only the LoRA matrices (A and B) are trainable.
2. *Scaling:* Include the $\frac{\alpha}{r}$ scaling factor where $r$ is the rank.
3. *Initialization:* Use the standard LoRA initialization (Matrix $A$ with Kaiming/Gaussian and Matrix $B$ with zeros) to ensure the initial output of the adapter is zero.
4. *Efficiency:* Instead of creating separate `nn.Linear` layers for LoRA, implement them as `nn.Parameter` pairs to show you understand the underlying math: $W_{updated} = W_{base} + \frac{\alpha}{r}(B \times A)$.

## Starter Code Template
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=8, alpha=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Projections (Frozen)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 2. LoRA Parameters for Q and V
        # Q-Adapter
        self.lora_q_A = nn.Parameter(torch.randn(embed_dim, rank))
        self.lora_q_B = nn.Parameter(torch.zeros(rank, embed_dim))
        
        # V-Adapter
        self.lora_v_A = nn.Parameter(torch.randn(embed_dim, rank))
        self.lora_v_B = nn.Parameter(torch.zeros(rank, embed_dim))

        self.reset_parameters()
        self.freeze_base_weights()

    def reset_parameters(self):
        # Standard LoRA init: A is Gaussian, B is Zero
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B)
        nn.init.zeros_(self.lora_v_B)

    def freeze_base_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            proj.weight.requires_grad = False

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        
        # YOUR TASK: Implement the forward pass such that:
        # 1. Query uses (base_weight + lora_update)
        # 2. Key uses only base_weight
        # 3. Value uses (base_weight + lora_update)
        # 4. Perform standard SDPA (Scaled Dot Product Attention)
        pass
```

## Attempt \#1
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=8, alpha=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Projections (Frozen)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 2. LoRA Parameters for Q and V
        # Q-Adapter
        self.lora_q_A = nn.Parameter(torch.randn(embed_dim, rank))
        self.lora_q_B = nn.Parameter(torch.zeros(rank, embed_dim))
        
        # V-Adapter
        self.lora_v_A = nn.Parameter(torch.randn(embed_dim, rank))
        self.lora_v_B = nn.Parameter(torch.zeros(rank, embed_dim))

        self.reset_parameters()
        self.freeze_base_weights()

    def reset_parameters(self):
        # Standard LoRA init: A is Gaussian, B is Zero
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B)
        nn.init.zeros_(self.lora_v_B)

    def freeze_base_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            proj.weight.requires_grad = False

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        B, S, D = x.shape

        # YOUR TASK: Implement the forward pass such that:
        # 1. Query uses (base_weight + lora_update)
        # 2. Key uses only base_weight
        # 3. Value uses (base_weight + lora_update)
        # 4. Perform standard SDPA (Scaled Dot Product Attention)
        q = self.q_proj(x) + self.scaling * (x @ self.lora_q_A) @ self.lora_q_B
        k = self.k_proj(x)
        v = self.v_proj(x) + self.scaling * (x @ self.lora_v_A) @ self.lora_v_B

        q = q.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # B, H, S, D_H
        k = k.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-1, -2)) / torch.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1) # B, H, S, S
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().reshape(B, S, -1)
        return out
```

## Gemini Solution
```Python
def forward(self, x):
    # x shape: (B, S, D)
    B, S, _ = x.shape

    # 1. Compute Projections
    # Note: We compute LoRA in parallel to the base projection for efficiency
    q = self.q_proj(x) + (x @ self.lora_q_A @ self.lora_q_B) * self.scaling
    k = self.k_proj(x)
    v = self.v_proj(x) + (x @ self.lora_v_A @ self.lora_v_B) * self.scaling

    # 2. Reshape for Multi-Head Attention: (B, S, D) -> (B, H, S, Dh)
    q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    # 3. Scaled Dot Product Attention
    # This replaces manual softmax(q@k/scale)@v and is fused for speed
    attn_output = F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=None, 
        dropout_p=0.0 if not self.training else 0.1,
        is_causal=False
    )

    # 4. Recombine Heads: (B, H, S, Dh) -> (B, S, D)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
    
    return attn_output
```