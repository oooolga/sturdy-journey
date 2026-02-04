# The Coding Challenge: Patch Embedding Layer

**The Task:** Implement the Patch Embedding module. In a ViT, this is the very first layer that transforms an image into a sequence of tokens that a Transformer can understand.
**Requirements:**
1. *Input:* An image tensor of shape $(B, 3, H, W)$.
2. *Patchification:* Divide the image into non-overlapping patches of size $P \times P$.
3. *Projection:* Map each flattened patch to an embedding vector of size $D$ (the model dimension).
4. *Position Embedding:* Add a learnable 1D Position Embedding to the sequence of patches.
5. *The "CLS" Token:* Prepend a learnable `[CLS]` (classification) token to the start of the sequence.
**Constraints:**
You must return a tensor of shape $(B, N+1, D)$, where $N$ is the number of patches.The implementation should be "PyTorch-idiomatic" (use `nn.Conv2d` with specific kernel/stride settings for the patchification step).

## Starter Code Template
```Python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 1. Use a single Conv2d layer to patchify and project
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 2. Define the learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Define the learnable Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # Your logic here
        pass
```

## Attempt #1
```Python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 1. Use a single Conv2d layer to patchify and project
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 2. Define the learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Define the learnable Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # Your logic here
        x = self.proj(x).reshape(B, -1, self.n_patches) # B, embed_dim, n_patches
        x = x.permute(0, 2, 1) # B, n_patches, embed_dim

        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) # B, n_patches+1, embed_dim
        x = x + self.pos_embed
        return x
```