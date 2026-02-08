# The Challenge: InfoNCE (Information Noise-Contrastive Estimation) loss

**The Task:** Implement a ContrastiveLoss module that takes two batches of embeddings, `z_i` and `z_j` (where `z_i[k]` and `z_j[k]` are two different augmented views of the same image), and computes the loss.

**Requirements:**
1. *Cosine Similarity:* Use cosine similarity as the distance metric.
2. *Temperature Scaling:* Incorporate a temperature parameter $\tau$ to control the "sharpness" of the distribution.
3. *The Matrix Trick:* To be efficient, you must compute a single large similarity matrix $(2N \times 2N)$ that represents all possible pairs in the batch.
4. *The Mask:* You must mask out the self-similarities (the diagonal) so the model doesn't try to pull an embedding toward itself.
5. *The Objective:* For each sample, the "positive" is its augmented pair, and all other $2N-2$ samples in the batch are "negatives."
## Implementation Template
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: (batch_size, embed_dim)
        z_j: (batch_size, embed_dim)
        """
        batch_size = z_i.shape[0]
        
        # 1. Normalize the embeddings (Crucial for cosine similarity)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 2. Concatenate to get a total of 2*batch_size embeddings
        representations = torch.cat([z_i, z_j], dim=0) # (2N, D)

        # 3. Compute Similarity Matrix (2N x 2N)
        # YOUR TASK: Use matmul to get the pairwise similarities
        sim_matrix = None 

        # 4. Create a mask to remove self-similarity (diagonal)
        # 5. Identify which indices are the "positives" (e.g., i and i+N)
        # 6. Compute the CrossEntropy-style loss
        
        pass
```
## Gemini Solution
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        # 1. Normalize and Concatenate
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0) # (2N, D)

        # 2. Similarity Matrix (2N x 2N)
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        # 3. Mask out self-similarity
        # We use a large negative number instead of -inf to avoid NaN in some edge cases
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # 4. Target Indices (The "Positives")
        # For row i (0 to N-1), the positive is i + N
        # For row i (N to 2N-1), the positive is i - N
        targets = torch.arange(2 * batch_size, device=z_i.device)
        targets = (targets + batch_size) % (2 * batch_size)

        # 5. Stable Cross Entropy
        # This treats each row as a classification problem:
        # "Which of the 2N-1 other samples is my augmented pair?"
        loss = F.cross_entropy(sim_matrix, targets)

        return loss
```