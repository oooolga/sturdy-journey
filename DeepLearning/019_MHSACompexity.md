# Runtime Complexity of Multi-Head Self-Attention (MHSA)

The runtime complexity of Multi-Head Self-Attention (MHSA) is typically cited as $O(n^2 \cdot d)$, where $n$ is the sequence length and $d$ is the embedding dimension.

However, in a technical interview, a precise answer requires breaking down the individual matrix operations that occur within the block.

1. **Breakdown of Operations**
    The total complexity is the sum of four main steps:
    | Step             | Operation                 | Complexity |
    |------------------|---------------------------|------------|
    | 1-Projections   | Linearly transform input $X \in \mathbb{R}^{n \times d}$ into $Q, K, V$ using weights $W \in \mathbb{R}^{d \times d}$. | $O(n \cdot d^2)$ |
    | 2-Attention Scores | Compute $Q K^T$ to get the affinity matrix of size $(n, n)$. | $O(n^2 \cdot d)$|
    | 3-Weighted Sum  | Multiply the attention weights $(n, n)$ by the values $V \in \mathbb{R}^{n \times d}$. |$O(n^2 \cdot d)$|
    | 4-Output Projection | Project the concatenated head outputs back to dimension $d$. | $O(n \cdot d^2)$ |


    **Total Complexity:** $O(n^2 d + n d^2)$
2. $O(n^2 d)$ vs. $O(n d^2)$
    The "quadratic bottleneck" everyone refers to is the $n^2$ term.
    - When $n > d$: (Long sequences, like in document analysis or high-res images), the $O(n^2 d)$ term dominates. This is why Transformers struggle with very long contexts.
    - When $d > n$: (Short sequences with massive hidden dimensions), the projection cost $O(n d^2)$ actually dominates.
3. Does the number of heads ($h$) change the complexity?
   No. In MHSA, we split the $d$ dimensions into $h$ heads, so each head has a dimension of $d_h = d/h$. 
    - Complexity per head: $O(n^2 \cdot \frac{d}{h})$
    - Total for $h$ heads: $h \cdot O(n^2 \cdot \frac{d}{h}) = \mathbf{O(n^2 d)}$
    - The multi-head structure is designed for representational power (attending to different subspaces), not for reducing computational complexity.