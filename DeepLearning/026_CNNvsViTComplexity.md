# The Challenge: Scaling to High Resolution
Assume you have an image of size $H \times W$ (total pixels $N = H \cdot W$). You want to compare a standard Convolutional Layer (CNN) with a Self-Attention Layer (ViT) in terms of computational complexity (FLOPs).

**The Scenario:**
- *CNN:* A convolution with kernel size $K \times K$, input channels $C$, and output channels $C$.
- *ViT:* A self-attention layer with embedding dimension $C$ and $N$ patches (where each patch is 1x1 for a direct pixel comparison).

## The Questions:
1. **Big-O Complexity vs. Resolution ($N$):** Express the computational complexity of a single forward pass for both the CNN and the ViT self-attention layer in terms of $N$, $C$, and $K$. How does the complexity scale as the image resolution increases (e.g., moving from 256p to 4K)?
2. **The "Crossover" Point:** In terms of $N$ and $K$, at what point does the Vision Transformer theoretically become more computationally expensive than the CNN?
3. **The 4D Spatio-Temporal Problem:** In your 4D reconstruction project, you have a sequence of $T$ frames. If you use 3D Convolutions ($T, K, K$), how does the complexity scale? If you use Full Spatio-Temporal Self-Attention (where every pixel in every frame attends to every other pixel in all frames), what is the complexity in terms of $T$ and $N$?

## Answer:
1. **Big-O Complexity vs. Resolution ($N$):** 
    > - CNN: $O((N-K)K^2C^2)$
    > - ViT: $O(NC^2+N^2C)$
    > - ViT becomes more expensive as resolution increases. CNN is $O(N)$ (Linear scaling with resolution).ViT is $O(N^2)$ (Quadratic scaling with resolution).
2. **The "Crossover" Point:**
    > When $K^2C < N$, ViT is computationally expensive than CNN.
3. **The 4D Spatio-Temporal Problem:** 
    > - 3D Convolutions ($T, K, K$): $O(N \times (T\times K^2) \times  C^2)$
    > - Full Spatio-Temporal Self-Attention: $O((N\times T)C^2 + (N\times T)^2 C)$