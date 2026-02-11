# DL Coding Interview: Cheat Note

## PyTorch Tricks
- Patchify: [reshape](../000_SpatioTemporalWindowAttn.md) or [convolution](../002_PatchEmbeddingLayer.md) `nn.Conv2d(kernel_size=patch_size, stride=patch_size)`(if need to embed)
- Padding: [`torch.pad(x, (*W_pad, *H_pad, T_pad))`](../002_PatchEmbeddingLayer.md)
- Mask filling: [`torch.masked_fill`](../003_MultiheadSelfAttention.md)
- Retrieve lower triangular part of 2D matrix: [`torch.tril()`](../004_CausalMHSAMasking.md); can be used to stop bi-directional attention
- Cummulative product: [`torch.cumprod`](../005_DifferentiableAlphaCompositing.md)
- LoRA Implementation: [`lora_out = (x @ self.lora_A) @ self.lora_B`](../013_LoRALayerImplementation.md)
- Matrix inverse: [`torch.linalg.inv`](../014_DepthToWorldCoordinate.md)
- Unfold patches for CNN: [`patches = F.unfold(x, kernel_size=K, stride=self.stride)# B, C*K*K, num_patches`](../018_Manual2DConvolution.md)
- Matrix distance: [`torch.dist`](../032_RotationalConsistencyCheck.md)

## DL Concepts
- LoRA + MHSA: [only apply to `q` and `v`](../017_LoRAIntegratedMHSA.md)
- 2D CNN weights' shape: [$C_{out}*C_{in}*K^2$](../018_Manual2DConvolution.md); bias's shape [$C_{out}$](../018_Manual2DConvolution.md)
- Biased vs unbiased variance in [BatchNorm](../020_Manual2DBatchNorm.md):
    | Type | Formula | Use Case in BatchNorm |
    |------|---------|-----------------------|
    | Biased Variance | $\frac{1}{n} \sum (x_i - \mu)^2$ | Training Step: Used to normalize the current batch. |
    | Unbiased Variance | $$\frac{1}{n-1} \sum (x_i - \mu)^2$ | Running Stats: Used to update the moving average for Inference.|
- [Log-Sum-Trick (LST)](../022_LogSumTrick.md):
    The Log-Sum-Exp (LSE) trick is a numerical stability identity defined as:$$\text{LSE}(\mathbf{x}) = \log \sum_{i=1}^{n} \exp(x_i) = a + \log \sum_{i=1}^{n} \exp(x_i - a)$$where $a = \max(x_1, \dots, x_n)$ is used to shift the exponents into a range where the maximum value is $e^0 = 1$, effectively preventing floating-point overflow.