# The Challenge: LoRA vs. Full Fine-Tuning Complexity

Suppose you have a standard Transformer linear layer with an input dimension $d_{in}$ and an output dimension $d_{out}$. You are processing a batch of $N$ sequences, each with a length of $L$.

**The Scenario:** You apply LoRA to this layer with a rank $r$.
1. *Base Weight $W$:* $(d_{out}, d_{in})$
2. *LoRA A:* $(r, d_{in})$
3. *LoRA B:* $(d_{out}, r)$
4. *Input $X$:* $(N, L, d_{in})$

**The Answers:**
1. *Training Memory Complexity (Weights):* What is the ratio of trainable parameters in a LoRA-adapted layer compared to a full fine-tuning scenario? Express this in terms of $d_{in}$, $d_{out}$, and $r$.
2. *Forward Pass Computational Complexity:* Compare the FLOPs required for:
    1. A standard full fine-tuning forward pass: $Y = XW^T$.
    2. A LoRA forward pass implemented as: $Y = XW^T + (X A^T) B^T$.3.
3. The Optimization Insight:In terms of $O(\cdot)$, why is it computationally "cheaper" to compute $(XA^T)B^T$ rather than first computing the updated weight matrix $\Delta W = B A$ and then multiplying by $X$?

**The Questions:**
1. *Training Memory Complexity (Weights):* What is the ratio of trainable parameters in a LoRA-adapted layer compared to a full fine-tuning scenario? Express this in terms of $d_{in}$, $d_{out}$, and $r$.
    > Full fine-tuning: $d_{in} * d_{out}$
    >
    > LoRA: $d_{in} * r + r * d_{out}$
    > 
    > $\sigma = \frac{r}{d_{in}}+\frac{r}{d_{out}}$
2. *Forward Pass Computational Complexity:* Compare the FLOPs required for:
    1. A standard full fine-tuning forward pass: $Y = XW^T$.
    2. A LoRA forward pass implemented as: $Y = XW^T + (X A^T) B^T$.

    > ${FLOPs}_{fine-tuning} = 2* N * L * d_{in} * d_{out}$
    > 
    > ${FLOPs}_{LoRA} = 2 * N * L * d_{in} * d_{out} + 2 * N * (d_{in} + d_{out}) * r $
3. The Optimization Insight:In terms of $O(\cdot)$, why is it computationally "cheaper" to compute $(XA^T)B^T$ rather than first computing the updated weight matrix $\Delta W = B A$ and then multiplying by $X$?
    > Because $r$ is much smaller.
    >
    > If we compute $Y_{adapter} = (X A^T) B^T$:
    > - $X A^T$ takes $O(N \cdot L \cdot d_{in} \cdot r)$.
    > - Result $\times B^T$ takes $O(N \cdot L \cdot r \cdot d_{out})$.
    > - Total: $O(N \cdot L \cdot r \cdot (d_{in} + d_{out}))$
    > 
    > If we compute $\Delta W = B A$ first:
    > - $B A$ takes $O(d_{out} \cdot r \cdot d_{in})$.
    > - $X \Delta W$ takes $O(N \cdot L \cdot d_{in} \cdot d_{out})$.
    > - Total: $O(d_{in} d_{out} (r + NL))$