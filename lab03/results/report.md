# Lab 3: Architectures and Their Costs (Summary Report)

## 1. Goal
Evaluate three PyTorch architectures—MLP, CNN, and Patch Attention—on Fashion-MNIST to analyze the trade-offs between parameter efficiency, FLOP count, inference latency ($B=1$ vs $B=128$), and test accuracy.

## 2. Results Summary

### Performance & Overhead Matrix

| Model | Parameters | THOP FLOPs ($B=1$) | Latency ($B=1$) | Latency ($B=128$) | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | 235,402 | 470,016 | 0.042 ms | 0.195 ms | 86.42% |
| **CNN** | 20,490 | 2,422,016 | 0.125 ms | 0.840 ms | **89.15%** |
| **Attention** | 37,254 | 1,146,880 | 0.380 ms | 1.850 ms | 85.30% |

### FLOP Verification (Hand-Calculated vs Profiler)
* **MLP Layer 1 ($784 \rightarrow 256$):** Hand: $2 \times 784 \times 256 = 401,408$ | THOP: $401,408$ (Exact)
* **CNN Conv 2 ($16 \rightarrow 32$):** Hand: $2 \times 3^2 \times 16 \times 32 \times 14 \times 14 = 1,806,336$ | THOP: $1,806,336$ (Exact)
* **Attention Encoder:** Hand: $\approx 1.32\text{ MFLOPs}$ | THOP: $1.15\text{ MFLOPs}$ (Gap: THOP excludes Softmax, LayerNorm, and element-wise additions).

---

## 3. Key Conclusions

1. **Parameter Efficiency:** The **CNN** is the most cost-effective architecture. Due to spatial inductive bias and weight sharing, it achieved the highest accuracy (**89.15%**) using under $10\%$ of the MLP's parameter count.
2. **Memory Bandwidth & Latency:** **Attention** exhibited the highest latency ($1.85\text{ ms}$ at $B=128$) despite running fewer total FLOPs than the CNN. Generating dynamic attention matrices ($QK^T$) causes non-contiguous memory access patterns, making it memory-bandwidth bound.
3. **Profiler Discrepancies:** Profilers like `thop` strictly count Multiply-Accumulate (MAC) operations, ignoring non-linear activations, Softmax, and normalization overheads in Transformer blocks.