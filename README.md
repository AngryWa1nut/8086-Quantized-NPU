# 8-bit Quantized Neural Inference Accelerator in 8086 Assembly

An optimized 16-bit Intel 8086 assembly-level hardware simulation of a Quantized Neural Network forward pass layer. This project implements a **Fully Connected (Dense) Layer** featuring **Kernel Fusion** of matrix-vector multiply-accumulate operations, **ReLU activation**, and **INT8 saturating quantization (Clamping)** under strict hardware resource constraints.

## Key Technical Novelties & Architecture

Rather than executing neural network operations sequentially—which causes memory I/O bottlenecks—this accelerator implements **Algorithmic Kernel Fusion**. It processes linear transformation, non-linear activation, and hardware-level precision clamping inside a single unified pipeline loop before the data settles into memory.

### 1. Neural Processing Pipeline
The core accelerator computes the following mathematical layer forward pass:

$$Y = \text{Clamp}\Big(\text{ReLU}(X \cdot W)\Big)$$

* **Linear Transformation ($X \cdot W$):** Performs 16-bit signed integer Multiply-Accumulate via `IMUL DL` and `ADD` over a $4 \times 7$ input activation matrix and a $7 \times 5$ weight matrix.
* **Non-linear Activation (ReLU):** Implements dynamic sparsity mapping ($$\text{Max}(0, x)$$). Negative intermediate signals are pruned down to `0` instantly, eliminating irrelevant noise.
* **INT8 Saturating Quantization (Clamping):** Compresses the 16-bit wide accumulator space down to a deployable INT8 scale ($0 \le Y \le 127$). Strongly activated features exceeding the upper bound are safely saturated at `127` to prevent integer overflow.

## Evaluation & Simulation Results

When executing the pipeline in EMU8086, the raw intermediate outputs (accumulating up to large numbers like `5634` or tumbling down to negative values like `-5118`) are successfully bounded on the fly. 

The final output matrix (`rm_r0` to `rm_r3`) yields quantized activations precisely restricted within the legal INT8 bounds:
```text
RM_R0: 0
RM_R1: 127
RM_R2: 127
RM_R3: 0
