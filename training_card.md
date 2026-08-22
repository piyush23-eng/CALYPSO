# Training Card — GATE CS Doubt Solver (Phase 2)

## Model & Architecture Summary
- **Base Foundation Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Architecture**: Decoder-only Transformer with RoPE embeddings, SwiGLU activations, and Grouped-Query Attention (GQA).
- **Fine-Tuning Paradigm**: 4-bit Quantized Low-Rank Adaptation (QLoRA) using `peft` and `bitsandbytes`.
- **Target Modules**: All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

---

## Hyperparameter Specification & Justification

| Hyperparameter | Value | Technical Justification |
|---|---|---|
| **Quantization** | 4-bit NormalFloat (`nf4`) | Maximizes precision for normally distributed pre-trained neural network weights compared to linear INT4. |
| **Double Quantization** | `bnb_4bit_use_double_quant=True` | Quantizes the quantization constants, saving ~0.37 bits per parameter (~70MB VRAM on 1.5B). |
| **Compute Dtype** | `bfloat16` (or `float16` fallback) | Matches Qwen2.5's native training precision; avoids underflow in matrix multiplication. |
| **LoRA Rank ($r$)** | `16` | Balances representation capacity for multi-step reasoning with parameter efficiency. |
| **LoRA Alpha ($\alpha$)** | `32` | Standard scaling ratio $\alpha / r = 2.0$, ensuring gradient update magnitudes remain well-conditioned. |
| **LoRA Dropout** | `0.05` | Light dropout to prevent overfitting on formulaic answer templates. |
| **Learning Rate** | `2e-4` | Optimal empirical learning rate for 1.5B QLoRA with 4-bit base weights. |
| **LR Scheduler** | `cosine` with 3% warmup | Smooth decay avoiding sharp learning rate drops; warmup prevents early gradient explosion. |
| **Optimizer** | `paged_adamw_8bit` | Automatically offloads optimizer memory states to CPU RAM during peak memory bursts. |
| **Batch Size** | `2` per-device $\times$ `8` grad accum $= 16$ | Stabilizes gradient estimates without triggering CUDA Out-Of-Memory errors. |
| **Context Length** | `1024` tokens | Accommodates complete problem statements and multi-paragraph CoT derivations. |
| **Checkpoint Criterion** | Lowest `eval_loss` on validation split | Selects checkpoint by empirical generalization on GATE 2021–2022 validation problems. |

---

## Memory & Compute Profile

```
┌────────────────────────────────────────────────────────────┐
│ Total VRAM Budget on T4 (16 GB): ~5.8 GB                   │
├───────────────────────────────┬────────────────────────────┤
│ Component                     │ VRAM Consumption           │
├───────────────────────────────┼────────────────────────────┤
│ 4-bit Base Model Weights      │ ~1.1 GB                    │
│ LoRA Trainable Parameters     │ ~36 MB (18.4M params, ~1%) │
│ Activation Memory (Grad Check)│ ~3.2 GB                    │
│ Optimizer States (Paged 8-bit)│ ~1.5 GB                    │
└───────────────────────────────┴────────────────────────────┘
```

---

## How to Run Training

### Option A: Free-Tier Google Colab / Kaggle (Recommended)
1. Open [`notebooks/gate_qlora_training.ipynb`](file:///Users/baleshwarpandit/.gemini/antigravity/scratch/gate-cs-doubt-solver/notebooks/gate_qlora_training.ipynb) in Colab.
2. Set runtime to **GPU (T4 or L4)**.
3. Run all cells sequentially. Training will log loss curves to Weights & Biases and save the best adapter to `models/gate_qwen_1.5b_lora`.

### Option B: Local CLI Execution (with CUDA GPU)
```bash
python scripts/run_training.py \
    --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset-dir "data/splits" \
    --epochs 3 \
    --lr 2e-4 \
    --batch-size 2 \
    --grad-accum 8 \
    --output-dir "models/checkpoints" \
    --adapter-dir "models/gate_qwen_1.5b_lora"
```

### Option C: Merging LoRA Weights into Standalone Checkpoint
```bash
python scripts/export_adapter.py \
    --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --adapter-dir "models/gate_qwen_1.5b_lora" \
    --output-dir "models/gate_qwen_1.5b_merged"
```
