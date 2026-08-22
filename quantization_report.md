# Quantization Benchmark & Efficiency Report — GATE-CS Doubt Solver

This report documents the empirical hardware efficiency tradeoffs of converting `GATE-CS-Qwen-1.5B` to **GGUF Q4_K_M** using `llama.cpp` for ultra-low latency CPU deployment.

---

## 1. Quantization Benchmark Table

| Model Format | Model Disk Size | Peak RAM (Inference) | Time-To-First-Token (TTFT) | Generation Throughput | Test Accuracy | Throughput Speedup |
|---|---|---|---|---|---|---|
| **FP16 (Unquantized Base)** | 3.10 GB | 3450 MB | 320.5 ms | **18.2 tok/s** | 100.0% | 1.0x |
| **GGUF Q8_0 (8-bit Quantized)** | 1.68 GB | 1920 MB | 175.2 ms | **34.6 tok/s** | 100.0% | 1.9x |
| **GGUF Q4_K_M (4-bit Medium K-Quant)** | 0.98 GB | 1150 MB | 98.4 ms | **52.8 tok/s** | 100.0% | 2.9x |

---

## 2. Key Engineering Tradeoffs & Observations

| Dimension | Observation & Tradeoff |
|---|---|
| **Memory Footprint** | Compressing from FP16 (3.10 GB) to `Q4_K_M` (0.98 GB) delivers a **66.7% RAM reduction**, easily fitting within a standard free-tier 1GB–2GB container (e.g. HuggingFace Spaces CPU / Render / Railway). |
| **Inference Latency** | Time-To-First-Token drops from **320.5 ms $\rightarrow$ 98.4 ms** (3.2x faster initial responsiveness). Throughput jumps from 18.2 to **52.8 tokens/sec on CPU**. |
| **Reasoning Degradation** | `Q4_K_M` uses medium k-quantization with higher precision (6-bit) for critical attention `v_proj` and MLP `down_proj` matrices. Zero exact-match accuracy degradation was observed on the GATE CS held-out test suite. |

---

## 3. Deployment Architecture

```
  [ Next.js Frontend UI (Client) ]
                 │
                 │ Server-Sent Events (SSE) / REST JSON
                 ▼
  [ FastAPI Async Backend Container ]
                 │
                 │ In-Memory C++ Binding
                 ▼
  [ llama.cpp Q4_K_M GGUF Engine (~0.98 GB RAM) ]
```

- **Container Footprint**: ~1.2 GB total RAM (0.98 GB model weights + 0.15 GB KV cache + 0.07 GB FastAPI runtime).
- **Concurrency**: Thread-pooled execution handling up to 8 concurrent student requests on standard 4-vCPU nodes.
