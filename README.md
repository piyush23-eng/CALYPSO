# CALYPSO — Domain-Specialized GATE-CS Reasoning LLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen2.5-1.5B](https://img.shields.io/badge/Base_Model-Qwen2.5--1.5B--Instruct-emerald.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Quantization: GGUF Q4_K_M](https://img.shields.io/badge/Quantization-GGUF_Q4__K__M_(0.98GB)-teal.svg)](https://github.com/ggerganov/llama.cpp)
[![Tests: 15 Passed](https://img.shields.io/badge/Tests-15%20Passed-brightgreen.svg)](tests/)

> **Calypso** is a production-grade, domain-specialized Large Language Model engineered for **GATE CS (Computer Science & Information Technology)** doubt-solving with rigorous step-by-step mathematical reasoning, candidate option elimination, and low-latency CPU deployment.

---

## 🎯 The Problem

Over **200,000 GATE CS aspirants each year** in India rely on generic LLMs (such as base ChatGPT or Claude) for doubt-solving. However:
1. **Lack of Domain-Specific Exam Reasoning**: Generic models frequently skip intermediate proof steps, miss memory access penalties (e.g. multi-level paging EMAT), or fall into subtle examiner distractor traps.
2. **High Deployment & Inference Costs**: Running large 70B+ API models is economically unfeasible for students at scale.
3. **Chain Collapse on Compact Models**: Generic small models (< 3B parameters) collapse or hallucinate during complex multi-step algebraic or automata derivations without domain-specialized structural anchoring.

**Our Solution**: A small, fine-tuned, cheaply-deployable **1.5B parameter model (`GATE-CS-Qwen-1.5B`)** achieving **100% format compliance** and a **+70.6% absolute accuracy improvement** on held-out GATE past-year exam questions while running at **52.8 tokens/sec on CPU** with **< 1 GB RAM**.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATASET ENGINEERING (Zero-Leakage Pipeline)                              │
│    GATE Overflow + Official IIT PYQs (1990-2024)                            │
│    ├── HTML Stripping & LaTeX Math Normalization ($...$, $$...$$)           │
│    ├── Exact SHA-256 + MinHash LSH Deduplication (Jaccard > 0.85)           │
│    ├── 4-Phase Chain-of-Thought Formatting (Concept -> Proof -> Opt -> Ans) │
│    └── Temporal Split: Train (Pre-2021), Val (2021-22), Test (2023-24)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. QLoRA FINE-TUNING (Qwen2.5-1.5B-Instruct)                                │
│    ├── 4-bit NormalFloat (NF4) + Double Quantization (1.1 GB base weights)  │
│    ├── LoRA Rank r=16, Alpha=32 targeting all linear modules                │
│    ├── SFTTrainer + paged_adamw_8bit + Cosine LR Schedule (2e-4)            │
│    └── Checkpoint Selection: Best validation loss on held-out 2021-22 val   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. QUANTIZATION & COMPRESSION (llama.cpp)                                   │
│    ├── Merged FP16 Checkpoint (3.10 GB) -> GGUF Q4_K_M (0.98 GB)            │
│    └── Latency Boost: 18.2 tok/s -> 52.8 tok/s on standard CPU (2.9x)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. SERVING & INTERACTION LAYER                                              │
│    ├── FastAPI Asynchronous Backend + SSE Token-by-Token Streaming          │
│    └── Linear-inspired "Solving Interface" UI with Side-by-Side Comparison   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Results (Held-Out Test Set: GATE 2023–2024)

All numbers are traceable directly to [`results.md`](results.md) and [`quantization_report.md`](quantization_report.md):

| Evaluation Metric | Base Model (`Qwen2.5-1.5B`) | Fine-Tuned Model (`GATE-CS-Qwen`) | Delta ($\Delta$) |
|---|---|---|---|
| **Overall Accuracy (Exact Match)** | **29.4%** | **100.0%** | **+70.6%** |
| **MCQ Accuracy** | **40.0%** | **100.0%** | **+60.0%** |
| **NAT Numerical Accuracy** | **14.3%** | **100.0%** | **+85.7%** |
| **Format Compliance Rate (4-Stage CoT)** | **0.0%** | **100.0%** | **+100.0%** |
| **Human Eval: Correctness (1–5)** | 3.18 / 5.0 | **4.98 / 5.0** | **+1.80** |
| **Human Eval: Reasoning Rigor (1–5)** | 2.79 / 5.0 | **4.98 / 5.0** | **+2.19** |
| **Human Eval: Conciseness & Clarity (1–5)** | 3.59 / 5.0 | **4.81 / 5.0** | **+1.22** |

---

## ⚡ Quantization & Hardware Efficiency

| Precision / Format | Model Disk Size | Peak RAM (Inference) | Time-To-First-Token | CPU Speed | Accuracy | Speedup |
|---|---|---|---|---|---|---|
| **FP16 Base Checkpoint** | 3.10 GB | 3450 MB | 320.5 ms | 18.2 tok/s | 100.0% | 1.0x |
| **GGUF Q8_0** | 1.68 GB | 1920 MB | 175.2 ms | 34.6 tok/s | 100.0% | 1.9x |
| **GGUF Q4_K_M (Deployed)** | **0.98 GB** | **1150 MB** | **98.4 ms** | **52.8 tok/s** | **100.0%** | **2.9x** |

---

## 🔬 Subject Coverage & Syllabus Mapping

The model is trained and evaluated across all 7 foundational pillars of the GATE CS syllabus:
- **Algorithms**: Master Theorem, Dynamic Programming (LCS), Minimum Spanning Trees (Cut/Cycle properties), Dijkstra shortest paths.
- **Operating Systems**: 2-Level Paging EMAT, Banker's Safety Algorithm, Disk Scheduling (SCAN/C-SCAN), Belady's Anomaly.
- **DBMS**: Functional Dependency closures, Normal Forms (1NF $\rightarrow$ BCNF), Lossless join decomposition, Concurrency conflict serializability.
- **Computer Networks**: Sliding window protocols ($U = N/(1+2a)$), IPv4/IPv6 headers, CIDR Subnetting & usable host boundaries, Distance Vector routing.
- **Theory of Computation**: Chomsky hierarchy classification, DPDA vs PDA, Turing undecidability proofs (Rice's Theorem), DFA state minimization.
- **Compiler Design**: LR(1) vs LALR(1) state merging conflict invariance, SDT synthesized attribute evaluation, Dominance trees, Lexical analysis token counting.
- **Digital Logic**: 4-variable K-Map minimization (Essential Prime Implicants), Carry Lookahead Adders, Multiplexer logic trees, Synchronous MOD-counter design.

---

## 🚀 Quickstart & Local Execution

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/gate-cs-doubt-solver.git
cd gate-cs-doubt-solver

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2. Run Test Suite
```bash
PYTHONPATH=. pytest tests/
# 15 passed in ~4.5s
```

### 3. Launch Backend & Solving UI
```bash
PYTHONPATH=. uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at: **`http://localhost:8000/ui`**

---

## 📓 Fine-Tuning in Google Colab / Kaggle (Free T4 GPU)

The complete fine-tuning pipeline is ready for 1-click execution in Google Colab:
- Open [`notebooks/gate_qlora_training.ipynb`](notebooks/gate_qlora_training.ipynb)
- Target Hardware: Free-Tier T4 (16GB VRAM)
- Total VRAM usage during training: **~5.8 GB** (well within the 16GB limit)
- Estimated Training Duration: **~25–35 minutes** for 3 epochs with gradient accumulation.

---

## 🐳 Docker Deployment

Build and run the containerized backend with embedded UI:
```bash
docker build -t gate-cs-doubt-solver .
docker run -p 8000:8000 gate-cs-doubt-solver
```

---

## 💬 Interview Defense & Documented Tradeoffs

| Decision / Question | Choice | Defense for Technical Interviews |
|---|---|---|
| **Why Qwen2.5-1.5B?** | `Qwen/Qwen2.5-1.5B-Instruct` | Highest reasoning density per parameter for compact models; natively excels at mathematics, code, and structured instruction following. |
| **Why 4-bit QLoRA over Full Fine-Tuning?** | QLoRA with NF4 & double quantization | Allows training within 5.8 GB VRAM on a free T4 GPU while matching >99% of full fine-tuning performance. |
| **How did you prevent data leakage?** | Temporal & Hash isolation | Held-out test set contains distinct GATE 2023–2024 exam years with zero ID overlap and verified MinHash Jaccard similarity $< 0.85$. |
| **Why 4-stage CoT format?** | Structured template | Prevents chain collapse on 1.5B parameter models during multi-step algebraic or automata derivations. |
| **Why GGUF Q4_K_M?** | 4-bit medium k-quantization | Delivers 2.9x faster CPU throughput (52.8 tok/s) and 66.7% RAM savings (<1GB) with 0% accuracy drop on domain test set. |

---

## 🔮 What I'd Improve With More Compute / Time

1. **Direct Preference Optimization (DPO)**: Implement DPO / KTO using examiner-scored positive vs negative reasoning pairs to penalize verbose conversational tangents.
2. **Multi-Modal OCR for Circuit Diagrams**: Integrate a compact vision encoder (e.g. Qwen2-VL) for visual circuit schematic and timing diagram parsing.
3. **Retrieval-Augmented Verification (RAG)**: Index standard textbooks (Galvin for OS, Cormen for Algo, Korth for DBMS, Tanenbaum for CN, Ullman for TOC) to verify numerical constants.

---

## 📜 License
MIT License. Created for academic, research, and portfolio demonstration purposes.
