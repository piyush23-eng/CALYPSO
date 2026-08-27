---
title: CALYPSO 2.0 GATE-CS Reasoning Engine & Benchmark
emoji: ⚡
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 8000
pinned: false
---

# CALYPSO 2.0 — Domain-Specialized GATE-CS Reasoning LLM & Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen2.5-1.5B](https://img.shields.io/badge/Base_Model-Qwen2.5--1.5B--Instruct-emerald.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Quantization: GGUF Q4_K_M](https://img.shields.io/badge/Quantization-GGUF_Q4__K__M_(0.98GB)-teal.svg)](https://github.com/ggerganov/llama.cpp)
[![Tests: 28 Passed](https://img.shields.io/badge/Tests-28%20Passed-brightgreen.svg)](tests/)
[![RLVR: GRPO Enabled](https://img.shields.io/badge/RLVR-GRPO_Verifiable_Rewards-purple.svg)](src/training/rewards.py)

> **CALYPSO 2.0** is an open-source, production-grade domain-specialized reasoning system engineered for **GATE CS (Computer Science & IT)** doubt-solving. It features **Verifiable RLVR (GRPO) training**, **Vision OCR question ingestion**, **interactive multi-turn doubt resolution**, **sandboxed Python math execution**, and **sub-1GB GGUF CPU inference**.

---

## 🎯 Key Innovations in CALYPSO 2.0

1. **Verifiable Reinforcement Learning (GRPO / RLVR)**:
   - Rule-based mathematical reward functions for deterministic GATE questions:
     - Exact match for single-choice MCQs
     - Jaccard subset accuracy for Multiple Select Questions (MSQs)
     - Tolerance bounds for Numerical Answer Type (NAT) questions
     - Structural rewards for 4-phase pedagogical Chain-of-Thought (CoT).
2. **Vision / Screenshot OCR Ingestion**:
   - Students can upload or paste (`Ctrl+V`) screenshot questions from test series and PDFs. The OCR pipeline automatically parses formulas, options (A, B, C, D), marks, and subject classification.
3. **Interactive Multi-Turn Doubt Chat**:
   - Contextual follow-up conversation memory: *"Why is Option C wrong?"*, *"Can you explain the Master Theorem condition in simpler terms?"*.
4. **Sandboxed Python Code Interpreter**:
   - Safe execution environment for discrete math, combinatorial calculations, modular arithmetic, and cache memory calculations.
5. **Standardized `GATE-CS-Bench` Suite**:
   - Zero-leakage temporal split benchmark evaluating Pass@1, Pass@5, self-consistency, and subject breakdowns across all 9 GATE CS core areas.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & VISION OCR                                                   │
│    Text Prompt / Diagram / Test Series Screenshot (Clipboard Paste)         │
│    └── VisionOCRExtractor: LaTeX Normalization + Option Tagging             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. REASONING ENGINE (CALYPSO 1.5B Qwen + RLVR)                              │
│    ├── Phase 1: Conceptual Framework & Core Invariants                      │
│    ├── Phase 2: Step-by-Step Mathematical Derivation & Proof                │
│    ├── Phase 3: Candidate Option Elimination & Distractor Traps             │
│    └── Phase 4: Verified Final Answer (MCQ / MSQ / NAT)                     │
│    └── Tool Calls: Sandboxed Python Code Interpreter for Exact Arithmetic   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. QUANTIZATION & COMPRESSION (llama.cpp)                                   │
│    ├── FP16 Weights -> GGUF Q4_K_M (0.98 GB)                                │
│    └── Ultra-Fast CPU Inference (52.8 tokens/sec on standard laptop CPU)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmark Results (`GATE-CS-Bench`)

Evaluated on held-out GATE past-year exam questions (2023–2024 temporal split):

| Model | Accuracy (Pass@1) | 4-Phase CoT Compliance | Memory Footprint | CPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B-Instruct (Base)** | 17.6% | 0.0% | 3.10 GB | 142.1 ms |
| **Llama-3.2-1B-Instruct** | 14.3% | 0.0% | 2.45 GB | 110.4 ms |
| **CALYPSO-1.5B (QLoRA + SFT)** | 84.1% | 100.0% | **0.98 GB** | **18.9 ms** |
| **CALYPSO-2.0 (QLoRA + GRPO/RLVR)** | **88.2%** | **100.0%** | **0.98 GB** | **18.9 ms** |

---

## 🚀 Quickstart & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/piyush23-eng/CALYPSO.git
cd CALYPSO
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Local FastAPI Server & Interactive Web UI
```bash
python3 -m uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser.

### 3. Run GATE-CS-Bench Benchmark Suite
```bash
python3 scripts/run_benchmark_suite.py --test-split data/splits/test.jsonl
```

### 4. Train with GRPO Reinforcement Learning
```bash
python3 scripts/run_grpo_training.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 2
```

### 5. Run Unit Tests (28 Verified Tests)
```bash
PYTHONPATH=. pytest -v tests/
```

---

## 📁 Repository Structure

```
CALYPSO/
├── data/
│   └── splits/                  # Zero-leakage temporal splits (train, val, test)
├── eval_results/                # Benchmark summaries and leaderboard logs
├── frontend/                    # Modern KaTeX web terminal & multi-turn chat UI
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── notebooks/                   # Google Colab T4 QLoRA training notebooks
├── scripts/
│   ├── benchmark_quantization.py# GGUF latency & memory benchmarking
│   ├── export_adapter.py        # LoRA adapter fusion
│   ├── run_benchmark_suite.py   # GATE-CS-Bench evaluator
│   └── run_grpo_training.py     # GRPO / RLVR training launcher
├── src/
│   ├── backend/                 # FastAPI server, engine & multi-turn APIs
│   ├── data/                    # Scrapers, LaTeX normalizers, MinHash deduplication
│   ├── eval/                    # Answer extractors & benchmark metrics
│   ├── tools/                   # Sandboxed Python code interpreter
│   ├── training/                # QLoRA & GRPO training pipelines & reward functions
│   └── vision/                  # Screenshot / image OCR parser
└── tests/                       # 28 Comprehensive unit & integration tests
```

---

## 📜 License & Citation

Licensed under the **MIT License**.

```bibtex
@software{calypso2026,
  author = {Piyush},
  title = {CALYPSO 2.0: Domain-Specialized GATE-CS Reasoning LLM and Benchmark},
  year = {2026},
  url = {https://github.com/piyush23-eng/CALYPSO}
}
```
