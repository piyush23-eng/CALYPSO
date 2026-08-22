#!/usr/bin/env bash
set -e

# GGUF Conversion and Quantization Script using llama.cpp
# Model: Qwen2.5-1.5B Fine-Tuned Checkpoint -> GGUF Q4_K_M

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED_DIR="${PROJECT_ROOT}/models/gate_qwen_1.5b_merged"
GGUF_DIR="${PROJECT_ROOT}/models/gguf"
LLAMA_CPP_DIR="${PROJECT_ROOT}/llama.cpp"

mkdir -p "${GGUF_DIR}"

echo "=========================================================="
echo "GATE-CS LLM: GGUF Quantization Pipeline (Q4_K_M)"
echo "=========================================================="

# Step 1: Clone llama.cpp if not present
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo "Cloning llama.cpp repository..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_CPP_DIR}"
    echo "Building llama.cpp binaries..."
    cmake -B "${LLAMA_CPP_DIR}/build" "${LLAMA_CPP_DIR}"
    cmake --build "${LLAMA_CPP_DIR}/build" --config Release -j 4
fi

# Step 2: Convert HuggingFace checkpoint to GGUF (FP16)
echo "Converting HuggingFace model to GGUF (fp16)..."
python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "${MERGED_DIR}" \
    --outfile "${GGUF_DIR}/gate-qwen-1.5b-f16.gguf" \
    --outtype f16

# Step 3: Quantize FP16 to Q4_K_M
echo "Quantizing FP16 GGUF to Q4_K_M..."
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
    "${GGUF_DIR}/gate-qwen-1.5b-f16.gguf" \
    "${GGUF_DIR}/gate-qwen-1.5b-q4_k_m.gguf" \
    Q4_K_M

echo "=========================================================="
echo "Quantization Complete!"
echo "GGUF Binary: ${GGUF_DIR}/gate-qwen-1.5b-q4_k_m.gguf"
echo "=========================================================="
ls -lh "${GGUF_DIR}"
