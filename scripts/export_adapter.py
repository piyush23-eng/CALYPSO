"""Merges trained LoRA adapter weights with base Qwen model into a standalone fp16 checkpoint."""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("export_adapter")


def merge_and_export(
    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_dir: str = "models/gate_qwen_1.5b_lora",
    output_dir: str = "models/gate_qwen_1.5b_merged",
    torch_dtype: str = "bfloat16",
):
    adapter_path = Path(adapter_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {base_model_id} in {torch_dtype}...")
    dtype = getattr(torch, torch_dtype, torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged standalone model to {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)

    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Export complete! Merged model ready at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="models/gate_qwen_1.5b_lora")
    parser.add_argument("--output-dir", type=str, default="models/gate_qwen_1.5b_merged")
    args = parser.parse_args()

    merge_and_export(
        base_model_id=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
    )
