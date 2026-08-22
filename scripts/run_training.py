"""CLI entrypoint to trigger SFT QLoRA Fine-Tuning."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import QLoRAConfig, TrainingConfig
from src.training.trainer import GATEModelTrainer
from src.utils.logger import setup_logger

logger = setup_logger("run_training")


def main():
    parser = argparse.ArgumentParser(description="GATE-CS LLM QLoRA Fine-Tuning")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    parser.add_argument("--dataset-dir", type=str, default="data/splits", help="Path to train/val splits")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Checkpoint output directory")
    parser.add_argument("--adapter-dir", type=str, default="models/gate_qwen_1.5b_lora", help="Final adapter directory")
    args = parser.parse_args()

    qlora_cfg = QLoRAConfig(
        base_model_id=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    train_cfg = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        final_adapter_dir=args.adapter_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )

    logger.info("Initializing GATE CS Model Trainer...")
    trainer = GATEModelTrainer(qlora_cfg=qlora_cfg, train_cfg=train_cfg)
    trainer.train_and_save()


if __name__ == "__main__":
    main()
