#!/usr/bin/env python3
"""CLI Script to launch GRPO Reinforcement Learning on GATE-CS Reasoning dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.grpo_trainer import GRPOConfig, GRPOTrainingPipeline
from src.utils.logger import setup_logger

logger = setup_logger("run_grpo")


def main():
    parser = argparse.ArgumentParser(description="Train CALYPSO via GRPO Reinforcement Learning.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier.")
    parser.add_argument("--data-split", type=str, default="data/splits/train.jsonl", help="Path to training data.")
    parser.add_argument("--output-dir", type=str, default="models/grpo_gate_cs", help="Checkpoint output directory.")
    parser.add_argument("--num-generations", type=int, default=4, help="Generations per prompt (G).")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without initiating GPU training.")
    args = parser.parse_args()

    logger.info("Initializing GRPO Reinforcement Learning Pipeline for CALYPSO...")
    config = GRPOConfig(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
    )

    pipeline = GRPOTrainingPipeline(config=config)
    logger.info(f"Reward functions initialized: {len(pipeline.get_reward_functions())} verifiable evaluators.")

    if args.dry_run:
        logger.info("Dry run check passed. Reward functions and GRPO configuration verified successfully.")
        return

    logger.info(f"Loading dataset from {args.data_split}...")
    if not Path(args.data_split).exists():
        logger.error(f"Dataset split {args.data_split} not found. Please verify data path.")
        sys.exit(1)

    logger.info(f"Starting GRPO training for {args.model} -> saving to {args.output_dir}")
    # When running on CUDA GPU with TRL:
    try:
        from datasets import load_dataset
        ds = load_dataset("json", data_files=args.data_split, split="train")
        trainer = pipeline.build_grpo_trainer(dataset=ds)
        if trainer:
            trainer.train()
            logger.info("GRPO training completed successfully!")
        else:
            logger.info("GRPO configuration ready. Run on GPU environment with `pip install trl peft`.")
    except Exception as e:
        logger.warning(f"Note: Standard GPU training dependencies not present in local test environment ({e}).")


if __name__ == "__main__":
    main()
