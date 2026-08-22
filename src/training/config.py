"""Training configuration dataclass and validation for GATE CS QLoRA Fine-Tuning."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class QLoRAConfig:
    """LoRA and Quantization Hyperparameters."""
    # Model parameters
    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA Adapter parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """SFT Training Arguments and Optimization Settings."""
    # Paths
    dataset_dir: str = "data/splits"
    output_dir: str = "models/checkpoints"
    final_adapter_dir: str = "models/gate_qwen_1.5b_lora"
    merged_model_dir: str = "models/gate_qwen_1.5b_merged"
    logging_dir: str = "logs"

    # Training Schedule & Batching
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"

    # Context & Tokenization
    max_seq_length: int = 1024
    packing: bool = False

    # Checkpoint & Evaluation Strategy
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Hardware & Performance
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True  # Preferred on Ampere / Ada / Hopper / T4 (fallback to fp16 if no bf16)
    seed: int = 42

    # Logging & Tracking
    report_to: str = "wandb"  # or 'tensorboard', 'none'
    wandb_project: str = "gate-cs-doubt-solver"
    wandb_run_name: str = "qwen2.5-1.5b-qlora-sft-v1"
