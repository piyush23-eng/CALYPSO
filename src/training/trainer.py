import os
from pathlib import Path
from typing import Optional, Tuple

from src.training.config import QLoRAConfig, TrainingConfig
from src.training.dataset_loader import SFTDatasetLoader
from src.utils.logger import setup_logger

logger = setup_logger("trainer")


class GATEModelTrainer:
    """Manages QLoRA setup, 4-bit model initialization, SFTTrainer loop,

    and checkpoint management.
    """

    def __init__(self, qlora_cfg: Optional[QLoRAConfig] = None, train_cfg: Optional[TrainingConfig] = None):
        self.qlora_cfg = qlora_cfg or QLoRAConfig()
        self.train_cfg = train_cfg or TrainingConfig()

    def get_bnb_config(self):
        """Constructs 4-bit NormalFloat BitsAndBytes quantization configuration."""
        import torch
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, self.qlora_cfg.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=self.qlora_cfg.use_4bit,
            bnb_4bit_quant_type=self.qlora_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.qlora_cfg.bnb_4bit_use_double_quant,
        )

    def get_lora_config(self):
        """Constructs PEFT LoRA adapter configuration."""
        from peft import LoraConfig

        return LoraConfig(
            r=self.qlora_cfg.lora_r,
            lora_alpha=self.qlora_cfg.lora_alpha,
            lora_dropout=self.qlora_cfg.lora_dropout,
            target_modules=self.qlora_cfg.target_modules,
            bias=self.qlora_cfg.bias,
            task_type=self.qlora_cfg.task_type,
        )

    def load_tokenizer_and_model(self, is_gpu: bool = True):
        """Loads tokenizer and 4-bit base model prepared for k-bit LoRA training."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, prepare_model_for_kbit_training

        logger.info(f"Loading tokenizer for {self.qlora_cfg.base_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.qlora_cfg.base_model_id,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading base model {self.qlora_cfg.base_model_id} (4-bit QLoRA: {is_gpu})...")

        if is_gpu and torch.cuda.is_available():
            bnb_config = self.get_bnb_config()
            model = AutoModelForCausalLM.from_pretrained(
                self.qlora_cfg.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.qlora_cfg.torch_dtype),
            )
            # Enable gradient checkpointing and prepare for kbit
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.train_cfg.gradient_checkpointing,
            )
        else:
            # Fallback for CPU / dry-run testing
            logger.warning("CUDA not available. Loading in standard CPU mode for validation.")
            model = AutoModelForCausalLM.from_pretrained(
                self.qlora_cfg.base_model_id,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

        lora_config = self.get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return tokenizer, model

    def build_sft_trainer(
        self,
        model,
        tokenizer,
        dataset_dict,
    ):
        """Constructs the TRL SFTTrainer instance with all logging and checkpoint hooks."""
        import torch
        from trl import SFTConfig, SFTTrainer

        training_args = SFTConfig(
            output_dir=self.train_cfg.output_dir,
            num_train_epochs=self.train_cfg.num_train_epochs,
            max_steps=self.train_cfg.max_steps,
            per_device_train_batch_size=self.train_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.train_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            learning_rate=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            warmup_ratio=self.train_cfg.warmup_ratio,
            lr_scheduler_type=self.train_cfg.lr_scheduler_type,
            optim=self.train_cfg.optim if torch.cuda.is_available() else "adamw_torch",
            logging_steps=self.train_cfg.logging_steps,
            eval_strategy=self.train_cfg.eval_strategy,
            eval_steps=self.train_cfg.eval_steps,
            save_strategy=self.train_cfg.save_strategy,
            save_steps=self.train_cfg.save_steps,
            save_total_limit=self.train_cfg.save_total_limit,
            load_best_model_at_end=self.train_cfg.load_best_model_at_end,
            metric_for_best_model=self.train_cfg.metric_for_best_model,
            greater_is_better=self.train_cfg.greater_is_better,
            gradient_checkpointing=self.train_cfg.gradient_checkpointing if torch.cuda.is_available() else False,
            bf16=self.train_cfg.bf16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else False,
            fp16=self.train_cfg.fp16 if (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()) else False,
            seed=self.train_cfg.seed,
            report_to=self.train_cfg.report_to if os.environ.get("WANDB_API_KEY") else "none",
            run_name=self.train_cfg.wandb_run_name,
            max_seq_length=self.train_cfg.max_seq_length,
            packing=self.train_cfg.packing,
            dataset_text_field=None,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            tokenizer=tokenizer,
            peft_config=self.get_lora_config(),
        )

        return trainer

    def train_and_save(self) -> None:
        """Runs the complete end-to-end SFT training loop and saves best adapter."""
        import torch

        logger.info("Initializing Dataset Loader...")
        loader = SFTDatasetLoader(Path(self.train_cfg.dataset_dir))
        datasets = loader.prepare_datasets()

        logger.info(f"Train samples: {len(datasets['train'])}, Val samples: {len(datasets['validation'])}")

        tokenizer, model = self.load_tokenizer_and_model(is_gpu=torch.cuda.is_available())
        trainer = self.build_sft_trainer(model, tokenizer, datasets)

        logger.info("Starting SFT training...")
        train_result = trainer.train()

        logger.info(f"Training completed! Metrics: {train_result.metrics}")

        # Save best adapter and tokenizer
        adapter_path = Path(self.train_cfg.final_adapter_dir)
        adapter_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logger.info(f"Best LoRA adapter saved to {adapter_path}")
