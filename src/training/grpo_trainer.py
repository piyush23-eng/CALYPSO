"""GRPO (Group Relative Policy Optimization) Trainer for GATE-CS Reasoning."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import os

from src.training.rewards import GATEVerifiableRewards
from src.utils.logger import setup_logger

logger = setup_logger("grpo_trainer")


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO Training."""
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "models/grpo_gate_cs"
    num_generations: int = 4  # G completions sampled per prompt
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 2
    temperature: float = 0.8
    beta: float = 0.04  # KL penalty coefficient
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class GRPOTrainingPipeline:
    """Orchestrates Group Relative Policy Optimization on GATE reasoning tasks."""

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()
        self.reward_engine = GATEVerifiableRewards()

    def get_reward_functions(self) -> List[Callable]:
        """Returns list of reward callables for TRL GRPOTrainer."""
        def format_reward_func(prompts, completions, **kwargs):
            return [self.reward_engine.format_reward(comp) for comp in completions]

        def accuracy_reward_func(prompts, completions, ground_truth=None, question_type=None, **kwargs):
            rewards = []
            q_types = question_type if question_type is not None else ["MCQ"] * len(completions)
            gts = ground_truth if ground_truth is not None else [""] * len(completions)
            for comp, gt, qt in zip(completions, gts, q_types):
                res = self.reward_engine.compute_reward(
                    prompt="",
                    completion=comp,
                    ground_truth=gt,
                    question_type=qt,
                )
                rewards.append(res["accuracy_reward"])
            return rewards

        return [format_reward_func, accuracy_reward_func]

    def build_grpo_trainer(self, dataset, tokenizer=None, peft_config=None):
        """Constructs TRL GRPOTrainer if trl is installed, or provides guided instructions."""
        try:
            from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
            
            trl_config = TRLGRPOConfig(
                output_dir=self.config.output_dir,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_train_epochs,
                num_generations=self.config.num_generations,
                max_prompt_length=self.config.max_prompt_length,
                max_completion_length=self.config.max_completion_length,
                temperature=self.config.temperature,
                beta=self.config.beta,
                logging_steps=10,
                save_strategy="epoch",
            )
            
            trainer = GRPOTrainer(
                model=self.config.model_name_or_path,
                reward_funcs=self.get_reward_functions(),
                args=trl_config,
                train_dataset=dataset,
                peft_config=peft_config,
            )
            return trainer
        except ImportError:
            logger.info("TRL library not installed in current environment. GRPOTrainingPipeline ready for external GPU runner.")
            return None
