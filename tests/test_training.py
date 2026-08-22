"""Unit tests for Phase 2 training configuration and dataset loading."""

from pathlib import Path
import pytest
from src.training.config import QLoRAConfig, TrainingConfig
from src.training.dataset_loader import SFTDatasetLoader


def test_qlora_config_defaults():
    cfg = QLoRAConfig()
    assert cfg.base_model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules
    assert "gate_proj" in cfg.target_modules
    assert cfg.use_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.learning_rate == 2e-4
    assert cfg.per_device_train_batch_size == 2
    assert cfg.gradient_accumulation_steps == 8
    assert cfg.load_best_model_at_end is True
    assert cfg.metric_for_best_model == "eval_loss"
    assert cfg.greater_is_better is False
    assert cfg.max_seq_length == 1024


def test_sft_dataset_loader(tmp_path):
    # Create mock train.jsonl and val.jsonl
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    mock_row = {
        "id": "T1",
        "subject": "Algorithms",
        "topic": "Sorting",
        "year": 2020,
        "question_type": "MCQ",
        "marks": 1,
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    }

    import json

    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(mock_row) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(mock_row) + "\n")

    loader = SFTDatasetLoader(tmp_path)
    ds = loader.prepare_datasets()

    assert "train" in ds
    assert "validation" in ds
    assert len(ds["train"]) == 1
    assert len(ds["validation"]) == 1
    assert "messages" in ds["train"][0]
    assert len(ds["train"][0]["messages"]) == 3
