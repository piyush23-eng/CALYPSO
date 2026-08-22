"""Dataset loader and tokenizer chat template formatting for SFT training."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import Dataset, DatasetDict


class SFTDatasetLoader:
    """Loads and formats ChatML JSONL splits for SFT training."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Reads a JSONL split file into a list of dictionaries."""
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset split not found: {filepath}")

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records

    def prepare_datasets(self) -> DatasetDict:
        """Loads train and val splits and formats them into HuggingFace DatasetDict."""
        train_file = self.dataset_dir / "train.jsonl"
        val_file = self.dataset_dir / "val.jsonl"

        train_data = self.load_jsonl(train_file)
        val_data = self.load_jsonl(val_file)

        # Extract only messages list formatted for SFTTrainer
        train_formatted = [{"messages": item["messages"]} for item in train_data]
        val_formatted = [{"messages": item["messages"]} for item in val_data]

        ds_dict = DatasetDict({
            "train": Dataset.from_list(train_formatted),
            "validation": Dataset.from_list(val_formatted),
        })

        return ds_dict

    @staticmethod
    def format_chat_prompt(tokenizer, messages: List[Dict[str, str]], tokenize: bool = False, max_length: int = 1024):
        """Applies tokenizer's native chat template to messages."""
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=False,
            max_length=max_length if tokenize else None,
            truncation=tokenize,
        )
