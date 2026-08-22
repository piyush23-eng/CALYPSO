"""Pilot sample execution script for Phase 1: Dataset Engineering.

Generates 50-100 high-quality sample instruction pairs across all 7 subjects,
validates cleaning, deduplication, and leakage-free splitting, and exports datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    DataCleaner,
    DatasetSplitter,
    InstructionFormatter,
    InstructionRecord,
    RawDataCollector,
)
from src.utils.logger import setup_logger

logger = setup_logger("run_pilot")


def run_pipeline(sample_size: int = 70, output_dir: Path = Path("data/splits")):
    logger.info(f"Starting Phase 1 Dataset Pipeline (Target: {sample_size} samples)...")

    collector = RawDataCollector(cache_dir=PROJECT_ROOT / "data" / "raw")
    cleaner = DataCleaner(lsh_threshold=0.85)
    formatter = InstructionFormatter()
    splitter = DatasetSplitter(test_years={2023, 2024}, val_years={2021, 2022})

    # Step 1: Collect Raw Questions
    raw_questions = collector.generate_synthetic_and_scraped_bank(target_count=sample_size)
    logger.info(f"Collected {len(raw_questions)} raw question records.")

    # Step 2: Clean and Deduplicate
    cleaned_questions = []
    for raw in raw_questions:
        cleaned = cleaner.clean_raw_question(raw)
        if cleaned:
            cleaned_questions.append(cleaned)

    logger.info(f"Cleaning & Deduplication complete. Valid records: {len(cleaned_questions)} / {len(raw_questions)}")

    # Step 3: Format into SFT Reasoning Instruction Pairs
    instruction_records: List[InstructionRecord] = []
    for cq in cleaned_questions:
        rec = formatter.to_instruction_record(cq)
        instruction_records.append(rec)

    # Save complete pilot file
    output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_file = output_dir / "sample_pilot.jsonl"
    with open(pilot_file, "w", encoding="utf-8") as f:
        for r in instruction_records:
            f.write(r.model_dump_json() + "\n")
    logger.info(f"Saved complete pilot dataset ({len(instruction_records)} items) to {pilot_file}")

    # Step 4: Split with zero-leakage guarantee
    train, val, test = splitter.split_records(instruction_records)
    splitter.save_splits(train, val, test, output_dir)

    # Step 5: Verify Leakage
    leakage_checks = splitter.verify_no_leakage(train, val, test)
    logger.info(f"Leakage Verification: {leakage_checks}")

    # Step 6: Compute Statistics
    subject_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    prompt_lens: List[int] = []
    response_lens: List[int] = []

    for r in instruction_records:
        subj = r.subject.value
        qtype = r.question_type.value
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        u_len = len(r.messages[1].content.split())
        a_len = len(r.messages[2].content.split())
        prompt_lens.append(u_len)
        response_lens.append(a_len)

    avg_p = sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0
    avg_r = sum(response_lens) / len(response_lens) if response_lens else 0

    stats = {
        "total_examples": len(instruction_records),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "subject_distribution": subject_counts,
        "question_type_distribution": type_counts,
        "average_prompt_words": round(avg_p, 1),
        "average_response_words": round(avg_r, 1),
        "leakage_verification": leakage_checks,
    }

    # Save summary stats
    stats_file = PROJECT_ROOT / "data" / "pilot_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Pipeline finished successfully! Summary:\n{json.dumps(stats, indent=2)}")
    return stats, instruction_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 Pilot Dataset Pipeline")
    parser.add_argument("--sample-size", type=int, default=70, help="Number of pilot samples")
    args = parser.parse_args()
    run_pipeline(sample_size=args.sample_size)
