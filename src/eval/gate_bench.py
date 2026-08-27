"""Standardized GATE-CS-Bench Benchmark Suite and Leaderboard Exporter."""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eval.answer_extractor import AnswerExtractor
from src.training.rewards import GATEVerifiableRewards
from src.utils.logger import setup_logger

logger = setup_logger("gate_bench")


class GATECSBench:
    """Evaluates language models on the standardized GATE-CS-Bench benchmark."""

    def __init__(self, data_path: str = "data/splits/test.jsonl"):
        self.data_path = Path(data_path)
        self.extractor = AnswerExtractor()
        self.reward_engine = GATEVerifiableRewards()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads benchmark test instances."""
        if not self.data_path.exists():
            logger.warning(f"Benchmark file {self.data_path} not found. Returning empty sample.")
            return []
        
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def evaluate_model(
        self,
        inference_fn,
        limit: Optional[int] = None,
        model_name: str = "CALYPSO-1.5B",
    ) -> Dict[str, Any]:
        """Runs evaluation over benchmark samples and produces categorized metrics."""
        samples = self.load_dataset()
        if limit:
            samples = samples[:limit]

        total = len(samples)
        if total == 0:
            return {"error": "No test samples available."}

        correct_count = 0
        format_compliant_count = 0
        subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        latencies = []

        logger.info(f"Running GATE-CS-Bench evaluation on {total} samples for {model_name}...")

        for idx, item in enumerate(samples):
            # Parse item fields
            subj = item.get("subject", "General")
            q_type = item.get("question_type", "MCQ")
            gt = item.get("correct_answer") or item.get("ground_truth") or "A"

            subject_stats[subj]["total"] += 1
            type_stats[q_type]["total"] += 1

            start_t = time.perf_counter()
            response = inference_fn(item)
            latency_ms = (time.perf_counter() - start_t) * 1000
            latencies.append(latency_ms)

            # Compute verifiable reward
            rewards = self.reward_engine.compute_reward(
                prompt="",
                completion=response,
                ground_truth=gt,
                question_type=q_type,
            )

            is_correct = rewards["accuracy_reward"] >= 0.99
            is_compliant = rewards["format_reward"] >= 0.75

            if is_correct:
                correct_count += 1
                subject_stats[subj]["correct"] += 1
                type_stats[q_type]["correct"] += 1

            if is_compliant:
                format_compliant_count += 1

        overall_accuracy = round((correct_count / total) * 100, 2)
        format_compliance = round((format_compliant_count / total) * 100, 2)
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

        # Compile subject accuracy
        subject_breakdown = {}
        for s, counts in subject_stats.items():
            acc = round((counts["correct"] / counts["total"]) * 100, 1) if counts["total"] > 0 else 0
            subject_breakdown[s] = {"accuracy": acc, "total": counts["total"], "correct": counts["correct"]}

        # Compile question type accuracy
        type_breakdown = {}
        for t, counts in type_stats.items():
            acc = round((counts["correct"] / counts["total"]) * 100, 1) if counts["total"] > 0 else 0
            type_breakdown[t] = {"accuracy": acc, "total": counts["total"], "correct": counts["correct"]}

        summary = {
            "benchmark": "GATE-CS-Bench v1.0",
            "model_name": model_name,
            "total_samples": total,
            "overall_accuracy_pct": overall_accuracy,
            "format_compliance_pct": format_compliance,
            "average_latency_ms": avg_latency,
            "subject_breakdown": subject_breakdown,
            "question_type_breakdown": type_breakdown,
        }

        return summary

    def generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generates a publication-ready Markdown table for GitHub and Hugging Face."""
        md = []
        md.append(f"# 📊 GATE-CS-Bench Leaderboard Evaluation: {summary.get('model_name')}\n")
        md.append(f"- **Overall Pass@1 Accuracy**: **`{summary.get('overall_accuracy_pct')}%`**")
        md.append(f"- **4-Phase Format Compliance**: **`{summary.get('format_compliance_pct')}%`**")
        md.append(f"- **Average Latency**: `{summary.get('average_latency_ms')} ms / query`\n")

        md.append("### 📚 Subject-Wise Accuracy Breakdown\n")
        md.append("| Subject Area | Total Questions | Correct | Accuracy (%) |")
        md.append("| :--- | :---: | :---: | :---: |")
        for subj, d in summary.get("subject_breakdown", {}).items():
            md.append(f"| {subj} | {d['total']} | {d['correct']} | **{d['accuracy']}%** |")

        md.append("\n### 🧩 Question-Type Performance\n")
        md.append("| Question Format | Total | Correct | Accuracy (%) |")
        md.append("| :--- | :---: | :---: | :---: |")
        for q_type, d in summary.get("question_type_breakdown", {}).items():
            md.append(f"| {q_type} | {d['total']} | {d['correct']} | **{d['accuracy']}%** |")

        return "\n".join(md)
