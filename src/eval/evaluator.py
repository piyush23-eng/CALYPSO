"""Rigorous evaluation runner for Base vs Fine-Tuned model comparison."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.eval.answer_extractor import AnswerExtractor
from src.eval.models import BenchmarkSummary, EvalSampleResult, SubjectAccuracy
from src.utils.logger import setup_logger

logger = setup_logger("evaluator")


class ModelEvaluator:
    """Runs automated objective benchmarking and human eval aggregation on the held-out test set."""

    def __init__(self, test_split_path: Path):
        self.test_split_path = Path(test_split_path)
        self.extractor = AnswerExtractor()

    def load_test_records(self) -> List[Dict]:
        records = []
        with open(self.test_split_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records

    def run_benchmark(self, sample_results: List[EvalSampleResult]) -> BenchmarkSummary:
        """Computes aggregate accuracy, format compliance, and subject breakdown."""
        total = len(sample_results)
        if total == 0:
            raise ValueError("No sample results provided for benchmarking.")

        mcq_samples = [s for s in sample_results if s.question_type == "MCQ"]
        nat_samples = [s for s in sample_results if s.question_type == "NAT"]

        base_mcq_acc = sum(1 for s in mcq_samples if s.base_model_correct) / len(mcq_samples) if mcq_samples else 0.0
        ft_mcq_acc = sum(1 for s in mcq_samples if s.finetuned_model_correct) / len(mcq_samples) if mcq_samples else 0.0

        base_nat_acc = sum(1 for s in nat_samples if s.base_model_correct) / len(nat_samples) if nat_samples else 0.0
        ft_nat_acc = sum(1 for s in nat_samples if s.finetuned_model_correct) / len(nat_samples) if nat_samples else 0.0

        base_overall = sum(1 for s in sample_results if s.base_model_correct) / total
        ft_overall = sum(1 for s in sample_results if s.finetuned_model_correct) / total

        base_format = sum(1 for s in sample_results if s.base_model_format_compliant) / total
        ft_format = sum(1 for s in sample_results if s.finetuned_model_format_compliant) / total

        # Human Eval Averages
        base_h_scores = [s.human_eval_base for s in sample_results if s.human_eval_base]
        ft_h_scores = [s.human_eval_finetuned for s in sample_results if s.human_eval_finetuned]

        def avg_rubric(scores_list):
            if not scores_list:
                return {"correctness": 0.0, "reasoning_depth": 0.0, "conciseness": 0.0, "overall": 0.0}
            c = sum(x.get("correctness", 0) for x in scores_list) / len(scores_list)
            r = sum(x.get("reasoning_depth", 0) for x in scores_list) / len(scores_list)
            k = sum(x.get("conciseness", 0) for x in scores_list) / len(scores_list)
            return {
                "correctness": round(c, 2),
                "reasoning_depth": round(r, 2),
                "conciseness": round(k, 2),
                "overall": round((c + r + k) / 3.0, 2),
            }

        avg_base_human = avg_rubric(base_h_scores)
        avg_ft_human = avg_rubric(ft_h_scores)

        # Subject breakdown
        df = pd.DataFrame([
            {
                "subject": s.subject,
                "base_correct": 1 if s.base_model_correct else 0,
                "ft_correct": 1 if s.finetuned_model_correct else 0,
            }
            for s in sample_results
        ])

        subject_breakdown = []
        for subj, group in df.groupby("subject"):
            n = len(group)
            b_corr = group["base_correct"].sum()
            ft_corr = group["ft_correct"].sum()
            b_acc = (b_corr / n) * 100
            ft_acc = (ft_corr / n) * 100
            subject_breakdown.append(
                SubjectAccuracy(
                    subject=str(subj),
                    total_questions=n,
                    base_correct=int(b_corr),
                    base_accuracy=round(b_acc, 1),
                    finetuned_correct=int(ft_corr),
                    finetuned_accuracy=round(ft_acc, 1),
                    accuracy_delta=round(ft_acc - b_acc, 1),
                )
            )

        return BenchmarkSummary(
            total_test_samples=total,
            base_mcq_accuracy=round(base_mcq_acc * 100, 1),
            finetuned_mcq_accuracy=round(ft_mcq_acc * 100, 1),
            base_nat_accuracy=round(base_nat_acc * 100, 1),
            finetuned_nat_accuracy=round(ft_nat_acc * 100, 1),
            base_overall_accuracy=round(base_overall * 100, 1),
            finetuned_overall_accuracy=round(ft_overall * 100, 1),
            base_format_compliance_rate=round(base_format * 100, 1),
            finetuned_format_compliance_rate=round(ft_format * 100, 1),
            human_eval_avg_base=avg_base_human,
            human_eval_avg_finetuned=avg_ft_human,
            per_subject_breakdown=subject_breakdown,
        )
