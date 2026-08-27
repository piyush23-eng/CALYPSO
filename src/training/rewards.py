"""Verifiable Reward Functions for GATE-CS RLVR / GRPO Training."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.eval.answer_extractor import AnswerExtractor


class GATEVerifiableRewards:
    """Computes verifiable rule-based reward signals for GATE questions."""

    def __init__(self):
        self.extractor = AnswerExtractor()

    def format_reward(self, completion: str) -> float:
        """Evaluates whether the generation follows the 4-phase CoT format."""
        score = 0.0
        lower = completion.lower()

        # Phase 1: Concept
        if any(h in lower for h in ["phase 1", "core concept", "key concept", "formula", "background theory", "conceptual framework"]):
            score += 0.25

        # Phase 2: Derivation
        if any(h in lower for h in ["phase 2", "derivation", "step-by-step", "calculation", "proof"]):
            score += 0.25

        # Phase 3: Option Elimination / Trap Analysis
        if any(h in lower for h in ["phase 3", "option elimination", "eliminat", "trap analysis", "verification", "option evaluation", "value calculation"]):
            score += 0.25

        # Phase 4: Final Answer
        if any(h in lower for h in ["phase 4", "final answer", "correct answer", "conclusion", "numerical answer"]):
            score += 0.25

        # Penalize degenerate repetition
        lines = [line.strip() for line in completion.split("\n") if len(line.strip()) > 20]
        if len(lines) > 5:
            unique_ratio = len(set(lines)) / len(lines)
            if unique_ratio < 0.6:
                score *= 0.5

        return score

    def mcq_reward(self, completion: str, ground_truth: str) -> float:
        """Evaluates single-choice MCQ correctness (+1.0 for exact match, 0.0 otherwise)."""
        extracted = self.extractor.extract_mcq_answer(completion)
        if not extracted:
            return 0.0
        
        gt = ground_truth.strip().upper()
        if extracted.upper() == gt:
            return 1.0
        return 0.0

    def msq_reward(self, completion: str, ground_truth: Union[str, List[str]]) -> float:
        """Evaluates Multiple Select Question (MSQ) correctness using set Jaccard score."""
        extracted_list = self.extractor.extract_msq_answers(completion)
        extracted_set = set(k.upper() for k in extracted_list)
        
        if isinstance(ground_truth, str):
            gt_set = set(re.findall(r"[A-D]", ground_truth.upper()))
        else:
            gt_set = set(k.upper() for k in ground_truth)

        if not extracted_set or not gt_set:
            return 0.0

        intersection = len(extracted_set & gt_set)
        union = len(extracted_set | gt_set)
        
        if extracted_set == gt_set:
            return 1.0
        
        return round(intersection / union, 3)

    def nat_reward(self, completion: str, ground_truth: Union[float, int, str], tolerance: float = 0.05) -> float:
        """Evaluates Numerical Answer Type (NAT) precision with tolerance margin."""
        extracted = self.extractor.extract_nat_answer(completion)
        if extracted is None:
            return 0.0

        try:
            pred_val = float(extracted)
            gt_val = float(ground_truth)
            if abs(pred_val - gt_val) <= max(tolerance, 1e-4):
                return 1.0
            
            if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) <= 0.01:
                return 0.9
        except (ValueError, TypeError):
            if isinstance(ground_truth, str) and "to" in ground_truth:
                parts = re.findall(r"[-+]?\d*\.?\d+", ground_truth)
                if len(parts) >= 2:
                    low, high = float(parts[0]), float(parts[1])
                    try:
                        val = float(extracted)
                        if low <= val <= high:
                            return 1.0
                    except ValueError:
                        pass

        return 0.0

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        ground_truth: Any,
        question_type: str = "MCQ",
        format_weight: float = 0.3,
        accuracy_weight: float = 0.7,
    ) -> Dict[str, float]:
        """Calculates total composite reward combining format compliance and verifiable accuracy."""
        f_score = self.format_reward(completion)
        
        q_type_upper = question_type.upper()
        if q_type_upper == "MCQ":
            acc_score = self.mcq_reward(completion, str(ground_truth))
        elif q_type_upper == "MSQ":
            acc_score = self.msq_reward(completion, ground_truth)
        elif q_type_upper == "NAT":
            acc_score = self.nat_reward(completion, ground_truth)
        else:
            acc_score = self.mcq_reward(completion, str(ground_truth))

        total = (format_weight * f_score) + (accuracy_weight * acc_score)
        
        return {
            "total_reward": round(total, 4),
            "accuracy_reward": round(acc_score, 4),
            "format_reward": round(f_score, 4),
        }
