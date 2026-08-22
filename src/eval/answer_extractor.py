"""Answer extraction and normalization for objective MCQ/NAT benchmarking."""

import math
import re
from typing import Optional, Tuple


class AnswerExtractor:
    """Extracts machine-evaluable final answers from LLM generation text."""

    @staticmethod
    def extract_mcq_answer(text: str) -> Optional[str]:
        """Extracts option letter (A, B, C, D) from response text."""
        if not text:
            return None

        # Priority 1: Check Section 4 / Final Answer line
        final_section = re.search(r"###\s*4\.\s*Final Answer.*?(?:$|\n\n)", text, re.IGNORECASE | re.DOTALL)
        search_scope = final_section.group(0) if final_section else text[-300:]

        # Patterns like: **Correct Option**: **(B)** or Option (B) or Option B or (B)
        patterns = [
            r"Correct Option\**\s*:\s*\**\(([A-D])\)\**",
            r"Correct Option\**\s*:\s*\**([A-D])\b",
            r"Option\s*\(([A-D])\)",
            r"Option\s*([A-D])\b",
            r"\(([A-D])\)\s*is correct",
            r"\(([A-D])\)\s*—",
            r"\b([A-D])\s*is the correct option",
            r"\(([A-D])\)",
        ]

        for pat in patterns:
            match = re.search(pat, search_scope, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Fallback to whole text search
        for pat in patterns[:4]:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    @staticmethod
    def extract_nat_answer(text: str) -> Optional[str]:
        """Extracts numerical answer value from response text."""
        if not text:
            return None

        # Priority 1: Check Section 4 / Final Answer
        final_section = re.search(r"###\s*4\.\s*Final Answer.*?(?:$|\n\n)", text, re.IGNORECASE | re.DOTALL)
        search_scope = final_section.group(0) if final_section else text[-250:]

        patterns = [
            r"Numerical Answer\**\s*:\s*\**([-+]?\d*\.?\d+)\**",
            r"Calculated Result\**\s*:\s*`?([-+]?\d*\.?\d+)`?",
            r"Answer\**\s*:\s*\**([-+]?\d*\.?\d+)\**",
            r"is\s*\\mathbf\{([-+]?\d*\.?\d+)\}",
            r"is\s*\*\*([-+]?\d*\.?\d+)\*\*",
            r"`([-+]?\d*\.?\d+)`",
            r"\b([-+]?\d*\.?\d+)\b",
        ]

        for pat in patterns:
            match = re.search(pat, search_scope)
            if match:
                return match.group(1).strip()

        return None

    @classmethod
    def evaluate_correctness(
        cls,
        response_text: str,
        ground_truth: str,
        question_type: str,
        float_tolerance: float = 0.05,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether the extracted answer matches the ground truth."""
        gt_clean = ground_truth.strip().upper()

        if question_type == "MCQ":
            extracted = cls.extract_mcq_answer(response_text)
            if not extracted:
                return False, None
            return extracted == gt_clean, extracted

        elif question_type == "NAT":
            extracted = cls.extract_nat_answer(response_text)
            if not extracted:
                return False, None
            try:
                pred_val = float(extracted)
                gt_val = float(gt_clean)
                is_correct = math.isclose(pred_val, gt_val, abs_tol=float_tolerance)
                return is_correct, extracted
            except ValueError:
                return extracted == gt_clean, extracted

        else:  # MSQ or other
            extracted = cls.extract_mcq_answer(response_text)
            return (extracted == gt_clean) if extracted else False, extracted

    @staticmethod
    def check_format_compliance(text: str) -> bool:
        """Verifies if the response follows the 4-phase pedagogical CoT structure."""
        if not text:
            return False
        c1 = "### 1. Conceptual Framework" in text or "1. Conceptual Framework" in text
        c2 = "### 2. Step-by-Step Derivation" in text or "2. Step-by-Step Derivation" in text
        c3 = "### 3. Option Evaluation" in text or "3. Value Calculation" in text or "Option Evaluation" in text
        c4 = "### 4. Final Answer" in text or "4. Final Answer" in text
        return bool(c1 and c2 and c3 and c4)
