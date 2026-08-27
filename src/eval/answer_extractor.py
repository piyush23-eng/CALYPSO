"""Answer extraction and normalization for objective MCQ/MSQ/NAT benchmarking."""

import math
import re
from typing import List, Optional, Tuple


class AnswerExtractor:
    """Extracts machine-evaluable final answers from LLM generation text."""

    @staticmethod
    def extract_mcq_answer(text: str) -> Optional[str]:
        """Extracts single option letter (A, B, C, D) from response text."""
        if not text:
            return None

        # Priority 1: Check Section 4 / Final Answer line
        final_section = re.search(r"###\s*4\.\s*Final Answer.*?(?:$|\n\n)", text, re.IGNORECASE | re.DOTALL)
        search_scope = final_section.group(0) if final_section else text[-300:]

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
    def extract_msq_answers(text: str) -> List[str]:
        """Extracts multiple select option letters (e.g. ['A', 'C']) from response text."""
        if not text:
            return []

        final_section = re.search(r"###\s*4\.\s*Final Answer.*?(?:$|\n\n)", text, re.IGNORECASE | re.DOTALL)
        search_scope = final_section.group(0) if final_section else text[-400:]

        # Look for patterns like **(A, C)** or Options A, C or (A) and (C)
        multi_pat = r"Correct Options?\**\s*:\s*\**\(?([A-D](?:\s*,\s*[A-D])+)\)?"
        m = re.search(multi_pat, search_scope, re.IGNORECASE)
        if m:
            return sorted(list(set(re.findall(r"[A-D]", m.group(1).upper()))))

        # Look for all (A), (B), etc. in final scope
        matches = re.findall(r"\(([A-D])\)", search_scope)
        if matches:
            return sorted(list(set(m.upper() for m in matches)))

        # Fallback to single MCQ extraction
        single = AnswerExtractor.extract_mcq_answer(text)
        return [single] if single else []

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

        else:  # MSQ
            extracted_list = cls.extract_msq_answers(response_text)
            extracted_str = ", ".join(extracted_list)
            gt_set = set(re.findall(r"[A-D]", gt_clean))
            is_correct = set(extracted_list) == gt_set
            return is_correct, extracted_str

    @staticmethod
    def check_format_compliance(text: str) -> bool:
        """Verifies if the response follows the 4-phase pedagogical CoT structure."""
        if not text:
            return False
        c1 = "### 1. Conceptual Framework" in text or "1. Conceptual Framework" in text or "Phase 1" in text
        c2 = "### 2. Step-by-Step Derivation" in text or "2. Step-by-Step Derivation" in text or "Phase 2" in text
        c3 = "### 3. Option Evaluation" in text or "3. Value Calculation" in text or "Option Evaluation" in text or "Phase 3" in text
        c4 = "### 4. Final Answer" in text or "4. Final Answer" in text or "Phase 4" in text
        return bool(c1 and c2 and c3 and c4)
