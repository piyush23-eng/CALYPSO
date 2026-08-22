"""Unit tests for Phase 3 evaluation pipeline."""

import pytest
from src.eval.answer_extractor import AnswerExtractor
from src.eval.models import EvalSampleResult
from src.eval.evaluator import ModelEvaluator


def test_mcq_answer_extraction():
    extractor = AnswerExtractor()

    text1 = "Therefore, the correct option is (C) — 3NF but not BCNF."
    assert extractor.extract_mcq_answer(text1) == "C"

    text2 = "### 4. Final Answer\n**Correct Option**: **(B)** — $\\Theta(n \\log n)$"
    assert extractor.extract_mcq_answer(text2) == "B"

    text3 = "Option A is the right answer."
    assert extractor.extract_mcq_answer(text3) == "A"


def test_nat_answer_extraction():
    extractor = AnswerExtractor()

    text1 = "### 4. Final Answer\n**Numerical Answer**: **96**"
    assert extractor.extract_nat_answer(text1) == "96"

    text2 = "The calculated shortest path weight is `8`."
    assert extractor.extract_nat_answer(text2) == "8"


def test_correctness_evaluation():
    extractor = AnswerExtractor()

    # MCQ match
    corr, ext = extractor.evaluate_correctness("Correct Option: (B)", "B", "MCQ")
    assert corr is True
    assert ext == "B"

    # MCQ mismatch
    corr, ext = extractor.evaluate_correctness("Correct Option: (A)", "B", "MCQ")
    assert corr is False

    # NAT match with float tolerance
    corr, ext = extractor.evaluate_correctness("Calculated Result: `96.0`", "96", "NAT")
    assert corr is True


def test_format_compliance_check():
    extractor = AnswerExtractor()

    compliant_text = (
        "### 1. Conceptual Framework & Core Principles\n- Theory\n"
        "### 2. Step-by-Step Derivation & Analysis\n- Step 1\n"
        "### 3. Option Evaluation & Verification\n- (A) is wrong\n"
        "### 4. Final Answer\n**Correct Option**: **(A)**"
    )
    assert extractor.check_format_compliance(compliant_text) is True

    non_compliant_text = "The answer is option A because of master theorem."
    assert extractor.check_format_compliance(non_compliant_text) is False
