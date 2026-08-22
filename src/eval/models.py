"""Evaluation schemas and data models."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EvalSampleResult(BaseModel):
    """Evaluation result for an individual held-out test problem."""
    id: str
    subject: str
    topic: str
    year: Optional[int]
    question_type: str
    marks: int
    ground_truth_answer: str
    base_model_response: str
    base_model_extracted_answer: Optional[str] = None
    base_model_correct: bool = False
    base_model_format_compliant: bool = False
    finetuned_model_response: str
    finetuned_model_extracted_answer: Optional[str] = None
    finetuned_model_correct: bool = False
    finetuned_model_format_compliant: bool = False
    human_eval_base: Optional[Dict[str, float]] = None  # correctness, reasoning, conciseness
    human_eval_finetuned: Optional[Dict[str, float]] = None


class SubjectAccuracy(BaseModel):
    subject: str
    total_questions: int
    base_correct: int
    base_accuracy: float
    finetuned_correct: int
    finetuned_accuracy: float
    accuracy_delta: float


class BenchmarkSummary(BaseModel):
    """Aggregate benchmark comparison between base and fine-tuned model."""
    total_test_samples: int
    base_mcq_accuracy: float
    finetuned_mcq_accuracy: float
    base_nat_accuracy: float
    finetuned_nat_accuracy: float
    base_overall_accuracy: float
    finetuned_overall_accuracy: float
    base_format_compliance_rate: float
    finetuned_format_compliance_rate: float
    human_eval_avg_base: Dict[str, float]
    human_eval_avg_finetuned: Dict[str, float]
    per_subject_breakdown: List[SubjectAccuracy]
