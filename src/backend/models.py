"""API schemas for FastAPI Backend."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    subject: str = Field(..., json_schema_extra={"example": "Algorithms"})
    topic: Optional[str] = Field("General", json_schema_extra={"example": "Asymptotic Analysis"})
    question_type: str = Field("MCQ", json_schema_extra={"example": "MCQ"})  # MCQ, MSQ, NAT
    marks: int = Field(1, json_schema_extra={"example": 1})
    question: str = Field(..., json_schema_extra={"example": "What is the time complexity of building a heap with Floyd's method?"})
    options: Dict[str, str] = Field(default_factory=dict, json_schema_extra={"example": {"A": "O(n)", "B": "O(n log n)", "C": "O(n^2)", "D": "O(1)"}})
    model_type: str = Field("finetuned", json_schema_extra={"example": "finetuned"})  # 'finetuned' or 'base'


class SolveResponse(BaseModel):
    model_name: str
    solution_markdown: str
    extracted_answer: Optional[str] = None
    inference_latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    device: str


class CompareRequest(BaseModel):
    subject: str
    topic: Optional[str] = "General"
    question_type: str = "MCQ"
    marks: int = 1
    question: str
    options: Dict[str, str] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    base_model_result: SolveResponse
    finetuned_model_result: SolveResponse
    quality_delta: str
