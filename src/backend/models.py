"""API schemas for FastAPI Backend."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    subject: str = Field(..., json_schema_extra={"example": "Algorithms"})
    topic: Optional[str] = Field("General", json_schema_extra={"example": "Asymptotic Analysis"})
    question_type: str = Field("MCQ", json_schema_extra={"example": "MCQ"})  # MCQ, MSQ, NAT
    marks: int = Field(1, json_schema_extra={"example": 1})
    question: str = Field(..., json_schema_extra={"example": "What is the time complexity of building a heap with Floyd's method?"})
    options: Dict[str, str] = Field(default_factory=dict, json_schema_extra={"example": {"A": "O(n)", "B": "O(n log n)", "C": "O(n^2)", "D": "O(1)"}})
    model_type: str = Field("finetuned", json_schema_extra={"example": "finetuned"})  # 'finetuned' or 'base'


class StructuredPhases(BaseModel):
    phase1_concept: Optional[str] = None
    phase2_derivation: Optional[str] = None
    phase3_elimination: Optional[str] = None
    phase4_answer: Optional[str] = None


class SolveResponse(BaseModel):
    model_name: str
    solution_markdown: str
    extracted_answer: Optional[str] = None
    structured_phases: Optional[StructuredPhases] = None
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


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class MultiTurnChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chronological conversation turns")
    subject: Optional[str] = "Algorithms"
    topic: Optional[str] = "General"
    question_type: Optional[str] = "MCQ"
    marks: Optional[int] = 2
    stream: bool = False


class ImageSolveRequest(BaseModel):
    image_data: str = Field(..., description="Base64 data URI of the question screenshot")
    subject: Optional[str] = None
    topic: Optional[str] = None


class ImageSolveResponse(BaseModel):
    extracted_data: Dict[str, Any]
    solution: SolveResponse


class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="e.g. 'code_interpreter'")
    code: str = Field(..., description="Python code block to execute in sandbox")


class ToolExecutionResponse(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    variables: Optional[Dict[str, str]] = None
    elapsed_ms: float
