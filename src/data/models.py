"""Data models and schemas for the GATE CS dataset engineering pipeline."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    MCQ = "MCQ"  # Multiple Choice Question (Single Correct)
    MSQ = "MSQ"  # Multiple Select Question (One or more correct)
    NAT = "NAT"  # Numerical Answer Type (Value / Range)


class Subject(str, Enum):
    ALGORITHMS = "Algorithms"
    OPERATING_SYSTEMS = "Operating Systems"
    DBMS = "DBMS"
    COMPUTER_NETWORKS = "Computer Networks"
    THEORY_OF_COMPUTATION = "Theory of Computation"
    COMPILER_DESIGN = "Compiler Design"
    DIGITAL_LOGIC = "Digital Logic"


class RawQuestion(BaseModel):
    """Represents a raw scraped question entry."""
    id: str
    source_url: str
    source_name: str
    subject: str
    topic: Optional[str] = "General"
    year: Optional[int] = None
    question_type: QuestionType = QuestionType.MCQ
    marks: Optional[int] = 1
    raw_question_html: str
    raw_options_html: Dict[str, str] = Field(default_factory=dict)
    correct_answer: str
    raw_explanation_html: Optional[str] = None


class CleanedQuestion(BaseModel):
    """Represents a cleaned, normalized question entry with LaTeX math."""
    id: str
    source_url: str
    source_name: str
    subject: Subject
    topic: str
    year: Optional[int]
    question_type: QuestionType
    marks: int
    question_text: str
    options: Dict[str, str] = Field(default_factory=dict)
    correct_answer: str
    raw_explanation: str
    content_hash: str


class ReasoningChain(BaseModel):
    """Structured chain-of-thought derivation for GATE problems."""
    concept_summary: str
    step_by_step_derivation: List[str]
    option_analysis: Dict[str, str] = Field(default_factory=dict)
    final_answer: str
    full_solution_markdown: str


class ChatMessage(BaseModel):
    role: str
    content: str


class InstructionRecord(BaseModel):
    """Standardized instruction-tuning entry in ChatML / SFT format."""
    id: str
    subject: Subject
    topic: str
    year: Optional[int]
    question_type: QuestionType
    marks: int
    messages: List[ChatMessage]
    metadata: Dict[str, str] = Field(default_factory=dict)
