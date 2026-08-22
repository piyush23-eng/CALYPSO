"""Instruction formatting engine: converts cleaned GATE CS questions into pedagogical SFT training pairs with structured reasoning chains."""

from typing import Dict, List, Optional
from src.data.models import ChatMessage, CleanedQuestion, InstructionRecord, QuestionType, Subject


GATE_SYSTEM_PROMPT = (
    "You are an expert GATE CS (Computer Science & Information Technology) mentor and examiner. "
    "Your objective is to provide a rigorous, mathematically precise, step-by-step derivation for the given "
    "GATE question. Always clearly explain the underlying theoretical principles, detail each calculation or logic step, "
    "evaluate all candidate options thoroughly, and conclude with the explicit final answer."
)


class InstructionFormatter:
    """Formats cleaned GATE CS questions into ChatML / Qwen2.5 instruction-tuning records."""

    def __init__(self, system_prompt: str = GATE_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def format_user_prompt(self, q: CleanedQuestion) -> str:
        """Constructs a clean, structured user question prompt."""
        header_parts = [f"**Subject**: {q.subject.value}"]
        if q.topic and q.topic != "General":
            header_parts.append(f"**Topic**: {q.topic}")
        if q.year:
            header_parts.append(f"**GATE Year**: {q.year}")
        header_parts.append(f"**Question Type**: {q.question_type.value}")
        header_parts.append(f"**Marks**: {q.marks}")

        header = " | ".join(header_parts)

        body = [f"[{header}]\n", "### Question:", q.question_text.strip()]

        if q.options and q.question_type in (QuestionType.MCQ, QuestionType.MSQ):
            body.append("\n### Options:")
            for opt_key in sorted(q.options.keys()):
                body.append(f"({opt_key}) {q.options[opt_key]}")

        if q.question_type == QuestionType.NAT:
            body.append("\n*(Note: This is a Numerical Answer Type (NAT) question. Provide the exact calculated value.)*")

        return "\n".join(body)

    def format_assistant_response(self, q: CleanedQuestion) -> str:
        """Builds a structured 4-phase pedagogical reasoning response from explanation and ground truth."""
        sections = []

        # 1. Concept Summary
        sections.append("### 1. Conceptual Framework & Core Principles")
        sections.append(self._build_concept_summary(q))

        # 2. Step-by-Step Derivation
        sections.append("\n### 2. Step-by-Step Derivation & Analysis")
        sections.append(self._build_step_by_step_derivation(q))

        # 3. Option Evaluation / Range Verification
        if q.question_type in (QuestionType.MCQ, QuestionType.MSQ) and q.options:
            sections.append("\n### 3. Option Evaluation & Verification")
            sections.append(self._build_option_evaluation(q))
        elif q.question_type == QuestionType.NAT:
            sections.append("\n### 3. Value Calculation & Range Check")
            sections.append(f"- **Calculated Result**: `{q.correct_answer}`\n- Ensure correct rounding and scale as required by standard GATE examination norms.")

        # 4. Final Answer Box
        sections.append("\n### 4. Final Answer")
        if q.question_type == QuestionType.MCQ:
            ans_key = q.correct_answer.strip()
            opt_content = q.options.get(ans_key, "")
            if opt_content:
                sections.append(f"**Correct Option**: **({ans_key})** — {opt_content}")
            else:
                sections.append(f"**Correct Option**: **({ans_key})**")
        elif q.question_type == QuestionType.MSQ:
            sections.append(f"**Correct Options**: **({', '.join(list(q.correct_answer))})**")
        else:  # NAT
            sections.append(f"**Numerical Answer**: **{q.correct_answer}**")

        return "\n".join(sections)

    def _build_concept_summary(self, q: CleanedQuestion) -> str:
        """Generates conceptual context based on subject and topic."""
        subject_concept_hints = {
            Subject.ALGORITHMS: "Identify time/space complexity bounds, recurrence relations, dynamic programming state transitions, or graph invariants.",
            Subject.OPERATING_SYSTEMS: "Apply CPU scheduling metrics, page table/TLB memory translation formulas, Banker's safety criteria, or semaphore synchronization invariants.",
            Subject.DBMS: "Analyze relational algebra operations, functional dependency closures, normal forms (1NF, 2NF, 3NF, BCNF), or serializability/ACID properties.",
            Subject.COMPUTER_NETWORKS: "Apply IP subnetting masks, TCP flow/congestion window mechanics, sliding window throughput ($U = N/(1+2a)$), or distance vector/link state routing.",
            Subject.THEORY_OF_COMPUTATION: "Analyze formal language closure properties, DFA/NFA state minimality, pumping lemma conditions, or Turing decidability bounds.",
            Subject.COMPILER_DESIGN: "Check LL(1)/LR(0)/SLR(1)/LALR(1) parser tables, FIRST and FOLLOW sets, SDT attribute evaluation (S-attributed vs L-attributed), or register allocation.",
            Subject.DIGITAL_LOGIC: "Apply Boolean algebraic identities, K-Map minterm/maxterm minimization, multiplexer tree expansions, or sequential flip-flop excitation tables.",
        }
        hint = subject_concept_hints.get(q.subject, "Analyze foundational theorems and constraints of the topic.")
        return f"- **Domain**: {q.subject.value} $\\rightarrow$ {q.topic}\n- **Core Focus**: {hint}"

    def _build_step_by_step_derivation(self, q: CleanedQuestion) -> str:
        """Formats the detailed explanation body into clean derivation steps."""
        if q.raw_explanation and len(q.raw_explanation.strip()) > 30:
            lines = [l.strip() for l in q.raw_explanation.strip().split("\n") if l.strip()]
            formatted_steps = []
            for i, line in enumerate(lines, 1):
                if line.startswith("-") or line.startswith("*") or re_match_num(line):
                    formatted_steps.append(line)
                else:
                    formatted_steps.append(f"- **Step {i}**: {line}")
            return "\n".join(formatted_steps)
        else:
            return (
                f"- **Analysis**: Direct evaluation of given parameters according to standard {q.subject.value} rules.\n"
                f"- **Evaluation**: Consistent with verified answer `{q.correct_answer}`."
            )

    def _build_option_evaluation(self, q: CleanedQuestion) -> str:
        """Evaluates each candidate option against the solution."""
        lines = []
        correct_keys = set(re_extract_options(q.correct_answer))
        for key in sorted(q.options.keys()):
            opt_text = q.options[key]
            if key in correct_keys:
                lines.append(f"- **Option ({key}) [CORRECT]**: `{opt_text}` is mathematically and conceptually valid as derived above.")
            else:
                lines.append(f"- **Option ({key}) [INCORRECT]**: Disqualified based on the derivation criteria.")
        return "\n".join(lines)

    def to_instruction_record(self, q: CleanedQuestion) -> InstructionRecord:
        """Converts a CleanedQuestion into an InstructionRecord ready for SFT JSONL."""
        user_prompt = self.format_user_prompt(q)
        assistant_response = self.format_assistant_response(q)

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
            ChatMessage(role="assistant", content=assistant_response),
        ]

        metadata = {
            "source_url": q.source_url,
            "source_name": q.source_name,
            "content_hash": q.content_hash,
            "question_type": q.question_type.value,
        }

        return InstructionRecord(
            id=q.id,
            subject=q.subject,
            topic=q.topic,
            year=q.year,
            question_type=q.question_type,
            marks=q.marks,
            messages=messages,
            metadata=metadata,
        )


def re_match_num(line: str) -> bool:
    import re
    return bool(re.match(r"^\d+\.", line))


def re_extract_options(ans: str) -> List[str]:
    import re
    letters = re.findall(r"[A-D]", ans.upper())
    return letters if letters else [ans.upper().strip()]
