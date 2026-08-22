"""Unit tests for the GATE CS dataset engineering pipeline."""

import pytest
from src.data.cleaner import DataCleaner
from src.data.formatter import InstructionFormatter
from src.data.models import CleanedQuestion, QuestionType, RawQuestion, Subject
from src.data.splitter import DatasetSplitter


def test_cleaner_math_and_html_entities():
    cleaner = DataCleaner()
    html_input = "<p>What is the time complexity &le; <i>O</i>(<i>n</i> log <i>n</i>)? Also consider 2<sup>k</sup> and &alpha; + &beta;.</p>"
    cleaned = cleaner.clean_html_to_markdown(html_input)

    assert "\\le" in cleaned or "<=" in cleaned or "$\\le$" in cleaned
    assert "$O(n log n)$" in cleaned or "O(n" in cleaned
    assert "^{k}" in cleaned
    assert "\\alpha" in cleaned
    assert "\\beta" in cleaned


def test_cleaner_deduplication():
    cleaner = DataCleaner()
    raw1 = RawQuestion(
        id="Q1",
        source_url="http://test.com/1",
        source_name="test",
        subject="Algorithms",
        question_type=QuestionType.MCQ,
        raw_question_html="<p>What is the worst case time complexity of QuickSort?</p>",
        raw_options_html={"A": "O(n)", "B": "O(n^2)"},
        correct_answer="B",
    )
    raw2 = RawQuestion(
        id="Q2",
        source_url="http://test.com/2",
        source_name="test",
        subject="Algorithms",
        question_type=QuestionType.MCQ,
        raw_question_html="<p>  What is the WORST case time complexity of QuickSort? </p>",
        raw_options_html={"A": "O(n)", "B": "O(n^2)"},
        correct_answer="B",
    )

    c1 = cleaner.clean_raw_question(raw1)
    assert c1 is not None

    # Duplicate should be dropped
    c2 = cleaner.clean_raw_question(raw2)
    assert c2 is None


def test_formatter_reasoning_structure():
    formatter = InstructionFormatter()
    cq = CleanedQuestion(
        id="TEST-1",
        source_url="http://test.com",
        source_name="GATE-TEST",
        subject=Subject.OPERATING_SYSTEMS,
        topic="Paging",
        year=2023,
        question_type=QuestionType.MCQ,
        marks=2,
        question_text="Consider a system with 32-bit virtual addresses and 4KB page size.",
        options={"A": "1024 pages", "B": "1048576 pages", "C": "4096 pages", "D": "256 pages"},
        correct_answer="B",
        raw_explanation="Virtual address = 32 bits. Page offset = 12 bits (4KB). Page number = 32 - 12 = 20 bits. Total pages = 2^20 = 1048576.",
        content_hash="abc12345",
    )

    record = formatter.to_instruction_record(cq)

    assert len(record.messages) == 3
    assert record.messages[0].role == "system"
    assert record.messages[1].role == "user"
    assert record.messages[2].role == "assistant"

    resp = record.messages[2].content
    assert "### 1. Conceptual Framework & Core Principles" in resp
    assert "### 2. Step-by-Step Derivation & Analysis" in resp
    assert "### 3. Option Evaluation & Verification" in resp
    assert "### 4. Final Answer" in resp
    assert "(B)" in resp


def test_splitter_zero_leakage():
    splitter = DatasetSplitter(test_years={2024}, val_years={2023})
    formatter = InstructionFormatter()

    records = []
    for i, yr in enumerate([2021, 2022, 2023, 2024]):
        cq = CleanedQuestion(
            id=f"TEST-{i}",
            source_url=f"http://test.com/{i}",
            source_name="GATE",
            subject=Subject.ALGORITHMS,
            topic="Sorting",
            year=yr,
            question_type=QuestionType.NAT,
            marks=1,
            question_text=f"Question body {i}",
            options={},
            correct_answer=str(i),
            raw_explanation=f"Explanation {i}",
            content_hash=f"hash_{i}",
        )
        records.append(formatter.to_instruction_record(cq))

    train, val, test = splitter.split_records(records)
    leakage = splitter.verify_no_leakage(train, val, test)

    assert leakage["zero_id_leakage"] is True
    assert leakage["zero_content_hash_leakage"] is True
