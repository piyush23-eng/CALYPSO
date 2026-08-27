"""Tests for Vision OCR Extractor and Python Code Interpreter."""

import base64
import pytest
from src.tools.code_interpreter import CodeInterpreter, SecurityError
from src.vision.ocr_extractor import VisionOCRExtractor


def test_code_interpreter_basic_math():
    interpreter = CodeInterpreter()
    res = interpreter.execute("x = 2 ** 10\nprint(x)")
    assert res["success"] is True
    assert res["output"] == "1024"
    assert res["variables"]["x"] == "1024"


def test_code_interpreter_security_block_os():
    interpreter = CodeInterpreter()
    res = interpreter.execute("import os\nos.system('ls')")
    assert res["success"] is False
    assert "prohibited" in res["error"].lower()


def test_code_interpreter_security_block_eval():
    interpreter = CodeInterpreter()
    res = interpreter.execute("eval('1 + 1')")
    assert res["success"] is False
    assert "prohibited" in res["error"].lower()


def test_ocr_text_structuring():
    extractor = VisionOCRExtractor()
    sample_text = (
        "Consider a 2-level paging system where main memory access time is 100ns and TLB access time is 10ns.\n"
        "(A) 110 ns\n"
        "(B) 120 ns\n"
        "(C) 130 ns\n"
        "(D) 140 ns"
    )
    parsed = extractor.parse_structured_question(sample_text)
    assert parsed["question_type"] == "MCQ"
    assert "A" in parsed["options"]
    assert "B" in parsed["options"]
    assert "C" in parsed["options"]
    assert "D" in parsed["options"]
    assert parsed["subject"] == "Operating Systems"


def test_ocr_base64_decode():
    extractor = VisionOCRExtractor()
    raw = b"Sample Image Bytes"
    b64 = "data:image/png;base64," + base64.b64encode(raw).decode("utf-8")
    decoded = extractor.decode_image_data(b64)
    assert decoded == raw
