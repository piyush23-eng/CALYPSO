"""Vision & OCR Question Ingestion Engine for CALYPSO."""

import base64
import io
import re
from typing import Any, Dict, Optional, Tuple

from src.data.cleaner import DataCleaner
from src.data.models import QuestionType, Subject
from src.utils.logger import setup_logger

logger = setup_logger("vision_ocr")


class VisionOCRExtractor:
    """Extracts, standardizes, and parses GATE questions from images/screenshots."""

    def __init__(self):
        self.cleaner = DataCleaner()

    def decode_image_data(self, image_data: str) -> bytes:
        """Decodes base64 string (including data URI) into raw bytes."""
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        return base64.b64decode(image_data)

    def extract_text_from_bytes(self, image_bytes: bytes) -> str:
        """Extracts text using pytesseract, easyocr, or fallback."""
        text = ""
        # Try pytesseract if available
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(img)
            if text and len(text.strip()) > 5:
                return text
        except Exception as e:
            logger.debug(f"Pytesseract not available or failed: {e}")

        # Try easyocr if available
        try:
            import easyocr

            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(image_bytes, detail=0)
            text = "\n".join(results)
            if text and len(text.strip()) > 5:
                return text
        except Exception as e:
            logger.debug(f"EasyOCR not available or failed: {e}")

        # If no OCR library is installed or extraction returned empty, return fallback informative text
        if not text:
            logger.info("No optical OCR library active. Using mock/structured image processor fallback.")
            text = "Consider the following GATE question in the uploaded image. Solve step-by-step."

        return text

    def parse_structured_question(self, raw_text: str) -> Dict[str, Any]:
        """Parses extracted raw text into structured question components."""
        cleaned_text = self.cleaner.clean_text(raw_text)

        # Detect Options (A), (B), (C), (D) or A., B., C., D.
        options = {}
        opt_pattern = r"(?:^|\n)\s*(?:\(?([A-Da-d])\)|\b([A-Da-d])[\.:\)])\s*(.*?)(?=(?:\n\s*(?:\(?[A-Da-d]\)|\b[A-Da-d][\.:\)]))|$)"
        matches = list(re.finditer(opt_pattern, cleaned_text, re.DOTALL))

        question_text = cleaned_text
        if len(matches) >= 2:
            first_opt_idx = matches[0].start()
            question_text = cleaned_text[:first_opt_idx].strip()
            for m in matches:
                key = (m.group(1) or m.group(2)).upper()
                val = m.group(3).strip()
                options[key] = val

        # Infer Question Type
        if len(options) >= 2:
            q_type = QuestionType.MCQ
        elif re.search(r"numerical|integer|round to|range", cleaned_text, re.IGNORECASE):
            q_type = QuestionType.NAT
        else:
            q_type = QuestionType.MCQ

        # Infer Subject
        subject = self._infer_subject(cleaned_text)

        return {
            "question": question_text or raw_text,
            "options": options,
            "question_type": q_type.value,
            "subject": subject.value,
            "marks": 2 if len(options) >= 4 else 1,
            "topic": "Extracted from Image",
        }

    def _infer_subject(self, text: str) -> Subject:
        """Infers subject category from keyword frequency."""
        lower = text.lower()
        if any(w in lower for w in ["paging", "semaphore", "deadlock", "fork", "scheduling", "page replacement", "tlb", "thread", "virtual memory"]):
            return Subject.OPERATING_SYSTEMS
        if any(w in lower for w in ["dfa", "nfa", "regular expression", "grammar", "turing", "decidable", "pda", "cfg", "undecidable"]):
            return Subject.THEORY_OF_COMPUTATION
        if any(w in lower for w in ["b-tree", "b+ tree", "sql", "transaction", "serializability", "relational algebra", "functional dependency", "normalization", "acid"]):
            return Subject.DBMS
        if any(w in lower for w in ["tcp", "congestion", "sliding window", "ip address", "subnet", "routing", "crc", "ethernet", "csmacd", "go-back-n"]):
            return Subject.COMPUTER_NETWORKS
        if any(w in lower for w in ["binary search", "dijkstra", "dynamic programming", "time complexity", "recurrence", "heap", "quick sort", "mst", "kruskal", "graph"]):
            return Subject.ALGORITHMS
        if any(w in lower for w in ["lalr", "slr", "syntax analysis", "parsing", "intermediate code", "three address code", "dag", "activation record"]):
            return Subject.COMPILER_DESIGN
        if any(w in lower for w in ["cache", "pipeline", "hazard", "instruction cycle", "addressing mode", "microprogramming", "cpi", "speedup"]):
            return Subject.COMPUTER_ORGANIZATION
        if any(w in lower for w in ["k-map", "boolean", "multiplexer", "decoder", "flip flop", "counter", "combinational", "logic gate"]):
            return Subject.DIGITAL_LOGIC
        if any(w in lower for w in ["eigenvalue", "probability", "bayes", "matrix", "combinatorics", "permutation", "graph coloring", "discrete"]):
            return Subject.ENGINEERING_MATHEMATICS

        return Subject.ALGORITHMS

    def process_image(self, image_data: str) -> Dict[str, Any]:
        """End-to-end extraction from base64 image data."""
        raw_bytes = self.decode_image_data(image_data)
        raw_text = self.extract_text_from_bytes(raw_bytes)
        return self.parse_structured_question(raw_text)
