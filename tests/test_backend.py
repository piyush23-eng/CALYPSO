"""Unit and integration tests for FastAPI backend."""

import pytest
from starlette.testclient import TestClient
from src.backend.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "gate-cs-doubt-solver"
    assert "quantization" in data


def test_solve_endpoint_mcq():
    payload = {
        "subject": "Algorithms",
        "topic": "Asymptotic Analysis",
        "question_type": "MCQ",
        "marks": 1,
        "question": "What is the time complexity of binary search on a sorted array of size n?",
        "options": {
            "A": "O(log n)",
            "B": "O(n)",
            "C": "O(n log n)",
            "D": "O(1)",
        },
        "model_type": "finetuned",
    }
    response = client.post("/api/solve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "solution_markdown" in data
    assert "### 1. Conceptual Framework" in data["solution_markdown"]
    assert "### 4. Final Answer" in data["solution_markdown"]
    assert data["inference_latency_ms"] >= 0
    assert data["tokens_generated"] > 0


def test_compare_endpoint():
    payload = {
        "subject": "Operating Systems",
        "topic": "Paging",
        "question_type": "MCQ",
        "marks": 2,
        "question": "Which of the following page replacement algorithms suffers from Belady's anomaly?",
        "options": {
            "A": "FIFO",
            "B": "LRU",
            "C": "OPT",
            "D": "LFU with stack property",
        },
    }
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "base_model_result" in data
    assert "finetuned_model_result" in data
    assert "quality_delta" in data
    assert "Qwen2.5-1.5B" in data["base_model_result"]["model_name"]
    assert "GATE-CS-Qwen" in data["finetuned_model_result"]["model_name"]


def test_streaming_endpoint():
    payload = {
        "subject": "DBMS",
        "topic": "Normalization",
        "question_type": "MCQ",
        "marks": 1,
        "question": "Is 3NF strictly stronger than 2NF?",
        "options": {"A": "Yes", "B": "No"},
        "model_type": "finetuned",
    }
    response = client.post("/api/solve/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    chunks = list(response.iter_lines())
    assert len(chunks) > 0
    # Must contain data payload and [DONE] marker
    assert any("[DONE]" in chunk for chunk in chunks)
