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
    assert "CALYPSO" in data["service"]
    assert "quantization" in data
    assert data["features"]["multi_turn_chat"] is True


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
    assert "Conceptual Framework" in data["solution_markdown"]
    assert "Final Answer" in data["solution_markdown"]
    assert data["inference_latency_ms"] >= 0
    assert data["tokens_generated"] > 0
    assert data["structured_phases"] is not None


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
    assert any("[DONE]" in chunk for chunk in chunks)


def test_multi_turn_chat_endpoint():
    payload = {
        "messages": [
            {"role": "user", "content": "Explain why TLB miss increases access time in 2-level paging."},
        ],
        "subject": "Operating Systems",
        "topic": "Paging",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 10


def test_tool_execute_endpoint():
    payload = {
        "tool_name": "code_interpreter",
        "code": "a = 5\nb = 10\nprint(a * b)",
    }
    response = client.post("/api/tools/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["output"] == "50"
