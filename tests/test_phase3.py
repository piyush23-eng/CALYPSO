import pytest
import json
import os
from pathlib import Path
from src.ingestion.indexer import DualIndexManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.relevance_gate import CorrectiveRelevanceGate, GatedRetrievalResult


@pytest.fixture(scope="module")
def shared_gate():
    index_manager = DualIndexManager(
        persist_dir="./data/processed/chroma_db",
        bm25_persist_path="./data/processed/bm25_index.pkl"
    )
    index_manager.load_indices()
    retriever = HybridRetriever(index_manager=index_manager, rrf_k=60)
    reranker = CrossEncoderReranker()
    test_log = "./data/eval/test_crag_log.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)
        
    gate = CorrectiveRelevanceGate(
        retriever=retriever,
        reranker=reranker,
        relevance_threshold=0.50,
        max_attempts=2,
        log_file_path=test_log
    )
    yield gate
    # Teardown
    if os.path.exists(test_log):
        os.remove(test_log)


def test_clear_query_passes_first_try(shared_gate):
    """
    A precise GATE CS query must pass the relevance gate immediately without reformulation.
    """
    query = "How is Effective Memory Access Time calculated in 2-level paging with TLB hit ratio?"
    result = shared_gate.retrieve_with_gate(query=query)

    assert isinstance(result, GatedRetrievalResult)
    assert result.passed_gate is True
    assert result.is_low_confidence is False
    assert result.reformulation_count == 0
    assert result.max_relevance_score >= 0.50
    assert len(result.chunks) > 0
    assert len(result.reformulation_history) == 0


def test_vague_query_triggers_reformulation(shared_gate):
    """
    A vague query with low initial score must trigger query reformulation and achieve a higher score.
    """
    query = "slow speed when network packet drops"
    result = shared_gate.retrieve_with_gate(query=query)

    assert isinstance(result, GatedRetrievalResult)
    # The vague query should have triggered at least 1 reformulation
    assert result.reformulation_count >= 1
    assert len(result.reformulation_history) >= 1
    
    # Check that score improved after reformulation
    first_attempt = result.reformulation_history[0]
    assert first_attempt.max_relevance_score_after > first_attempt.max_relevance_score_before
    assert result.max_relevance_score >= 0.50
    assert result.passed_gate is True


def test_completely_off_topic_query_returns_low_confidence(shared_gate):
    """
    A completely off-topic query should exhaust max attempts (2) and return is_low_confidence=True.
    """
    off_topic_query = "recipe for chocolate chip banana pancakes with maple syrup"
    result = shared_gate.retrieve_with_gate(query=off_topic_query)

    assert isinstance(result, GatedRetrievalResult)
    assert result.reformulation_count == 2
    assert len(result.reformulation_history) == 2
    assert result.passed_gate is False
    assert result.is_low_confidence is True


def test_jsonl_log_persistence(shared_gate):
    """
    Verifies that reformulation attempts are faithfully logged to the JSONL audit file.
    """
    log_file = shared_gate.log_file_path
    assert log_file.exists()

    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 3  # At least 1 from vague test, 2 from off-topic test
    for line in lines:
        data = json.loads(line)
        assert "original_query" in data
        assert "rewritten_query" in data
        assert "max_relevance_score_before" in data
        assert "max_relevance_score_after" in data
        assert "passed_gate" in data
