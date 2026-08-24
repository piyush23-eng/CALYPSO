import pytest
from src.ingestion.indexer import DualIndexManager
from src.agent.orchestrator import CalypsoAgentOrchestrator


@pytest.fixture(scope="module")
def agent_orchestrator():
    index_manager = DualIndexManager(
        persist_dir="./data/processed/chroma_db",
        bm25_persist_path="./data/processed/bm25_index.pkl"
    )
    index_manager.load_indices()
    return CalypsoAgentOrchestrator(index_manager=index_manager)


def test_agent_graph_compilation(agent_orchestrator):
    """
    Verifies that the LangGraph state machine compiles with all nodes and conditional routing edges.
    """
    assert agent_orchestrator.app is not None
    assert agent_orchestrator.graph is not None

    mermaid = agent_orchestrator.export_mermaid()
    assert "```mermaid" in mermaid
    assert "classify_query" in mermaid or "Classify" in mermaid
    assert "RelevanceCheck" in mermaid
    assert "Citations" in mermaid


def test_agent_end_to_end_query_1_os_emat(agent_orchestrator):
    """
    Integration test for GATE OS query (EMAT). Asserts non-empty final answer and citations.
    """
    query = "How is Effective Memory Access Time calculated in 2-level paging with TLB hit ratio?"
    state = agent_orchestrator.run(query=query)

    assert state["query"] == query
    assert state["subject_hint"] == "Operating Systems"
    assert len(state["rerank_results"]) > 0
    assert state["relevance_score"] >= 0.50
    assert state["passed_gate"] is True
    assert len(state["final_answer"]) > 20
    assert len(state["citations"]) > 0
    assert any(c["source_file"] in ["os_notes.md", "gate_pyq_archive.md", "gate_cs_syllabus.md"] for c in state["citations"])


def test_agent_end_to_end_query_2_dbms_strict_2pl(agent_orchestrator):
    """
    Integration test for GATE DBMS query (Strict 2PL). Asserts non-empty final answer and citations.
    """
    query = "Why does Strict 2-Phase Locking eliminate cascading aborts in database transactions?"
    state = agent_orchestrator.run(query=query)

    assert state["query"] == query
    assert state["subject_hint"] == "Database Management Systems"
    assert len(state["rerank_results"]) > 0
    assert state["relevance_score"] >= 0.50
    assert state["passed_gate"] is True
    assert len(state["final_answer"]) > 20
    assert len(state["citations"]) > 0
    assert any(c["source_file"] in ["dbms_notes.md", "gate_pyq_archive.md"] for c in state["citations"])


def test_agent_end_to_end_query_3_algo_heap(agent_orchestrator):
    """
    Integration test for GATE Algorithms query (Heap Build). Asserts non-empty final answer and citations.
    """
    query = "What is the worst-case time complexity of constructing a binary max heap from an unsorted array?"
    state = agent_orchestrator.run(query=query)

    assert state["query"] == query
    assert state["subject_hint"] == "Algorithms"
    assert len(state["rerank_results"]) > 0
    assert state["relevance_score"] >= 0.50
    assert state["passed_gate"] is True
    assert len(state["final_answer"]) > 20
    assert len(state["citations"]) > 0
    assert any(c["source_file"] in ["algorithms_notes.md", "gate_pyq_archive.md"] for c in state["citations"])
