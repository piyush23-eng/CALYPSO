import pytest
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.retrieval.relevance_gate import GatedRetrievalResult
from src.generation.calypso_client import CalypsoPromptBuilder, CalypsoClient
from src.generation.citation_mapper import CitationMapper, SentenceCitation, GenerationOutput


def test_prompt_builder_structure():
    """
    Verifies that RAG generation prompt enforces negative constraints and attributes context chunks.
    """
    chunks = [
        RetrievedChunk(
            chunk_id="chk_os_1",
            content="In 2-level paging, EMAT = h*(t_tlb + t_m) + (1-h)*(t_tlb + 3*t_m).",
            topic="Operating Systems",
            subtopic="Paging and EMAT",
            source_type="notes",
            source_file="os_notes.md",
            rrf_score=0.032,
            rerank_score=0.98
        )
    ]
    query = "What is the EMAT formula for 2-level paging?"
    prompt = CalypsoPromptBuilder.build_rag_prompt(query=query, chunks=chunks)

    assert "STRICT GROUNDING RULES" in prompt
    assert "The question is not covered in retrieved material" in prompt
    assert "[Chunk ID: chk_os_1" in prompt
    assert "os_notes.md" in prompt
    assert query in prompt


def test_calypso_client_fallback_mode():
    """
    Tests CalypsoClient deterministic fallback logic for offline/mock environments.
    """
    client = CalypsoClient(mock_mode=True)
    chunks = [
        RetrievedChunk(
            chunk_id="chk_algo_1",
            content="Building a max-heap takes O(n) worst-case time using Floyd's bottom-up build-heap.",
            topic="Algorithms",
            subtopic="Heap Construction",
            source_type="notes",
            source_file="algorithms_notes.md",
            rrf_score=0.031,
            rerank_score=0.95
        )
    ]
    query = "What is the time complexity of building a max heap?"
    answer = client.generate(query=query, chunks=chunks)

    assert "O(n)" in answer
    assert "Floyd's bottom-up" in answer or "heap" in answer.lower()


def test_citation_mapper_sentence_attribution():
    """
    Verifies that individual sentences in the generated answer are attributed to the correct source chunk.
    """
    chunks = [
        RetrievedChunk(
            chunk_id="chk_dbms_1",
            content="Strict 2-Phase Locking (Strict 2PL) guarantees conflict serializability and eliminates cascading aborts by holding exclusive locks until commit.",
            topic="Database Management Systems",
            subtopic="Concurrency Control",
            source_type="notes",
            source_file="dbms_notes.md",
            rrf_score=0.032,
            rerank_score=0.99
        ),
        RetrievedChunk(
            chunk_id="chk_os_1",
            content="Effective Memory Access Time (EMAT) formula in 2-level paging accounts for TLB hit ratio and memory access latency.",
            topic="Operating Systems",
            subtopic="Virtual Memory",
            source_type="notes",
            source_file="os_notes.md",
            rrf_score=0.030,
            rerank_score=0.85
        )
    ]
    
    gated_result = GatedRetrievalResult(
        original_query="Why does Strict 2PL eliminate cascading aborts?",
        effective_query="Why does Strict 2PL eliminate cascading aborts?",
        reformulation_count=0,
        max_relevance_score=0.99,
        passed_gate=True,
        is_low_confidence=False,
        chunks=chunks
    )

    answer_text = (
        "Strict 2-Phase Locking guarantees conflict serializability and prevents cascading aborts. "
        "It achieves this by requiring all exclusive write locks to be held until transaction commit or abort. "
        "Therefore, dirty reads are completely avoided in the schedule."
    )

    mapper = CitationMapper(similarity_threshold=0.60)
    output = mapper.map_citations(
        query=gated_result.original_query,
        answer_text=answer_text,
        gated_result=gated_result
    )

    assert isinstance(output, GenerationOutput)
    assert len(output.citations) >= 2
    assert all(isinstance(c, SentenceCitation) for c in output.citations)
    # The DBMS answer must cite the DBMS chunk, not the OS chunk
    assert output.citations[0].chunk_id == "chk_dbms_1"
    assert output.citations[0].source_file == "dbms_notes.md"
    assert output.citations[0].similarity_score >= 0.60
    assert output.confidence > 0.70
    assert output.is_low_confidence is False


def test_empty_context_triggers_uncovered_flag():
    """
    Tests that empty context produces low confidence flag and zero citations.
    """
    mapper = CitationMapper(similarity_threshold=0.60)
    gated_result = GatedRetrievalResult(
        original_query="Unrelated question",
        effective_query="Unrelated question",
        reformulation_count=2,
        max_relevance_score=0.10,
        passed_gate=False,
        is_low_confidence=True,
        chunks=[]
    )

    output = mapper.map_citations(
        query="Unrelated question",
        answer_text="The question is not covered in retrieved material.",
        gated_result=gated_result
    )

    assert output.is_low_confidence is True
    assert len(output.citations) == 0
    assert output.confidence == 0.0
