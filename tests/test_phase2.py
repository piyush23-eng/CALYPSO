import pytest
import math
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.reranker import CrossEncoderReranker
from src.ingestion.indexer import DualIndexManager


def test_rrf_algorithm_from_scratch():
    """
    Validates that our from-scratch RRF computation matches the exact formula:
    RRF(d) = sum( 1 / (k + rank) ) with k = 60.
    """
    retriever = HybridRetriever(index_manager=None, rrf_k=60)

    # Mock BM25 result list
    bm25_list = [
        {"chunk_id": "doc_A", "content": "OS Paging", "topic": "OS", "subtopic": "Paging", "source_type": "notes", "source_file": "os.md", "score": 12.5},
        {"chunk_id": "doc_B", "content": "DBMS 3NF", "topic": "DBMS", "subtopic": "3NF", "source_type": "notes", "source_file": "dbms.md", "score": 8.1},
        {"chunk_id": "doc_C", "content": "Algorithms Heap", "topic": "Algorithms", "subtopic": "Heaps", "source_type": "notes", "source_file": "algo.md", "score": 5.2}
    ]

    # Mock Dense result list (doc_B is rank 1, doc_A is rank 2, doc_D is rank 3)
    dense_list = [
        {"chunk_id": "doc_B", "content": "DBMS 3NF", "topic": "DBMS", "subtopic": "3NF", "source_type": "notes", "source_file": "dbms.md", "score": 0.92},
        {"chunk_id": "doc_A", "content": "OS Paging", "topic": "OS", "subtopic": "Paging", "source_type": "notes", "source_file": "os.md", "score": 0.85},
        {"chunk_id": "doc_D", "content": "Networks TCP", "topic": "CN", "subtopic": "TCP", "source_type": "notes", "source_file": "cn.md", "score": 0.70}
    ]

    fused = retriever.compute_rrf_fusion(bm25_results=bm25_list, dense_results=dense_list, top_k=4)

    # Expected RRF scores with k=60:
    # doc_A: BM25 rank 1 (1/61 = 0.0163934), Dense rank 2 (1/62 = 0.0161290) -> Total = 0.032522
    # doc_B: BM25 rank 2 (1/62 = 0.0161290), Dense rank 1 (1/61 = 0.0163934) -> Total = 0.032522 (tie)
    # doc_C: BM25 rank 3 (1/63 = 0.0158730)
    # doc_D: Dense rank 3 (1/63 = 0.0158730)

    expected_score_ab = round((1.0 / 61) + (1.0 / 62), 6)
    expected_score_c = round(1.0 / 63, 6)

    assert len(fused) == 4
    top_cids = [fused[0].chunk_id, fused[1].chunk_id]
    assert "doc_A" in top_cids and "doc_B" in top_cids
    assert fused[0].rrf_score == expected_score_ab
    assert fused[1].rrf_score == expected_score_ab
    assert fused[2].rrf_score == expected_score_c

    # Check individual ranks and scores are preserved
    chunk_a = next(c for c in fused if c.chunk_id == "doc_A")
    assert chunk_a.bm25_rank == 1
    assert chunk_a.bm25_score == 12.5
    assert chunk_a.dense_rank == 2
    assert chunk_a.dense_score == 0.85


def test_cross_encoder_reranker_scoring():
    """
    Tests CrossEncoder scoring and downsampling to top-k.
    """
    reranker = CrossEncoderReranker()
    
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            content="In 2-level paging, Effective Memory Access Time EMAT = h * (t_tlb + t_m) + (1-h)*(t_tlb + 2*t_m).",
            topic="Operating Systems",
            subtopic="Paging",
            source_type="notes",
            source_file="os.md",
            rrf_score=0.032
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            content="Max heap construction takes O(n) worst-case time using bottom-up build-heap.",
            topic="Algorithms",
            subtopic="Heaps",
            source_type="notes",
            source_file="algo.md",
            rrf_score=0.031
        ),
        RetrievedChunk(
            chunk_id="chunk_3",
            content="Strict 2PL prevents cascading aborts by holding exclusive locks until transaction commit.",
            topic="DBMS",
            subtopic="Transactions",
            source_type="notes",
            source_file="dbms.md",
            rrf_score=0.030
        )
    ]

    query = "How to calculate EMAT for two level paging with TLB hit ratio?"
    reranked = reranker.rerank(query=query, chunks=chunks, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "chunk_1"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score
    assert 0.0 <= reranked[0].rerank_score <= 1.0


def test_hybrid_retrieval_integration():
    """
    End-to-end integration test against the real ingested indices.
    """
    index_manager = DualIndexManager(
        persist_dir="./data/processed/chroma_db",
        bm25_persist_path="./data/processed/bm25_index.pkl"
    )
    retriever = HybridRetriever(index_manager=index_manager, rrf_k=60)
    reranker = CrossEncoderReranker()

    query = "What is Belady's Anomaly and FIFO page replacement?"
    
    # 1. Hybrid retrieval (top 10)
    fused_chunks = retriever.retrieve(query=query, top_candidates_per_source=20, fused_top_k=10)
    assert len(fused_chunks) > 0
    assert all(c.rrf_score is not None for c in fused_chunks)

    # 2. Cross-Encoder reranking (top 3)
    reranked_chunks = reranker.rerank(query=query, chunks=fused_chunks, top_k=3)
    assert len(reranked_chunks) <= 3
    assert len(reranked_chunks) > 0
    
    # Top result should mention Belady or FIFO or Page Replacement
    top_content = reranked_chunks[0].content.lower()
    assert "belady" in top_content or "fifo" in top_content or "page" in top_content
