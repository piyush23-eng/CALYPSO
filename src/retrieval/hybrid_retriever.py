from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import concurrent.futures
from src.ingestion.indexer import DualIndexManager


class RetrievedChunk(BaseModel):
    """
    Structured representation of a chunk retrieved through the hybrid RAG pipeline.
    Preserves provenance, individual lexical & dense scores, and fused ranking metrics.
    """
    chunk_id: str = Field(description="Unique deterministic chunk ID")
    content: str = Field(description="Normalized textual content of the chunk")
    topic: str = Field(description="Primary GATE CS subject")
    subtopic: str = Field(description="Specific concept or question header")
    source_type: str = Field(description="Origin category: syllabus, pyq, or notes")
    source_file: str = Field(description="Source filename")
    bm25_score: Optional[float] = Field(default=None, description="Raw BM25 Okapi lexical score")
    bm25_rank: Optional[int] = Field(default=None, description="Rank in BM25 search (1-indexed)")
    dense_score: Optional[float] = Field(default=None, description="Cosine similarity score from dense vector index")
    dense_rank: Optional[int] = Field(default=None, description="Rank in Dense vector search (1-indexed)")
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder joint relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context and telemetry")


class HybridRetriever:
    """
    Production Hybrid Retriever for GATE CS Knowledge Base.
    
    Combines:
    1. BM25 Lexical Index (keyword, exact symbol, formula precision)
    2. Dense Vector Index (semantic similarity via BAAI/bge-small-en-v1.5)
    3. Custom Reciprocal Rank Fusion (RRF) with constant k=60 implemented from scratch.
    """

    def __init__(self, index_manager: DualIndexManager, rrf_k: int = 60):
        self.index_manager = index_manager
        self.rrf_k = rrf_k

    def _search_bm25_wrapper(self, query: str, top_k: int, topic_filter: Optional[str]) -> List[Dict[str, Any]]:
        return self.index_manager.search_bm25(query=query, top_k=top_k, topic_filter=topic_filter)

    def _search_dense_wrapper(self, query: str, top_k: int, topic_filter: Optional[str]) -> List[Dict[str, Any]]:
        return self.index_manager.search_dense(query=query, top_k=top_k, topic_filter=topic_filter)

    def compute_rrf_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[RetrievedChunk]:
        """
        Calculates Reciprocal Rank Fusion (RRF) from scratch.
        
        Formula:
            RRF_Score(d) = \\sum_{m \\in M} \\frac{1}{k + r_m(d)}
        where:
            - M = {BM25, Dense}
            - r_m(d) is the 1-based rank of document d in ranking list m.
            - k is the smoothing parameter (default 60).
        """
        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        bm25_rank_map: Dict[str, Tuple[int, float]] = {}
        dense_rank_map: Dict[str, Tuple[int, float]] = {}

        # 1. Process BM25 rankings
        for rank_idx, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rank = rank_idx + 1
            score = float(item["score"])
            bm25_rank_map[cid] = (rank, score)
            chunk_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 2. Process Dense rankings
        for rank_idx, item in enumerate(dense_results):
            cid = item["chunk_id"]
            rank = rank_idx + 1
            score = float(item["score"])
            dense_rank_map[cid] = (rank, score)
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by fused RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)

        fused_chunks: List[RetrievedChunk] = []
        for cid in sorted_cids[:top_k]:
            data = chunk_lookup[cid]
            b_rank, b_score = bm25_rank_map.get(cid, (None, None))
            d_rank, d_score = dense_rank_map.get(cid, (None, None))

            fused_chunks.append(RetrievedChunk(
                chunk_id=cid,
                content=data["content"],
                topic=data["topic"],
                subtopic=data["subtopic"],
                source_type=data["source_type"],
                source_file=data["source_file"],
                bm25_score=b_score,
                bm25_rank=b_rank,
                dense_score=d_score,
                dense_rank=d_rank,
                rrf_score=round(fused_scores[cid], 6),
                metadata={
                    "retrieval_method": "hybrid_rrf",
                    "found_in_bm25": b_rank is not None,
                    "found_in_dense": d_rank is not None
                }
            ))

        return fused_chunks

    def retrieve(
        self,
        query: str,
        top_candidates_per_source: int = 20,
        fused_top_k: int = 10,
        topic_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Executes parallel BM25 and Dense vector search, then fuses results via RRF.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_bm25 = executor.submit(
                self._search_bm25_wrapper, query, top_candidates_per_source, topic_filter
            )
            future_dense = executor.submit(
                self._search_dense_wrapper, query, top_candidates_per_source, topic_filter
            )

            bm25_results = future_bm25.result()
            dense_results = future_dense.result()

        return self.compute_rrf_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            top_k=fused_top_k
        )
