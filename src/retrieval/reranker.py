from typing import List, Optional
import math
from sentence_transformers import CrossEncoder
from src.retrieval.hybrid_retriever import RetrievedChunk


class CrossEncoderReranker:
    """
    Reranker using a Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2).
    
    Unlike bi-encoders (which encode query and document independently), the Cross-Encoder
    performs full cross-attention over all token pairs between query and document text,
    providing high-precision relevance assessment for GATE CS problems.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            print(f"Loading Cross-Encoder reranker: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Applies standard sigmoid to map unbounded logits to [0, 1] probability range."""
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 3
    ) -> List[RetrievedChunk]:
        """
        Reranks a list of candidate RetrievedChunks against the query and returns top_k.
        """
        if not chunks:
            return []

        # Prepare pairs for joint cross-attention
        pairs = [[query, chunk.content] for chunk in chunks]
        
        raw_scores = self.model.predict(pairs)

        reranked_chunks: List[RetrievedChunk] = []
        for idx, score in enumerate(raw_scores):
            chunk = chunks[idx].model_copy(deep=True)
            norm_score = round(self._sigmoid(float(score)), 4)
            chunk.rerank_score = norm_score
            chunk.metadata["raw_cross_encoder_logit"] = round(float(score), 4)
            chunk.metadata["rerank_model"] = self.model_name
            reranked_chunks.append(chunk)

        # Sort by rerank score descending
        reranked_chunks.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else 0.0, reverse=True)
        return reranked_chunks[:top_k]
