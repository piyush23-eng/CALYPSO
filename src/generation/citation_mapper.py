import re
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.retrieval.hybrid_retriever import RetrievedChunk
from src.retrieval.relevance_gate import GatedRetrievalResult


class SentenceCitation(BaseModel):
    """
    Sentence-level semantic attribution mapping.
    Links an individual sentence of the generated answer back to the exact source chunk and file.
    """
    sentence: str = Field(description="Individual sentence from the answer")
    chunk_id: str = Field(description="Cited chunk ID")
    source_file: str = Field(description="Origin document filename")
    topic: str = Field(description="GATE Subject / Topic of the cited chunk")
    similarity_score: float = Field(description="Cosine similarity between sentence and cited chunk")


class GenerationOutput(BaseModel):
    """
    Standard output schema for CALYPSO-RAG generation phase.
    """
    query: str
    answer_text: str
    citations: List[SentenceCitation] = Field(default_factory=list)
    confidence: float
    is_low_confidence: bool
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationMapper:
    """
    Computes sentence-level semantic attribution using dense embeddings (BAAI/bge-small-en-v1.5).
    Threshold: 0.60 cosine similarity for valid attribution.
    """

    def __init__(
        self,
        embedder: Optional[SentenceTransformer] = None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.60
    ):
        self._embedder = embedder
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Splits text into distinct sentences while preserving LaTeX math delimiters and bullet points.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        # Split on sentence ending punctuation followed by space or newline, while avoiding initials / decimals
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\$\*\#\-])', cleaned)
        
        sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Ignore headers or markdown decoration without semantic content
            if len(s_clean) >= 15:
                sentences.append(s_clean)

        return sentences if sentences else [cleaned]

    def map_citations(
        self,
        query: str,
        answer_text: str,
        gated_result: GatedRetrievalResult
    ) -> GenerationOutput:
        """
        Calculates sentence-level semantic attribution against retrieved context chunks.
        """
        chunks = gated_result.chunks
        sentences = self.split_sentences(answer_text)

        if not chunks or not sentences:
            return GenerationOutput(
                query=query,
                answer_text=answer_text,
                citations=[],
                confidence=0.0,
                is_low_confidence=True,
                retrieval_metadata={
                    "reformulation_count": gated_result.reformulation_count,
                    "max_relevance_score": gated_result.max_relevance_score,
                    "num_retrieved_chunks": len(chunks)
                }
            )

        # Generate embeddings for answer sentences and retrieved chunks
        sentence_embeddings = self.embedder.encode(sentences, normalize_embeddings=True)
        chunk_texts = [c.content for c in chunks]
        chunk_embeddings = self.embedder.encode(chunk_texts, normalize_embeddings=True)

        # Compute cosine similarity matrix: (num_sentences, num_chunks)
        similarity_matrix = np.dot(sentence_embeddings, chunk_embeddings.T)

        citations: List[SentenceCitation] = []
        cited_sentence_count = 0

        for s_idx, sentence in enumerate(sentences):
            sims = similarity_matrix[s_idx]
            best_chunk_idx = int(np.argmax(sims))
            best_sim = float(sims[best_chunk_idx])

            if best_sim >= self.similarity_threshold:
                cited_chunk = chunks[best_chunk_idx]
                citations.append(SentenceCitation(
                    sentence=sentence,
                    chunk_id=cited_chunk.chunk_id,
                    source_file=cited_chunk.source_file,
                    topic=cited_chunk.topic,
                    similarity_score=round(best_sim, 4)
                ))
                cited_sentence_count += 1

        citation_coverage = cited_sentence_count / len(sentences) if sentences else 0.0
        overall_confidence = round(float(0.6 * gated_result.max_relevance_score + 0.4 * citation_coverage), 4)

        return GenerationOutput(
            query=query,
            answer_text=answer_text,
            citations=citations,
            confidence=overall_confidence,
            is_low_confidence=gated_result.is_low_confidence or overall_confidence < 0.45,
            retrieval_metadata={
                "effective_query": gated_result.effective_query,
                "reformulation_count": gated_result.reformulation_count,
                "max_relevance_score": gated_result.max_relevance_score,
                "citation_coverage": round(citation_coverage, 4),
                "num_sentences": len(sentences),
                "num_cited_sentences": cited_sentence_count
            }
        )
