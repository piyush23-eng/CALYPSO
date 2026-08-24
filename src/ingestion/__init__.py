"""Ingestion and Chunking Module for GATE CS Knowledge Base"""
from src.ingestion.chunker import TopicAwareChunker, DocumentChunk
from src.ingestion.indexer import DualIndexManager

__all__ = ["TopicAwareChunker", "DocumentChunk", "DualIndexManager"]
