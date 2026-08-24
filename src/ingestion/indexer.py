import os
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.ingestion.chunker import DocumentChunk


class DualIndexManager:
    """
    Manages dual-index ingestion for GATE CS Knowledge Base:
    1. Lexical BM25 Index (for exact keyword/formula/symbol matches)
    2. Dense Vector Index via BAAI/bge-small-en-v1.5 in persistent ChromaDB (for semantic similarity)
    """

    def __init__(
        self,
        persist_dir: str = "./data/processed/chroma_db",
        bm25_persist_path: str = "./data/processed/bm25_index.pkl",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "calypso_gate_kb"
    ):
        self.persist_dir = Path(persist_dir)
        self.bm25_persist_path = Path(bm25_persist_path)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        self.persist_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Lazy loaded components
        self._embedder: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Any] = None
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[DocumentChunk] = []

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"Loading dense embedding model: {self.embedding_model_name}...")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Fast alphanumeric + math symbol tokenizer for BM25
        return [tok.lower() for tok in text.replace("\n", " ").split(" ") if tok.strip()]

    def build_and_save(self, chunks: List[DocumentChunk]):
        """
        Builds both BM25 and ChromaDB Dense indices from document chunks.
        """
        self._chunks = chunks
        if not chunks:
            raise ValueError("No chunks provided to build indices.")

        print(f"Building dual indices over {len(chunks)} chunks...")
        
        # 1. Build BM25 Index
        tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        bm25_payload = {
            "bm25": self._bm25,
            "chunks": [c.model_dump() for c in chunks]
        }
        with open(self.bm25_persist_path, "wb") as f:
            pickle.dump(bm25_payload, f)
        print(f"✅ Saved BM25 index to {self.bm25_persist_path}")

        # 2. Build ChromaDB Dense Vector Index
        # Wipe existing collection to prevent duplicate accumulation on rebuilds
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "topic": c.topic,
                "subtopic": c.subtopic,
                "source_type": c.source_type,
                "source_file": c.source_file
            }
            for c in chunks
        ]

        print("Generating dense embeddings (bge-small-en-v1.5)...")
        embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"✅ Added {len(chunks)} chunks to ChromaDB collection '{self.collection_name}' at {self.persist_dir}")

    def load_indices(self):
        """
        Loads pre-built BM25 and connects to persistent ChromaDB collection.
        """
        if not self.bm25_persist_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {self.bm25_persist_path}. Run build_index.py first.")

        with open(self.bm25_persist_path, "rb") as f:
            data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunks = [DocumentChunk(**c) for c in data["chunks"]]

        # Ensure collection is loaded
        _ = self.collection
        print(f"Loaded BM25 index ({len(self._chunks)} chunks) and connected to ChromaDB.")

    def search_bm25(self, query: str, top_k: int = 20, topic_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index with optional topic metadata filter.
        """
        if self._bm25 is None:
            self.load_indices()

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk = self._chunks[idx]
            if topic_filter and chunk.topic.lower() != topic_filter.lower():
                continue
            scored_chunks.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "topic": chunk.topic,
                "subtopic": chunk.subtopic,
                "source_type": chunk.source_type,
                "source_file": chunk.source_file,
                "score": float(score)
            })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def search_dense(self, query: str, top_k: int = 20, topic_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB vector store with cosine similarity.
        """
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        
        where_clause = {"topic": topic_filter} if topic_filter else None
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_clause
        )

        dense_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                # Chroma cosine distance is in [0, 2], similarity = 1 - distance
                dist = results["distances"][0][i]
                sim_score = 1.0 - dist
                meta = results["metadatas"][0][i]
                dense_results.append({
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "topic": meta["topic"],
                    "subtopic": meta["subtopic"],
                    "source_type": meta["source_type"],
                    "source_file": meta["source_file"],
                    "score": float(sim_score)
                })

        return dense_results
