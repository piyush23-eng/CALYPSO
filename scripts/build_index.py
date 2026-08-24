import os
import sys
import argparse
from pathlib import Path
from typing import List

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunker import TopicAwareChunker, DocumentChunk
from src.ingestion.indexer import DualIndexManager


def build_knowledge_base(raw_dir: str = "./data/raw", processed_dir: str = "./data/processed"):
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory {raw_dir} does not exist.")

    files = list(raw_path.glob("*.md")) + list(raw_path.glob("*.txt"))
    if not files:
        print(f"No documents found in {raw_dir}. Please place .md or .txt files in data/raw/")
        return

    print(f"Found {len(files)} raw files to ingest from {raw_dir}:")
    for f in files:
        print(f" - {f.name}")

    chunker = TopicAwareChunker(max_chars=1200, min_chars=80, overlap_chars=100)
    all_chunks: List[DocumentChunk] = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = chunker.chunk_document(content=content, source_file=file_path.name)
        all_chunks.extend(chunks)
        print(f"Generated {len(chunks)} chunks from {file_path.name}")

    print(f"\nTotal Chunks across all documents: {len(all_chunks)}")
    
    # Subject breakdown
    topic_counts = {}
    for c in all_chunks:
        topic_counts[c.topic] = topic_counts.get(c.topic, 0) + 1
    
    print("\nChunk distribution by GATE Subject:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  * {topic}: {count} chunks")

    # Build Dual Index
    indexer = DualIndexManager(
        persist_dir=f"{processed_dir}/chroma_db",
        bm25_persist_path=f"{processed_dir}/bm25_index.pkl"
    )
    indexer.build_and_save(all_chunks)
    print("\n✅ Knowledge Base Ingestion & Dual Indexing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GATE CS Dual Index (BM25 + ChromaDB Dense)")
    parser.add_argument("--raw_dir", type=str, default="./data/raw", help="Path to raw markdown files")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Path to save processed indices")
    args = parser.parse_args()

    build_knowledge_base(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
