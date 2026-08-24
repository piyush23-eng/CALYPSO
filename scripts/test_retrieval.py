import sys
import argparse
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.indexer import DualIndexManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


GATE_BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "topic": "Operating Systems",
        "query": "How is Effective Memory Access Time calculated in 2-level paging with TLB hit ratio?"
    },
    {
        "id": "Q2",
        "topic": "Database Management Systems",
        "query": "Which concurrency control protocol guarantees conflict serializability and eliminates cascading aborts?"
    },
    {
        "id": "Q3",
        "topic": "Algorithms",
        "query": "What is the worst-case time complexity of constructing a binary max-heap from an unsorted array?"
    },
    {
        "id": "Q4",
        "topic": "Operating Systems",
        "query": "Under Shortest Remaining Time First SRTF CPU scheduling how to compute average waiting time?"
    },
    {
        "id": "Q5",
        "topic": "Database Management Systems",
        "query": "What is the highest normal form satisfied when every attribute in the relation is a prime attribute?"
    },
    {
        "id": "Q6",
        "topic": "Operating Systems",
        "query": "What is Belady's Anomaly and which page replacement algorithms are immune to it?"
    },
    {
        "id": "Q7",
        "topic": "Theory of Computation",
        "query": "What is the relationship between Chomsky Hierarchy context-free grammars and pushdown automata?"
    },
    {
        "id": "Q8",
        "topic": "Operating Systems",
        "query": "How does Banker's Algorithm determine whether a state is safe or unsafe using Need matrix?"
    },
    {
        "id": "Q9",
        "topic": "Compiler Design",
        "query": "What causes shift-reduce conflicts in LR(1) and LALR(1) parsing tables?"
    },
    {
        "id": "Q10",
        "topic": "Computer Networks",
        "query": "Explain TCP congestion control phases including slow start, congestion avoidance, and fast recovery."
    }
]


def run_retrieval_suite(
    processed_dir: str = "./data/processed",
    top_candidates: int = 10,
    top_rerank: int = 3
):
    print("=" * 80)
    print("🚀 CALYPSO-RAG: HYBRID RETRIEVAL & RERANKING VISUAL INSPECTION SUITE")
    print("=" * 80)

    # Initialize Index Manager, Retriever, and Reranker
    index_manager = DualIndexManager(
        persist_dir=f"{processed_dir}/chroma_db",
        bm25_persist_path=f"{processed_dir}/bm25_index.pkl"
    )
    index_manager.load_indices()

    retriever = HybridRetriever(index_manager=index_manager, rrf_k=60)
    reranker = CrossEncoderReranker()

    for item in GATE_BENCHMARK_QUERIES:
        qid = item["id"]
        topic = item["topic"]
        query = item["query"]

        print("\n" + "─" * 80)
        print(f"[{qid}] TOPIC: {topic}")
        print(f"QUERY: \"{query}\"")
        print("─" * 80)

        # 1. Direct BM25 retrieval
        bm25_raw = index_manager.search_bm25(query=query, top_k=3)
        # 2. Direct Dense retrieval
        dense_raw = index_manager.search_dense(query=query, top_k=3)
        # 3. Hybrid RRF Fusion
        fused = retriever.retrieve(query=query, top_candidates_per_source=20, fused_top_k=top_candidates)
        # 4. Cross-Encoder Reranking
        reranked = reranker.rerank(query=query, chunks=fused, top_k=top_rerank)

        # Display side-by-side / stage-by-stage comparison
        print("\n  [1] BM25 TOP 2 (Lexical Match):")
        for i, r in enumerate(bm25_raw[:2], 1):
            snippet = r["content"].replace("\n", " ")[:90]
            print(f"      {i}. [Score: {r['score']:6.2f}] ({r['source_type']}/{r['subtopic']}) -> {snippet}...")

        print("\n  [2] DENSE TOP 2 (BGE-Small Semantic Match):")
        for i, r in enumerate(dense_raw[:2], 1):
            snippet = r["content"].replace("\n", " ")[:90]
            print(f"      {i}. [Sim:   {r['score']:6.4f}] ({r['source_type']}/{r['subtopic']}) -> {snippet}...")

        print("\n  [3] FUSED TOP 2 (RRF k=60 Ranking):")
        for i, r in enumerate(fused[:2], 1):
            snippet = r.content.replace("\n", " ")[:90]
            print(f"      {i}. [RRF:   {r.rrf_score:6.4f}] [BM25 #{r.bm25_rank or '-':>2}, Dense #{r.dense_rank or '-':>2}] -> {snippet}...")

        print("\n  [4] RERANKED TOP 3 (Cross-Encoder ms-marco-MiniLM-L-6-v2):")
        for i, r in enumerate(reranked, 1):
            snippet = r.content.replace("\n", " ")[:90]
            print(f"      🏆 {i}. [Relevance: {r.rerank_score:.4f}] [ID: {r.chunk_id}] ({r.source_file}) -> {snippet}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Hybrid Retrieval and Cross-Encoder Reranking")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Path to processed indices")
    args = parser.parse_args()

    run_retrieval_suite(processed_dir=args.processed_dir)
