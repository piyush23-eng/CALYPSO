import sys
import os
import json
from pathlib import Path

# Ensure src in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.indexer import DualIndexManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.relevance_gate import CorrectiveRelevanceGate


CRAG_TEST_CASES = [
    # --- 3 Clear Queries (Expected to Pass on First Try) ---
    {
        "id": "CLEAR_1",
        "category": "Clear Query",
        "query": "How is Effective Memory Access Time calculated in 2-level paging with TLB hit ratio?",
        "expected_reformulation": False
    },
    {
        "id": "CLEAR_2",
        "category": "Clear Query",
        "query": "Why does Strict 2PL eliminate cascading aborts and guarantee conflict serializability?",
        "expected_reformulation": False
    },
    {
        "id": "CLEAR_3",
        "category": "Clear Query",
        "query": "What is the worst case time complexity of building a binary max heap from unsorted array?",
        "expected_reformulation": False
    },
    # --- 2 Vague / Noisy Queries (Expected to Trigger Reformulation Loop) ---
    {
        "id": "VAGUE_1",
        "category": "Vague / Colloquial Query",
        "query": "slow speed when network packet drops",
        "expected_reformulation": True
    },
    {
        "id": "VAGUE_2",
        "category": "Vague / Colloquial Query",
        "query": "time speed heap",
        "expected_reformulation": True
    }
]


def run_crag_demonstration():
    print("=" * 85)
    print("🔄 CALYPSO-RAG: CORRECTIVE-RAG (CRAG) RELEVANCE GATE DEMONSTRATION")
    print("=" * 85)

    log_path = "./data/eval/crag_reformulation_log.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)  # Fresh run log

    index_manager = DualIndexManager(
        persist_dir="./data/processed/chroma_db",
        bm25_persist_path="./data/processed/bm25_index.pkl"
    )
    index_manager.load_indices()

    retriever = HybridRetriever(index_manager=index_manager, rrf_k=60)
    reranker = CrossEncoderReranker()
    gate = CorrectiveRelevanceGate(
        retriever=retriever,
        reranker=reranker,
        relevance_threshold=0.50,
        max_attempts=2,
        log_file_path=log_path
    )

    print(f"\nConfiguration: Relevance Threshold = {gate.relevance_threshold}, Max Reformulations = {gate.max_attempts}\n")

    for case in CRAG_TEST_CASES:
        cid = case["id"]
        cat = case["category"]
        query = case["query"]

        print("─" * 85)
        print(f"[{cid}] Category: {cat}")
        print(f"Original Query: \"{query}\"")

        result = gate.retrieve_with_gate(query=query)

        print(f"  * Passed Gate:         {result.passed_gate}")
        print(f"  * Reformulation Count: {result.reformulation_count}")
        print(f"  * Max Relevance Score: {result.max_relevance_score:.4f}")
        print(f"  * Low Confidence Flag: {result.is_low_confidence}")
        print(f"  * Effective Query:     \"{result.effective_query[:80]}...\"")

        if result.reformulation_history:
            print("  * 📝 Reformulation Audit Trail:")
            for log in result.reformulation_history:
                print(f"      [Attempt {log.attempt_number}] Score {log.max_relevance_score_before:.4f} ──▶ {log.max_relevance_score_after:.4f} | Method: {log.reformulation_method}")
                print(f"      Rewritten: \"{log.rewritten_query[:75]}...\"")

        print("  * 🏆 Top Retrieved Evidence Chunk:")
        if result.chunks:
            top_c = result.chunks[0]
            print(f"      [{top_c.chunk_id}] ({top_c.source_file}) Relevance: {top_c.rerank_score:.4f}")
            print(f"      Content: {top_c.content.replace(chr(10), ' ')[:100]}...")
        print()

    # Verify JSONL log file
    print("=" * 85)
    print(f"📄 AUDIT TRAIL LOG FILE: {log_path}")
    print("=" * 85)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Total Logged Reformulation Attempts: {len(lines)}")
        for idx, line in enumerate(lines, 1):
            entry = json.loads(line)
            print(f"{idx}. [{entry['timestamp']}] \"{entry['original_query']}\" -> \"{entry['rewritten_query'][:50]}...\" (Score: {entry['max_relevance_score_before']} -> {entry['max_relevance_score_after']})")


if __name__ == "__main__":
    run_crag_demonstration()
