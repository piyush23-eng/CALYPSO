import sys
import argparse
from pathlib import Path

# Ensure src in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.indexer import DualIndexManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.relevance_gate import CorrectiveRelevanceGate
from src.generation.calypso_client import CalypsoClient
from src.generation.citation_mapper import CitationMapper


GENERATION_TEST_QUERIES = [
    {
        "id": "GEN_1",
        "topic": "Operating Systems",
        "query": "How is Effective Memory Access Time (EMAT) calculated in a 2-level paging system with TLB?"
    },
    {
        "id": "GEN_2",
        "topic": "Database Management Systems",
        "query": "Why does Strict 2-Phase Locking (Strict 2PL) eliminate cascading aborts?"
    },
    {
        "id": "GEN_3",
        "topic": "Algorithms",
        "query": "What is the worst-case time complexity of constructing a binary max-heap from an unsorted array of n elements?"
    },
    {
        "id": "GEN_4",
        "topic": "Computer Networks",
        "query": "Explain the TCP congestion control mechanism during Slow Start and Fast Recovery."
    },
    {
        "id": "GEN_5",
        "topic": "General / Out-of-Domain",
        "query": "What is the capital city of France and who is the current president?"
    }
]


def run_generation_suite(processed_dir: str = "./data/processed"):
    print("=" * 85)
    print("🧠 CALYPSO-RAG: GENERATION & SENTENCE-LEVEL CITATION MAPPING TEST SUITE")
    print("=" * 85)

    index_manager = DualIndexManager(
        persist_dir=f"{processed_dir}/chroma_db",
        bm25_persist_path=f"{processed_dir}/bm25_index.pkl"
    )
    index_manager.load_indices()

    retriever = HybridRetriever(index_manager=index_manager, rrf_k=60)
    reranker = CrossEncoderReranker()
    gate = CorrectiveRelevanceGate(
        retriever=retriever,
        reranker=reranker,
        relevance_threshold=0.50,
        max_attempts=2
    )

    client = CalypsoClient(endpoint_url="https://calypso-m1rz.onrender.com", mock_mode=False)
    citation_mapper = CitationMapper(embedder=index_manager.embedder, similarity_threshold=0.60)

    for item in GENERATION_TEST_QUERIES:
        qid = item["id"]
        topic = item["topic"]
        query = item["query"]

        print("\n" + "─" * 85)
        print(f"[{qid}] TOPIC: {topic}")
        print(f"QUERY: \"{query}\"")
        print("─" * 85)

        # 1. Corrective Retrieval
        gated_result = gate.retrieve_with_gate(query=query)

        # 2. Generation via Calypso Client
        answer_text = client.generate(
            query=query,
            chunks=gated_result.chunks,
            subject=topic
        )

        # 3. Sentence-level citation mapping
        output = citation_mapper.map_citations(
            query=query,
            answer_text=answer_text,
            gated_result=gated_result
        )

        print("\n📝 GENERATED REASONING & ANSWER:")
        print(output.answer_text)

        print(f"\n📊 METRICS & CONFIDENCE:")
        print(f"  * Overall Confidence:     {output.confidence:.4f}")
        print(f"  * Low Confidence Flag:    {output.is_low_confidence}")
        print(f"  * Reformulation Count:    {output.retrieval_metadata.get('reformulation_count', 0)}")
        print(f"  * Citation Coverage:      {output.retrieval_metadata.get('citation_coverage', 0.0) * 100:.1f}%")

        print(f"\n🔗 SENTENCE-LEVEL CITATIONS ({len(output.citations)} Attributions):")
        if output.citations:
            for c_idx, cit in enumerate(output.citations, 1):
                print(f"  [{c_idx}] Sentence: \"{cit.sentence[:75]}...\"")
                print(f"      └── Cites: [{cit.chunk_id}] in {cit.source_file} (Cosine Sim: {cit.similarity_score:.4f})")
        else:
            print("  (No direct citations matched threshold >= 0.60)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Generation and Citation Mapping")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Path to processed indices")
    args = parser.parse_args()

    run_generation_suite(processed_dir=args.processed_dir)
