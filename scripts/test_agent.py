import sys
import argparse
from pathlib import Path

# Ensure src in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.orchestrator import CalypsoAgentOrchestrator
from src.ingestion.indexer import DualIndexManager


TEST_QUERIES = [
    {
        "id": "AGENT_1",
        "query": "How is Effective Memory Access Time calculated in 2-level paging with TLB hit ratio?"
    },
    {
        "id": "AGENT_2",
        "query": "Why does Strict 2-Phase Locking eliminate cascading aborts in database transactions?"
    },
    {
        "id": "AGENT_3",
        "query": "What is the worst-case time complexity of constructing a binary max heap from an unsorted array?"
    }
]


def run_agent_demonstration(processed_dir: str = "./data/processed"):
    print("=" * 85)
    print("🤖 CALYPSO-RAG: LANGGRAPH AGENTIC STATE MACHINE ORCHESTRATION")
    print("=" * 85)

    index_manager = DualIndexManager(
        persist_dir=f"{processed_dir}/chroma_db",
        bm25_persist_path=f"{processed_dir}/bm25_index.pkl"
    )
    index_manager.load_indices()

    orchestrator = CalypsoAgentOrchestrator(index_manager=index_manager)

    print("\n--- ARCHITECTURE MERMAID FLOWCHART ---")
    print(orchestrator.export_mermaid())
    print("-" * 85)

    for item in TEST_QUERIES:
        qid = item["id"]
        query = item["query"]

        print(f"\n[{qid}] RUNNING AGENT FOR QUERY: \"{query}\"")
        print("─" * 85)

        final_state = orchestrator.run(query=query)

        print(f"  * Classified Subject:   {final_state.get('subject_hint')}")
        print(f"  * Passed Gate:          {final_state.get('passed_gate')}")
        print(f"  * Max Relevance Score:  {final_state.get('relevance_score'):.4f}")
        print(f"  * Reformulation Count:  {final_state.get('reformulation_count')}")
        print(f"  * Low Confidence Flag:  {final_state.get('is_low_confidence')}")
        print(f"  * Top Chunks Retrieved: {len(final_state.get('rerank_results', []))}")
        print(f"  * Citations Attached:   {len(final_state.get('citations', []))}")

        print("\n  📝 FINAL ANSWER:")
        print(f"  {final_state.get('final_answer')[:300]}...")

        if final_state.get("citations"):
            print("\n  🔗 SENTENCE CITATIONS:")
            for idx, cit in enumerate(final_state["citations"][:2], 1):
                print(f"    [{idx}] \"{cit['sentence'][:60]}...\" ──▶ [{cit['chunk_id']}] ({cit['source_file']})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LangGraph Agent Orchestration")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Path to processed indices")
    args = parser.parse_args()

    run_agent_demonstration(processed_dir=args.processed_dir)
