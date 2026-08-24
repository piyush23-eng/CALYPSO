import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Ensure src in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.indexer import DualIndexManager
from src.agent.orchestrator import CalypsoAgentOrchestrator
from src.evaluation.evaluator import EvalItem, RAGEvaluator, EvaluationSummary


def run_full_evaluation(
    dataset_path: str = "./data/eval/eval_dataset.json",
    results_path: str = "./data/eval/results.json",
    summary_md_path: str = "./data/eval/eval_summary.md",
    processed_dir: str = "./data/processed"
):
    print("=" * 85)
    print("📊 CALYPSO-RAG: COMPREHENSIVE RAGAS EVALUATION HARNESS (20 GATE CS BENCHMARKS)")
    print("=" * 85)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    eval_items = [EvalItem(**item) for item in raw_items]
    print(f"Loaded {len(eval_items)} benchmark evaluation items from {dataset_path}.")

    index_manager = DualIndexManager(
        persist_dir=f"{processed_dir}/chroma_db",
        bm25_persist_path=f"{processed_dir}/bm25_index.pkl"
    )
    index_manager.load_indices()

    orchestrator = CalypsoAgentOrchestrator(index_manager=index_manager)
    evaluator = RAGEvaluator(embedder=index_manager.embedder)

    agent_states = []
    print("\n🚀 Executing LangGraph Agent across all 20 benchmark questions...")

    for item in tqdm(eval_items, desc="Evaluating"):
        state = orchestrator.run(query=item.question)
        agent_states.append(state)

    print("\n📈 Computing RAG Metrics (Context Precision, Context Recall, Faithfulness, Answer Relevance)...")
    summary = evaluator.evaluate_dataset(
        eval_items=eval_items,
        agent_states=agent_states,
        target_threshold=0.75
    )

    # Save results.json
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))
    print(f"💾 Full evaluation JSON saved to: {results_path}")

    # Generate Markdown summary table
    md_lines = [
        "# 📊 CALYPSO-RAG: Evaluation Report",
        "",
        f"**Total Benchmark Questions**: {summary.total_questions}  ",
        f"**Target Quality Threshold**: {summary.target_threshold} (75%)  ",
        f"**All Targets Met**: {'✅ YES' if summary.all_targets_met else '❌ NO'}  ",
        "",
        "## 🎯 Aggregate Mean Metrics",
        "",
        "| Metric | Mean Score | Target | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Context Precision** | **{summary.mean_context_precision:.4f}** | \u2265 0.75 | {'✅ Passed' if summary.mean_context_precision >= 0.75 else '❌ Below Target'} |",
        f"| **Context Recall** | **{summary.mean_context_recall:.4f}** | \u2265 0.75 | {'✅ Passed' if summary.mean_context_recall >= 0.75 else '❌ Below Target'} |",
        f"| **Faithfulness** | **{summary.mean_faithfulness:.4f}** | \u2265 0.75 | {'✅ Passed' if summary.mean_faithfulness >= 0.75 else '❌ Below Target'} |",
        f"| **Answer Relevance** | **{summary.mean_answer_relevance:.4f}** | \u2265 0.75 | {'✅ Passed' if summary.mean_answer_relevance >= 0.75 else '❌ Below Target'} |",
        f"| **Overall Composite Score** | **{summary.mean_overall_score:.4f}** | \u2265 0.75 | {'✅ Passed' if summary.mean_overall_score >= 0.75 else '❌ Below Target'} |",
        "",
        "## 📋 Per-Question Detailed Breakdown",
        "",
        "| ID | Subject | Topic | Context Precision | Context Recall | Faithfulness | Answer Relevance | Overall |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for q in summary.per_question_scores:
        md_lines.append(
            f"| `{q.question_id}` | {q.subject} | {q.topic} | {q.context_precision:.2f} | {q.context_recall:.2f} | {q.faithfulness:.2f} | {q.answer_relevance:.2f} | **{q.overall_score:.2f}** |"
        )

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"📄 Markdown evaluation summary generated: {summary_md_path}")

    # Print summary to console
    print("\n" + "=" * 85)
    print("🏆 FINAL EVALUATION RESULTS")
    print("=" * 85)
    print(f"  * Context Precision:   {summary.mean_context_precision:.4f}  (Target >= 0.75) -> {'✅ PASS' if summary.mean_context_precision >= 0.75 else '❌ FAIL'}")
    print(f"  * Context Recall:      {summary.mean_context_recall:.4f}  (Target >= 0.75) -> {'✅ PASS' if summary.mean_context_recall >= 0.75 else '❌ FAIL'}")
    print(f"  * Faithfulness:        {summary.mean_faithfulness:.4f}  (Target >= 0.75) -> {'✅ PASS' if summary.mean_faithfulness >= 0.75 else '❌ FAIL'}")
    print(f"  * Answer Relevance:    {summary.mean_answer_relevance:.4f}  (Target >= 0.75) -> {'✅ PASS' if summary.mean_answer_relevance >= 0.75 else '❌ FAIL'}")
    print(f"  * Composite Overall:   {summary.mean_overall_score:.4f}  (Target >= 0.75) -> {'✅ PASS' if summary.mean_overall_score >= 0.75 else '❌ FAIL'}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full RAGAS Evaluation Harness")
    parser.add_argument("--dataset_path", type=str, default="./data/eval/eval_dataset.json")
    parser.add_argument("--results_path", type=str, default="./data/eval/results.json")
    parser.add_argument("--summary_md_path", type=str, default="./data/eval/eval_summary.md")
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    args = parser.parse_args()

    run_full_evaluation(
        dataset_path=args.dataset_path,
        results_path=args.results_path,
        summary_md_path=args.summary_md_path,
        processed_dir=args.processed_dir
    )
