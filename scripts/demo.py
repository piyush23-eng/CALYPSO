import sys
import argparse
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.prompt import Prompt

from src.ingestion.indexer import DualIndexManager
from src.agent.orchestrator import CalypsoAgentOrchestrator


console = Console()

SHOWCASE_QUERIES = [
    {
        "title": "Clear Technical Query (OS Paging & TLB EMAT)",
        "query": "How is Effective Memory Access Time calculated in 2-level paging with TLB hit ratio?"
    },
    {
        "title": "Vague / Colloquial Query (CRAG Self-Correction Triggered)",
        "query": "slow speed when network packet drops"
    },
    {
        "title": "Ambiguous Theory Query (CRAG Self-Correction Triggered)",
        "query": "time speed heap"
    },
    {
        "title": "Out-of-Domain Negative Constraint Query",
        "query": "What is the capital city of France and who is the current president?"
    }
]


def run_single_query_demo(orchestrator: CalypsoAgentOrchestrator, query: str):
    console.print()
    console.print(Panel(f"[bold cyan]USER QUERY:[/bold cyan] [white]{query}[/white]", title="[bold yellow]Step 1: Query Input[/bold yellow]", border_style="yellow"))

    # Execute LangGraph Agent
    with console.status("[bold green]Agentic LangGraph state machine executing (Retrieve ──▶ Rerank ──▶ Gate ──▶ Reason ──▶ Cite)...[/bold green]"):
        state = orchestrator.run(query=query)

    # 1. Routing & Subject Classification
    subj = state.get("subject_hint", "General CS")
    relevance = state.get("relevance_score", 0.0)
    passed = state.get("passed_gate", False)
    reform_count = state.get("reformulation_count", 0)

    routing_text = Text()
    routing_text.append("• Classified Subject: ", style="bold")
    routing_text.append(f"{subj}\n", style="magenta")
    routing_text.append("• Initial Cross-Encoder Relevance: ", style="bold")
    routing_text.append(f"{relevance:.4f}\n", style="green" if relevance >= 0.50 else "red")
    routing_text.append("• Gate Status: ", style="bold")
    routing_text.append(f"{'PASSED (First Try)' if (passed and reform_count == 0) else 'REFORMULATED & PASSED' if passed else 'LOW CONFIDENCE / FALLBACK'}\n", style="bold green" if passed else "bold red")
    routing_text.append("• CRAG Reformulation Loops: ", style="bold")
    routing_text.append(f"{reform_count}", style="cyan")

    console.print(Panel(routing_text, title="[bold blue]Step 2: Routing & CRAG Relevance Decision[/bold blue]", border_style="blue"))

    # 2. Query Reformulation Diff (if triggered)
    if reform_count > 0:
        diff_table = Table(title="🔄 CRAG Query Reformulation Trace", border_style="cyan", show_header=True, header_style="bold magenta")
        diff_table.add_column("Iteration", style="dim", width=12)
        diff_table.add_column("Original Input Query", style="red")
        diff_table.add_column("Rewritten GATE Terminology Query", style="green")

        diff_table.add_row(
            f"Attempt {reform_count}",
            state["query"],
            state["reformulated_query"]
        )
        console.print(diff_table)

    # 3. Retrieved Context Evidence
    chunks = state.get("rerank_results", [])
    if chunks:
        context_table = Table(title="📚 Retrieved Context Evidence Chunks (Top Cross-Encoder Reranked)", border_style="green", show_header=True, header_style="bold green")
        context_table.add_column("Rank", justify="center", width=6)
        context_table.add_column("Chunk ID", style="cyan", width=18)
        context_table.add_column("Source Document", style="yellow", width=22)
        context_table.add_column("Topic / Subtopic", style="magenta", width=24)
        context_table.add_column("Rerank Score", justify="right", width=14)

        for idx, chunk in enumerate(chunks, 1):
            score_str = f"{chunk.rerank_score:.4f}" if chunk.rerank_score is not None else "N/A"
            context_table.add_row(
                str(idx),
                chunk.chunk_id,
                chunk.source_file,
                f"{chunk.topic} / {chunk.subtopic}",
                score_str
            )
        console.print(context_table)

    # 4. Verified Solution & Answer
    ans_markdown = Markdown(state.get("final_answer", ""))
    console.print(Panel(ans_markdown, title="[bold green]Step 3: Calypso Fine-Tuned Verified Solution[/bold green]", border_style="green"))

    # 5. Sentence-Level Citations Table
    citations = state.get("citations", [])
    if citations:
        cit_table = Table(title="🔗 Sentence-Level Attribution Citations (BAAI/bge-small-en-v1.5 Cosine Similarity)", border_style="magenta", show_header=True, header_style="bold cyan")
        cit_table.add_column("#", justify="center", width=4)
        cit_table.add_column("Answer Sentence Claim", style="white", width=42)
        cit_table.add_column("Cited Chunk ID", style="cyan", width=18)
        cit_table.add_column("Source File", style="yellow", width=20)
        cit_table.add_column("Cosine Sim", justify="right", width=12)

        for c_idx, cit in enumerate(citations, 1):
            s_preview = cit['sentence']
            if len(s_preview) > 55:
                s_preview = s_preview[:52] + "..."
            cit_table.add_row(
                str(c_idx),
                s_preview,
                cit['chunk_id'],
                cit['source_file'],
                f"{cit['similarity_score']:.4f}"
            )
        console.print(cit_table)
    else:
        console.print(Panel("[dim italic]No direct sentence claims matched the semantic attribution threshold (Cosine Sim >= 0.60).[/dim italic]", title="Citations", border_style="dim"))

    # 6. Reliability Badge
    conf = state.get("telemetry", {}).get("confidence", 0.0)
    cov = state.get("telemetry", {}).get("citation_coverage", 0.0)
    is_low = state.get("is_low_confidence", False)

    badge_style = "bold white on green" if not is_low and conf >= 0.60 else "bold white on red"
    console.print(Panel(
        f"Composite Confidence: [bold]{conf:.4f}[/bold] | Citation Coverage: [bold]{cov*100:.1f}%[/bold] | Reliability Status: [{'bold green' if not is_low else 'bold red'}]{'VERIFIED GROUNDED' if not is_low else 'LOW CONFIDENCE / UNVERIFIED'}[/{'bold green' if not is_low else 'bold red'}]",
        title="[bold]Reliability Telemetry[/bold]",
        border_style="cyan"
    ))
    console.print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="CALYPSO-RAG Interactive & Showcase Demo")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Path to index data")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]CALYPSO-RAG[/bold cyan]: [bold white]Agentic Retrieval-Augmented Generation for GATE Computer Science[/bold white]\n"
        "[dim]Hybrid RRF Retrieval (k=60) • Cross-Encoder Reranking • Corrective-RAG (CRAG) • LangGraph State Graph • Sentence Attribution[/dim]",
        border_style="cyan"
    ))

    with console.status("[bold green]Loading indices and initializing LangGraph agent...[/bold green]"):
        index_manager = DualIndexManager(
            persist_dir=f"{args.processed_dir}/chroma_db",
            bm25_persist_path=f"{args.processed_dir}/bm25_index.pkl"
        )
        index_manager.load_indices()
        orchestrator = CalypsoAgentOrchestrator(index_manager=index_manager)

    if args.interactive:
        console.print("[bold green]Type your GATE CS question below (or 'exit' to quit):[/bold green]")
        while True:
            try:
                user_q = Prompt.ask("\n[bold yellow]Ask CALYPSO-RAG[/bold yellow]")
                if user_q.strip().lower() in ["exit", "quit", "q"]:
                    console.print("[bold cyan]Exiting demo. Good luck with GATE prep![/bold cyan]")
                    break
                if not user_q.strip():
                    continue
                run_single_query_demo(orchestrator, user_q.strip())
            except (KeyboardInterrupt, EOFError):
                break
    else:
        console.print("[bold yellow]Running Automated Multi-Scenario Showcase Demo...[/bold yellow]")
        for item in SHOWCASE_QUERIES:
            console.print(f"\n[bold magenta]=== SCENARIO: {item['title']} ===[/bold magenta]")
            run_single_query_demo(orchestrator, item["query"])


if __name__ == "__main__":
    main()
