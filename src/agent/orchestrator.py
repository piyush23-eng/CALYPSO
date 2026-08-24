from typing import TypedDict, List, Dict, Any, Optional, Literal
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END

from src.ingestion.indexer import DualIndexManager
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.relevance_gate import CorrectiveRelevanceGate, GatedRetrievalResult
from src.generation.calypso_client import CalypsoClient
from src.generation.citation_mapper import CitationMapper, GenerationOutput, SentenceCitation


class AgentState(TypedDict):
    """
    Complete state schema for the CALYPSO-RAG LangGraph state machine.
    """
    query: str
    reformulated_query: str
    subject_hint: Optional[str]
    retrieval_results: List[RetrievedChunk]
    rerank_results: List[RetrievedChunk]
    relevance_score: float
    reformulation_count: int
    passed_gate: bool
    is_low_confidence: bool
    generation: str
    citations: List[Dict[str, Any]]
    final_answer: str
    telemetry: Dict[str, Any]


class CalypsoAgentOrchestrator:
    """
    Agentic Orchestrator built using LangGraph state graph.
    
    Coordinates the 5-stage reasoning lifecycle:
    1. classify_query -> identifies GATE CS domain & subject bounds.
    2. retrieve -> executes hybrid lexical (BM25) & dense (BGE-Small) retrieval with RRF.
    3. rerank -> applies cross-encoder full attention scoring (ms-marco-MiniLM).
    4. check_relevance -> evaluates top candidate relevance against gate threshold (0.50).
       └── [Conditional Edge]:
           ├── If Score < 0.50 and count < 2 -> reformulate_and_retry (Loops back to retrieve)
           └── If Score >= 0.50 or count >= 2 -> generate (Proceeds to generation)
    5. generate -> invokes fine-tuned Calypso client with strict negative grounding constraints.
    6. attach_citations -> maps sentence-level semantic attribution with cosine similarity.
    """

    def __init__(
        self,
        index_manager: Optional[DualIndexManager] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        relevance_gate: Optional[CorrectiveRelevanceGate] = None,
        calypso_client: Optional[CalypsoClient] = None,
        citation_mapper: Optional[CitationMapper] = None,
        relevance_threshold: float = 0.50,
        max_reformulations: int = 2
    ):
        self.index_manager = index_manager or DualIndexManager()
        self.retriever = retriever or HybridRetriever(index_manager=self.index_manager, rrf_k=60)
        self.reranker = reranker or CrossEncoderReranker()
        self.relevance_gate = relevance_gate or CorrectiveRelevanceGate(
            retriever=self.retriever,
            reranker=self.reranker,
            relevance_threshold=relevance_threshold,
            max_attempts=max_reformulations
        )
        self.calypso_client = calypso_client or CalypsoClient()
        self.citation_mapper = citation_mapper or CitationMapper(embedder=self.index_manager.embedder)
        self.relevance_threshold = relevance_threshold
        self.max_reformulations = max_reformulations

        # Compile the LangGraph agent state machine
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    # ── Node 1: Classify Query ───────────────────────────────────────────────
    def _classify_query_node(self, state: AgentState) -> Dict[str, Any]:
        q = state["query"].lower()
        subject = "General CS"
        
        if any(w in q for w in ["page", "paging", "tlb", "process", "scheduling", "deadlock", "banker", "os", "semaphore"]):
            subject = "Operating Systems"
        elif any(w in q for w in ["normal", "3nf", "bcnf", "acid", "serializ", "2pl", "sql", "dbms", "database", "transaction"]):
            subject = "Database Management Systems"
        elif any(w in q for w in ["heap", "sort", "graph", "tree", "recurrence", "complexity", "asymptotic", "algorithm"]):
            subject = "Algorithms"
        elif any(w in q for w in ["tcp", "ip", "packet", "congestion", "sliding window", "gbn", "sr", "subnet", "network"]):
            subject = "Computer Networks"
        elif any(w in q for w in ["grammar", "automata", "dfa", "nfa", "pda", "turing", "chomsky", "regular"]):
            subject = "Theory of Computation"
        elif any(w in q for w in ["parse", "parser", "lr", "lalr", "syntax", "lexical", "compiler"]):
            subject = "Compiler Design"

        return {
            "subject_hint": subject,
            "reformulated_query": state.get("reformulated_query") or state["query"],
            "reformulation_count": state.get("reformulation_count", 0),
            "telemetry": {"classified_subject": subject}
        }

    # ── Node 2: Retrieve ─────────────────────────────────────────────────────
    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        active_query = state["reformulated_query"]
        fused = self.retriever.retrieve(
            query=active_query,
            top_candidates_per_source=20,
            fused_top_k=10
        )
        return {"retrieval_results": fused}

    # ── Node 3: Rerank ───────────────────────────────────────────────────────
    def _rerank_node(self, state: AgentState) -> Dict[str, Any]:
        active_query = state["reformulated_query"]
        candidates = state["retrieval_results"]
        reranked = self.reranker.rerank(query=active_query, chunks=candidates, top_k=3)
        return {"rerank_results": reranked}

    # ── Node 4: Check Relevance ──────────────────────────────────────────────
    def _check_relevance_node(self, state: AgentState) -> Dict[str, Any]:
        reranked = state["rerank_results"]
        max_score = reranked[0].rerank_score if (reranked and reranked[0].rerank_score is not None) else 0.0
        passed = max_score >= self.relevance_threshold
        
        return {
            "relevance_score": max_score,
            "passed_gate": passed,
            "is_low_confidence": not passed and state["reformulation_count"] >= self.max_reformulations
        }

    # ── Node 5: Reformulate & Retry ──────────────────────────────────────────
    def _reformulate_node(self, state: AgentState) -> Dict[str, Any]:
        count = state["reformulation_count"] + 1
        rewritten, method = self.relevance_gate.reformulate_query(query=state["query"], attempt=count)
        
        return {
            "reformulated_query": rewritten,
            "reformulation_count": count,
            "telemetry": {
                **state.get("telemetry", {}),
                f"reformulation_attempt_{count}": {
                    "rewritten_query": rewritten,
                    "method": method
                }
            }
        }

    # ── Node 6: Generate ─────────────────────────────────────────────────────
    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        reranked = state["rerank_results"]
        subject = state["subject_hint"]
        
        answer_text = self.calypso_client.generate(
            query=state["query"],
            chunks=reranked,
            subject=subject
        )
        return {"generation": answer_text}

    # ── Node 7: Attach Citations ─────────────────────────────────────────────
    def _attach_citations_node(self, state: AgentState) -> Dict[str, Any]:
        reranked = state["rerank_results"]
        ans = state["generation"]
        
        mock_gated = GatedRetrievalResult(
            original_query=state["query"],
            effective_query=state["reformulated_query"],
            reformulation_count=state["reformulation_count"],
            max_relevance_score=state["relevance_score"],
            passed_gate=state["passed_gate"],
            is_low_confidence=state["is_low_confidence"],
            chunks=reranked
        )
        
        output = self.citation_mapper.map_citations(
            query=state["query"],
            answer_text=ans,
            gated_result=mock_gated
        )
        
        return {
            "citations": [c.model_dump() for c in output.citations],
            "final_answer": output.answer_text,
            "is_low_confidence": output.is_low_confidence,
            "telemetry": {
                **state.get("telemetry", {}),
                "confidence": output.confidence,
                "citation_coverage": output.retrieval_metadata.get("citation_coverage", 0.0)
            }
        }

    # ── Conditional Router ───────────────────────────────────────────────────
    def _route_after_relevance_check(self, state: AgentState) -> Literal["generate", "reformulate_and_retry"]:
        """
        Agentic Conditional Edge:
        If relevance score >= threshold OR max reformulation attempts reached -> Generate.
        Otherwise -> Reformulate and loop back to Retrieve.
        """
        if state["passed_gate"] or state["reformulation_count"] >= self.max_reformulations:
            return "generate"
        return "reformulate_and_retry"

    def _build_graph(self) -> StateGraph:
        """
        Constructs the cyclic LangGraph state machine.
        """
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("check_relevance", self._check_relevance_node)
        workflow.add_node("reformulate_and_retry", self._reformulate_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("attach_citations", self._attach_citations_node)

        # Add Linear Edges
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "check_relevance")

        # Add Conditional Edge for Corrective RAG
        workflow.add_conditional_edges(
            "check_relevance",
            self._route_after_relevance_check,
            {
                "generate": "generate",
                "reformulate_and_retry": "reformulate_and_retry"
            }
        )

        # Feedback loop edge
        workflow.add_edge("reformulate_and_retry", "retrieve")

        # Final generation & citation edges
        workflow.add_edge("generate", "attach_citations")
        workflow.add_edge("attach_citations", END)

        return workflow

    def run(self, query: str) -> Dict[str, Any]:
        """
        Runs the LangGraph agent state graph to completion for a user query.
        """
        initial_state: AgentState = {
            "query": query,
            "reformulated_query": query,
            "subject_hint": None,
            "retrieval_results": [],
            "rerank_results": [],
            "relevance_score": 0.0,
            "reformulation_count": 0,
            "passed_gate": False,
            "is_low_confidence": False,
            "generation": "",
            "citations": [],
            "final_answer": "",
            "telemetry": {}
        }
        return self.app.invoke(initial_state)

    def export_mermaid(self) -> str:
        """
        Exports the LangGraph state machine as a Mermaid flowchart for architecture documentation.
        """
        mermaid_code = """```mermaid
flowchart TD
    START([User Query Input]) --> Classify[Classify Query & Subject]
    Classify --> Retrieve[Parallel Hybrid Retrieval: BM25 + Dense BGE-Small]
    Retrieve --> Rerank[Cross-Encoder Reranker: ms-marco-MiniLM]
    Rerank --> RelevanceCheck{Check Relevance Score >= 0.50?}
    
    RelevanceCheck -- "No (Score < 0.50 & Attempt < 2)" --> Reformulate[CRAG Query Reformulation & Expansion]
    Reformulate -.-> Retrieve
    
    RelevanceCheck -- "Yes (Score >= 0.50 OR Attempt >= 2)" --> Generate[Calypso LLM Generation with Negative Grounding]
    Generate --> Citations[Sentence-Level Cosine Citation Mapper]
    Citations --> END([Verified Answer with Citations & Confidence])
```"""
        return mermaid_code
