import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.reranker import CrossEncoderReranker


class ReformulationLog(BaseModel):
    """
    Audit log entry recording query reformulation attempts for Corrective-RAG (CRAG).
    """
    timestamp: str = Field(description="ISO timestamp of attempt")
    original_query: str = Field(description="Initial user query")
    attempt_number: int = Field(description="Reformulation iteration (1 or 2)")
    rewritten_query: str = Field(description="Expanded/reformulated query")
    max_relevance_score_before: float = Field(description="Top rerank score prior to reformulation")
    max_relevance_score_after: float = Field(description="Top rerank score achieved after reformulation")
    passed_gate: bool = Field(description="Whether the rewritten query passed the threshold")
    reformulation_method: str = Field(description="Method used for reformulation (llm or domain_rule_expansion)")


class GatedRetrievalResult(BaseModel):
    """
    Final output of Corrective-RAG Relevance Gate.
    """
    original_query: str
    effective_query: str
    reformulation_count: int
    max_relevance_score: float
    passed_gate: bool
    is_low_confidence: bool
    chunks: List[RetrievedChunk]
    reformulation_history: List[ReformulationLog] = Field(default_factory=list)


class CorrectiveRelevanceGate:
    """
    Implements Corrective-RAG (CRAG) for GATE Computer Science problem solving.
    
    Workflow:
    1. Retrieve & Cross-Encoder rerank candidate chunks.
    2. Check max relevance score against threshold (default: 0.50).
    3. If score < threshold, reformulate the query using domain-specific GATE terminology expansion.
    4. Re-run hybrid retrieval + reranking with reformulated query.
    5. Cap at max_attempts (default: 2). If threshold still not met, return best-effort with is_low_confidence=True.
    6. Log every attempt to JSONL for auditability and transparency.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        relevance_threshold: float = 0.50,
        max_attempts: int = 2,
        log_file_path: str = "./data/eval/crag_reformulation_log.jsonl"
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.relevance_threshold = relevance_threshold
        self.max_attempts = max_attempts
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_attempt(self, entry: ReformulationLog):
        """Appends reformulation log to persistent JSONL file."""
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def reformulate_query(self, query: str, attempt: int) -> Tuple[str, str]:
        """
        Reformulates vague or colloquial user queries into formal GATE CS terminology.
        Uses structured domain ontology mappings with fallback acronym/concept expansions.
        """
        q_lower = query.lower().strip()
        
        # Domain expansion ontology for common GATE CS concepts
        expansions = [
            (["paging time", "page access", "paging memory", "tlb time", "memory time"],
             "Effective Memory Access Time EMAT 2-level paging TLB hit ratio main memory access latency"),
            
            (["process scheduling", "wait time", "process waiting", "scheduling time", "burst time"],
             "Shortest Remaining Time First SRTF CPU scheduling average waiting time burst time preemptive"),
            
            (["heap speed", "heap build", "building heap", "make heap", "binary heap time", "time speed heap"],
             "worst-case time complexity constructing binary max-heap unsorted array build-heap"),
            
            (["concurrency lock", "abort rollback", "dirty read lock", "serializable lock", "abort", "rollback"],
             "Strict 2-Phase Locking Strict 2PL conflict serializability eliminate cascading aborts"),
            
            (["normal form", "all prime", "candidate key normal", "highest normal"],
             "relation highest normal form every attribute prime 3NF BCNF functional dependencies"),
            
            (["page fault anomaly", "more frames more faults", "fifo fault anomaly"],
             "Belady's Anomaly FIFO page replacement stack algorithms LRU immunity"),
            
            (["deadlock safe", "need matrix", "avoid deadlock"],
             "Banker's Algorithm safe state resource allocation Need matrix Max Allocation"),
            
            (["tcp", "packet drop", "packet drops", "network packet", "congestion window", "slow start", "rate control", "network drop", "slow speed"],
             "TCP congestion control Slow Start Congestion Avoidance Fast Recovery cwnd ssthresh"),
            
            (["shift reduce", "parser conflict", "lr parser conflict"],
             "LR(1) LALR(1) parser shift-reduce conflict lookahead parsing table state"),
             
            (["context free", "pushdown", "grammar automaton", "chomsky"],
             "Chomsky Hierarchy Context-Free Grammar CFG Pushdown Automata PDA regular language")
        ]

        for triggers, formal_expansion in expansions:
            if any(t in q_lower for t in triggers):
                if attempt == 1:
                    return f"{query} {formal_expansion}", "domain_rule_expansion_hybrid"
                else:
                    return formal_expansion, "domain_rule_expansion_pure_formal"

        # General Fallback: Add general GATE CS problem keywords
        if attempt == 1:
            return f"{query} GATE Computer Science theory formula derivation", "general_domain_tagging"
        return f"{query} GATE CS Previous Year Question definition properties", "general_pyq_grounding"

    def retrieve_with_gate(
        self,
        query: str,
        top_candidates_per_source: int = 20,
        fused_top_k: int = 10,
        final_top_k: int = 3,
        topic_filter: Optional[str] = None
    ) -> GatedRetrievalResult:
        """
        Executes Corrective-RAG:
        Retrieves -> Reranks -> Checks relevance -> Reformulates if needed -> Logs attempts.
        """
        current_query = query
        reformulation_history: List[ReformulationLog] = []
        attempt = 0

        # Initial Retrieval and Rerank
        fused = self.retriever.retrieve(
            query=current_query,
            top_candidates_per_source=top_candidates_per_source,
            fused_top_k=fused_top_k,
            topic_filter=topic_filter
        )
        reranked = self.reranker.rerank(query=current_query, chunks=fused, top_k=final_top_k)

        max_score = reranked[0].rerank_score if (reranked and reranked[0].rerank_score is not None) else 0.0

        # If already passes threshold, return immediately
        if max_score >= self.relevance_threshold:
            return GatedRetrievalResult(
                original_query=query,
                effective_query=current_query,
                reformulation_count=0,
                max_relevance_score=max_score,
                passed_gate=True,
                is_low_confidence=False,
                chunks=reranked,
                reformulation_history=[]
            )

        # Corrective Loop
        while max_score < self.relevance_threshold and attempt < self.max_attempts:
            attempt += 1
            score_before = max_score
            
            # Reformulate query
            rewritten_query, method = self.reformulate_query(query=query, attempt=attempt)
            
            # Re-retrieve and re-rank with rewritten query
            new_fused = self.retriever.retrieve(
                query=rewritten_query,
                top_candidates_per_source=top_candidates_per_source,
                fused_top_k=fused_top_k,
                topic_filter=topic_filter
            )
            new_reranked = self.reranker.rerank(query=rewritten_query, chunks=new_fused, top_k=final_top_k)
            
            new_max_score = new_reranked[0].rerank_score if (new_reranked and new_reranked[0].rerank_score is not None) else 0.0
            passed = new_max_score >= self.relevance_threshold

            log_entry = ReformulationLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                original_query=query,
                attempt_number=attempt,
                rewritten_query=rewritten_query,
                max_relevance_score_before=round(score_before, 4),
                max_relevance_score_after=round(new_max_score, 4),
                passed_gate=passed,
                reformulation_method=method
            )
            self._log_attempt(log_entry)
            reformulation_history.append(log_entry)

            # Update working variables
            current_query = rewritten_query
            reranked = new_reranked
            max_score = new_max_score

            if passed:
                break

        passed_gate = max_score >= self.relevance_threshold
        is_low_confidence = not passed_gate

        return GatedRetrievalResult(
            original_query=query,
            effective_query=current_query,
            reformulation_count=attempt,
            max_relevance_score=round(max_score, 4),
            passed_gate=passed_gate,
            is_low_confidence=is_low_confidence,
            chunks=reranked,
            reformulation_history=reformulation_history
        )
