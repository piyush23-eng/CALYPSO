import pytest
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.evaluation.evaluator import EvalItem, RAGEvaluator, QuestionEvalScore, EvaluationSummary


@pytest.fixture
def sample_eval_item():
    return EvalItem(
        question_id="TEST_01",
        subject="Operating Systems",
        topic="Virtual Memory",
        question="How is Effective Memory Access Time (EMAT) calculated in 2-level paging?",
        ground_truth_answer="EMAT = h * (t_tlb + t_m) + (1 - h) * (t_tlb + 3 * t_m)",
        ground_truth_context_keywords=["emat", "tlb", "paging", "memory"],
        expected_source_file="os_notes.md"
    )


@pytest.fixture
def evaluator():
    return RAGEvaluator()


def test_context_precision_computation(evaluator, sample_eval_item):
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            content="In 2-level paging, EMAT formula accounts for TLB hit ratio and memory access time.",
            topic="Operating Systems",
            subtopic="Paging",
            source_type="notes",
            source_file="os_notes.md",
            rrf_score=0.03,
            rerank_score=0.95
        ),
        RetrievedChunk(
            chunk_id="c2",
            content="CPU scheduling algorithms such as SRTF minimize average waiting time.",
            topic="Operating Systems",
            subtopic="Scheduling",
            source_type="notes",
            source_file="os_notes.md",
            rrf_score=0.02,
            rerank_score=0.80
        )
    ]
    precision = evaluator.compute_context_precision(chunks, sample_eval_item)
    assert precision >= 0.75


def test_context_recall_computation(evaluator, sample_eval_item):
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            content="In 2-level paging, EMAT formula is effective memory access time with TLB and memory access.",
            topic="Operating Systems",
            subtopic="Paging",
            source_type="notes",
            source_file="os_notes.md",
            rrf_score=0.03,
            rerank_score=0.95
        )
    ]
    recall = evaluator.compute_context_recall(chunks, sample_eval_item)
    assert recall >= 0.75


def test_faithfulness_computation(evaluator):
    citations = [
        {"sentence": "EMAT accounts for TLB latency.", "similarity_score": 0.85, "chunk_id": "c1", "source_file": "os_notes.md"}
    ]
    faithfulness = evaluator.compute_faithfulness(
        final_answer="EMAT accounts for TLB latency.",
        citations=citations,
        is_low_confidence=False
    )
    assert faithfulness >= 0.75


def test_answer_relevance_computation(evaluator, sample_eval_item):
    relevance = evaluator.compute_answer_relevance(
        query=sample_eval_item.question,
        final_answer="In 2-level paging, EMAT = h * (t_tlb + t_m) + (1 - h) * (t_tlb + 3 * t_m).",
        ground_truth_answer=sample_eval_item.ground_truth_answer
    )
    assert relevance >= 0.75
