import pytest
import os
import shutil
from pathlib import Path
from src.ingestion.chunker import TopicAwareChunker, DocumentChunk
from src.ingestion.indexer import DualIndexManager


SAMPLE_GATE_DOC = """# Operating Systems Reference Notes

## Virtual Memory and Paging

### 2-Level Paging Architecture
In a 2-level paging system, the virtual address is divided into outer page table index, inner page table index, and page offset.
Translation Lookaside Buffer (TLB) stores recent virtual-to-physical address translations.

Effective Memory Access Time (EMAT) formula:
EMAT = h * (t_tlb + t_m) + (1 - h) * (t_tlb + (k + 1) * t_m)
where h is hit ratio, k is number of paging levels, t_tlb is TLB access latency, and t_m is main memory access latency.

### Page Replacement Policies
LRU (Least Recently Used) is an optimal practical stack algorithm that does not suffer from Belady's Anomaly.
FIFO (First-In, First-Out) can exhibit Belady's Anomaly where increasing allocated page frames increases page fault frequency.
"""

SAMPLE_PYQ_DOC = """# GATE CS PYQ Archive

## Question 1: OS CPU Scheduling [GATE 2023]
**Topic**: Operating Systems
**Subtopic**: CPU Scheduling
**Question**:
Consider three processes arriving at time 0 with burst times 10 ms, 5 ms, 2 ms. Under SRTF scheduling, what is the average waiting time?
Options: (A) 2.33 ms (B) 4.0 ms (C) 5.66 ms (D) 3.0 ms
**Answer and Reasoning**:
SRTF executes shortest remaining burst first. Average waiting time is 3.0 ms. Correct Answer: (D)

## Question 2: Database Normalization [GATE 2022]
**Topic**: Database Management Systems
**Subtopic**: Normalization
**Question**:
Given relation R(A, B, C, D, E) with F = {A -> BC, CD -> E, B -> D, E -> A}. What is the highest normal form?
Options: (A) 1NF (B) 2NF (C) 3NF (D) BCNF
**Answer and Reasoning**:
All attributes are prime. Every FD satisfies 3NF conditions. Not in BCNF because B is not a superkey. Correct Answer: (C)
"""


@pytest.fixture(scope="module")
def temp_index_dir():
    test_dir = Path("./data/processed/test_chroma_db")
    test_bm25 = Path("./data/processed/test_bm25_index.pkl")
    yield str(test_dir), str(test_bm25)
    # Cleanup after test
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    if test_bm25.exists():
        test_bm25.unlink(missing_ok=True)


def test_topic_aware_chunker_notes():
    chunker = TopicAwareChunker(max_chars=800, min_chars=50)
    chunks = chunker.chunk_document(content=SAMPLE_GATE_DOC, source_file="os_notes.md")

    assert len(chunks) >= 2
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(c.source_type == "notes" for c in chunks)
    assert all(c.topic == "Operating Systems" for c in chunks)
    assert any("2-Level Paging" in c.subtopic for c in chunks)
    assert any("Page Replacement" in c.subtopic for c in chunks)
    assert all(c.chunk_id.startswith("doc_") for c in chunks)


def test_topic_aware_chunker_pyqs():
    chunker = TopicAwareChunker(max_chars=1000, min_chars=50)
    chunks = chunker.chunk_document(content=SAMPLE_PYQ_DOC, source_file="gate_pyq_archive.md")

    assert len(chunks) == 2
    assert all(c.source_type == "pyq" for c in chunks)
    assert chunks[0].topic == "Operating Systems"
    assert chunks[0].subtopic == "CPU Scheduling"
    assert chunks[1].topic == "Database Management Systems"
    assert chunks[1].subtopic == "Normalization"
    assert all(c.chunk_id.startswith("pyq_") for c in chunks)


def test_dual_index_manager_smoke_test(temp_index_dir):
    chroma_dir, bm25_path = temp_index_dir
    chunker = TopicAwareChunker(max_chars=800, min_chars=50)
    doc_chunks = chunker.chunk_document(content=SAMPLE_GATE_DOC, source_file="os_notes.md")
    pyq_chunks = chunker.chunk_document(content=SAMPLE_PYQ_DOC, source_file="gate_pyq_archive.md")
    all_chunks = doc_chunks + pyq_chunks

    manager = DualIndexManager(
        persist_dir=chroma_dir,
        bm25_persist_path=bm25_path,
        collection_name="test_calypso_kb"
    )

    # 1. Build and persist indices
    manager.build_and_save(all_chunks)

    assert os.path.exists(bm25_path)
    assert os.path.exists(chroma_dir)

    # 2. Test BM25 exact keyword retrieval
    bm25_results = manager.search_bm25(query="EMAT formula 2-level paging hit ratio", top_k=2)
    assert len(bm25_results) > 0
    assert "EMAT" in bm25_results[0]["content"]
    assert bm25_results[0]["topic"] == "Operating Systems"

    # 3. Test Dense semantic retrieval (BAAI/bge-small-en-v1.5)
    dense_results = manager.search_dense(query="Which normal form is satisfied when all attributes are prime?", top_k=2)
    assert len(dense_results) > 0
    assert "Normalization" in dense_results[0]["subtopic"] or "Database" in dense_results[0]["topic"]

    # 4. Test Topic-Filtered retrieval
    filtered_dense = manager.search_dense(
        query="scheduling algorithm waiting time",
        top_k=2,
        topic_filter="Operating Systems"
    )
    assert len(filtered_dense) > 0
    assert all(r["topic"] == "Operating Systems" for r in filtered_dense)
