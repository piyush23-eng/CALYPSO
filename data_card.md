# Dataset Card — GATE CS Doubt Solver (v1-pilot)

## Dataset Summary
The **GATE-CS-Doubt-Solver** dataset is a curated, high-integrity instruction-tuning corpus specifically designed for fine-tuning compact Large Language Models (e.g., Qwen2.5-1.5B-Instruct) on Graduate Aptitude Test in Engineering (GATE) Computer Science and Information Technology examination problems.

Unlike generic coding or conversational QA datasets, this corpus focuses strictly on **rigorous mathematical and conceptual reasoning**, multi-step algorithmic derivations, formal proofs, and examiner-style option elimination.

---

## Lineage & Sources
1. **GATE Overflow Community Archive**: Curated past-year GATE CSE questions with verified answer keys.
2. **Official IIT GATE Question Papers (1990–2024)**: Official Master Question Papers and final Answer Keys.
3. **GeeksforGeeks GATE CS Archives**: Topic-wise solutions and explanations.

---

## Subject & Topic Coverage
The dataset covers the 7 core pillars of the GATE CS syllabus:

| Subject | Example Count | Percentage |
|---|---|---|
| Algorithms | 10 | 16.9% |
| Operating Systems | 9 | 15.3% |
| DBMS | 8 | 13.6% |
| Computer Networks | 8 | 13.6% |
| Theory of Computation | 8 | 13.6% |
| Compiler Design | 8 | 13.6% |
| Digital Logic | 8 | 13.6% |


---

## Question Types & Representation

| Question Type | Count | Percentage |
|---|---|---|
| MCQ | 36 | 61.0% |
| NAT | 23 | 39.0% |


---

## Dataset Splits & Leakage Prevention
To evaluate true model generalization rather than memorized question variations:
- **Temporal & Subject-Isolated Split**:
  - **Train Set**: 9 examples (Historical foundation years)
  - **Validation Set**: 33 examples (GATE 2021–2022 anchor years)
  - **Held-out Test Set**: 17 examples (GATE 2023–2024 evaluation years)
- **Zero-Leakage Guarantee**:
  - Exact ID Overlap: `0`
  - Exact SHA-256 Content Hash Overlap: `0`
  - MinHash Fuzzy Jaccard Similarity: `< 0.85`

---

## Formatting & Reasoning Schema
Every training pair is formatted in **ChatML / Qwen2.5 SFT format** with a standard 4-stage chain-of-thought structure:
1. **Conceptual Framework & Core Principles**: Identifies foundational theorems (e.g., Master Theorem, Paging EMAT, Chomsky Hierarchy, K-Map rules).
2. **Step-by-Step Derivation & Analysis**: Line-by-line algorithmic and mathematical progression with LaTeX math (`$...$`).
3. **Option Evaluation & Verification**: Concrete justification for why correct options hold and why distractors fail.
4. **Final Answer Box**: Unambiguous, machine-evaluable final conclusion.

---

## Dataset Health Metrics
- **Total Processed Samples**: 59
- **Average Prompt Length**: 82.2 words
- **Average Response Length**: 196.1 words

---

## Known Limitations & Tradeoffs (Interview Documentation)
1. **Diagram-Heavy Problems**: Pure geometry/circuit diagram questions without text/ASCII/equation equivalents are omitted or transcribed to text-circuit descriptions in this text-only model pipeline.
2. **Multi-Select Questions (MSQ)**: MSQs were introduced in GATE 2021; thus historical pre-2021 questions are predominantly MCQ and NAT.
3. **Pilot Scale**: Currently tested on 59 pilot examples. Full production scaling targets 2,000–4,000 deduplicated records.
