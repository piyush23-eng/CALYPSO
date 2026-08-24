from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
import hashlib
import re
from pathlib import Path


class DocumentChunk(BaseModel):
    """
    Schema for a topic-aware chunk generated from GATE CS documents.
    Preserves exact structural metadata for granular retrieval and citation mapping.
    """
    chunk_id: str = Field(description="Unique deterministic hash of chunk content and metadata")
    content: str = Field(description="Normalized text content of the chunk")
    topic: str = Field(description="Primary GATE CS subject (e.g. Operating Systems, DBMS)")
    subtopic: str = Field(description="Granular section or question title")
    source_type: Literal["syllabus", "pyq", "notes"] = Field(description="Origin category")
    source_file: str = Field(description="Relative filepath of the source document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or question attributes")


class TopicAwareChunker:
    """
    Topic-Aware Chunker designed specifically for GATE CS technical documents.
    
    Instead of blind fixed-character sliding windows (which split formulas and tables across boundaries),
    this chunker:
    1. Detects document source type based on filename conventions.
    2. Identifies macro structure using Markdown headers (#, ##, ###) or Question delimiters.
    3. Maintains contextual hierarchy (Topic -> Subtopic).
    4. Employs recursive semantic splitting (by paragraph, sentence) ONLY if a section exceeds max_chars.
    """

    def __init__(self, max_chars: int = 1200, min_chars: int = 100, overlap_chars: int = 150):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def infer_source_type(file_path: str) -> Literal["syllabus", "pyq", "notes"]:
        name = Path(file_path).name.lower()
        if "syllabus" in name:
            return "syllabus"
        elif "pyq" in name or "question" in name:
            return "pyq"
        return "notes"

    @staticmethod
    def infer_topic_from_header(header_text: str, fallback_topic: str = "General CS") -> str:
        text = header_text.lower()
        topic_map = {
            "operating system": "Operating Systems",
            "process": "Operating Systems",
            "memory management": "Operating Systems",
            "deadlock": "Operating Systems",
            "database": "Database Management Systems",
            "dbms": "Database Management Systems",
            "relational": "Database Management Systems",
            "transaction": "Database Management Systems",
            "algorithm": "Algorithms",
            "data structure": "Algorithms",
            "graph": "Algorithms",
            "heap": "Algorithms",
            "tree": "Algorithms",
            "theory of computation": "Theory of Computation",
            "automata": "Theory of Computation",
            "regular language": "Theory of Computation",
            "compiler": "Compiler Design",
            "parsing": "Compiler Design",
            "discrete math": "Engineering Mathematics",
            "linear algebra": "Engineering Mathematics",
            "calculus": "Engineering Mathematics",
            "digital logic": "Digital Logic",
            "computer organization": "Computer Organization and Architecture",
            "computer networks": "Computer Networks",
            "network": "Computer Networks"
        }
        for key, mapped_topic in topic_map.items():
            if key in text:
                return mapped_topic
        return fallback_topic

    def _recursive_character_fallback(self, text: str) -> List[str]:
        """
        Deterministic recursive character splitting as fallback for oversized sections.
        Splits by double newline -> single newline -> periods/semicolons.
        """
        if len(text) <= self.max_chars:
            return [text]

        splits = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.max_chars:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk and len(current_chunk) >= self.min_chars:
                    splits.append(current_chunk)
                
                # If paragraph itself is too large, split by sentences
                if len(para) > self.max_chars:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) + 1 <= self.max_chars:
                            sub_chunk = f"{sub_chunk} {sent}".strip()
                        else:
                            if sub_chunk:
                                splits.append(sub_chunk)
                            sub_chunk = sent
                    if sub_chunk:
                        current_chunk = sub_chunk
                else:
                    current_chunk = para

        if current_chunk and len(current_chunk) >= self.min_chars:
            splits.append(current_chunk)

        return splits if splits else [text[:self.max_chars]]

    def chunk_document(self, content: str, source_file: str) -> List[DocumentChunk]:
        """
        Parses Markdown/Text content into structured DocumentChunk objects.
        """
        source_type = self.infer_source_type(source_file)
        chunks: List[DocumentChunk] = []

        # Strategy 1: PYQ Question Boundary Splitting
        if source_type == "pyq":
            # Match headers like "## Question 1: ..." or "## Q1: ..."
            raw_sections = re.split(r'\n(?=##\s+Question|\n##\s+Q\d+)', content)
            current_topic = "GATE PYQ"

            q_idx = 0
            for section in raw_sections:
                section_text = section.strip()
                if not section_text:
                    continue

                # Skip document-level preface headers that don't contain a question
                if not (section_text.startswith("## Question") or section_text.startswith("## Q") or "**Question**:" in section_text):
                    continue

                lines = section_text.split('\n')
                subtopic = lines[0].replace('#', '').strip()
                
                # Extract explicit topic if provided in metadata lines
                topic_match = re.search(r'\*\*Topic\*\*:\s*(.*)', section_text)
                subtopic_match = re.search(r'\*\*Subtopic\*\*:\s*(.*)', section_text)
                
                resolved_topic = topic_match.group(1).strip() if topic_match else self.infer_topic_from_header(subtopic, current_topic)
                resolved_subtopic = subtopic_match.group(1).strip() if subtopic_match else subtopic

                chunk_id = hashlib.sha256(f"{source_file}_{resolved_topic}_{q_idx}_{section_text[:50]}".encode('utf-8')).hexdigest()[:12]
                
                chunks.append(DocumentChunk(
                    chunk_id=f"pyq_{chunk_id}",
                    content=section_text,
                    topic=resolved_topic,
                    subtopic=resolved_subtopic,
                    source_type=source_type,
                    source_file=source_file,
                    metadata={"question_number": q_idx + 1}
                ))
                q_idx += 1
            return chunks

        # Strategy 2: Topic & Subtopic Header Parsing for Notes & Syllabus
        # Split across ## headers (Macro sections)
        h1_match = re.search(r'^#\s+(.*)', content, re.MULTILINE)
        doc_title = h1_match.group(1).strip() if h1_match else Path(source_file).stem
        default_topic = self.infer_topic_from_header(doc_title, "Computer Science")

        sections = re.split(r'\n(?=##\s+)', content)
        
        for s_idx, sec in enumerate(sections):
            sec = sec.strip()
            if not sec:
                continue

            sec_lines = sec.split('\n')
            sec_header = sec_lines[0].replace('#', '').strip()
            section_topic = self.infer_topic_from_header(sec_header, default_topic)

            # Sub-split on ### headers if present
            subsections = re.split(r'\n(?=###\s+)', sec)
            for sub_idx, sub in enumerate(subsections):
                sub = sub.strip()
                if not sub:
                    continue

                sub_lines = sub.split('\n')
                sub_header = sub_lines[0].replace('#', '').strip() if sub.startswith('#') else sec_header

                # If the subsection is within token limit, keep whole to preserve mathematical reasoning
                if len(sub) <= self.max_chars:
                    if len(sub) >= self.min_chars:
                        chunk_id = hashlib.sha256(f"{source_file}_{section_topic}_{s_idx}_{sub_idx}_{sub[:40]}".encode('utf-8')).hexdigest()[:12]
                        chunks.append(DocumentChunk(
                            chunk_id=f"doc_{chunk_id}",
                            content=sub,
                            topic=section_topic,
                            subtopic=sub_header,
                            source_type=source_type,
                            source_file=source_file,
                            metadata={"section_index": s_idx, "subsection_index": sub_idx}
                        ))
                else:
                    # Recursive fallback
                    sub_pieces = self._recursive_character_fallback(sub)
                    for p_idx, piece in enumerate(sub_pieces):
                        chunk_id = hashlib.sha256(f"{source_file}_{section_topic}_{s_idx}_{sub_idx}_{p_idx}_{piece[:40]}".encode('utf-8')).hexdigest()[:12]
                        chunks.append(DocumentChunk(
                            chunk_id=f"doc_{chunk_id}",
                            content=piece,
                            topic=section_topic,
                            subtopic=f"{sub_header} (Part {p_idx+1})",
                            source_type=source_type,
                            source_file=source_file,
                            metadata={"section_index": s_idx, "subsection_index": sub_idx, "part": p_idx + 1}
                        ))

        return chunks
