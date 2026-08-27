"""Cleaning, LaTeX normalization, and deduplication engine for GATE CS data."""

import hashlib
import html
import re
from typing import Dict, List, Optional, Set, Tuple
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH

from src.data.models import CleanedQuestion, QuestionType, RawQuestion, Subject
from src.utils.logger import setup_logger

logger = setup_logger("cleaner")


class DataCleaner:
    """Handles HTML parsing, LaTeX formula normalization, option parsing,

    and multi-stage deduplication.
    """

    # Common HTML entity to LaTeX replacements
    ENTITY_TO_LATEX = {
        "&alpha;": r"\alpha",
        "&beta;": r"\beta",
        "&gamma;": r"\gamma",
        "&delta;": r"\delta",
        "&theta;": r"\theta",
        "&Theta;": r"\Theta",
        "&lambda;": r"\lambda",
        "&mu;": r"\mu",
        "&pi;": r"\pi",
        "&sigma;": r"\sigma",
        "&Sigma;": r"\Sigma",
        "&omega;": r"\omega",
        "&Omega;": r"\Omega",
        "&le;": r"\le",
        "&ge;": r"\ge",
        "&ne;": r"\neq",
        "&le;": r"\le",
        "&ge;": r"\ge",
        "&times;": r"\times",
        "&div;": r"\div",
        "&plusmn;": r"\pm",
        "&isin;": r"\in",
        "&notin;": r"\notin",
        "&sub;": r"\subset",
        "&sube;": r"\subseteq",
        "&cap;": r"\cap",
        "&cup;": r"\cup",
        "&empty;": r"\emptyset",
        "&rarr;": r"\rightarrow",
        "&larr;": r"\leftarrow",
        "&harr;": r"\leftrightarrow",
        "&rArr;": r"\Rightarrow",
        "&infin;": r"\infty",
        "&lang;": r"\langle",
        "&rang;": r"\rangle",
        "&lfloor;": r"\lfloor",
        "&rfloor;": r"\rfloor",
        "&lceil;": r"\lceil",
        "&rceil;": r"\rceil",
        "&exist;": r"\exists",
        "&forall;": r"\forall",
    }

    def __init__(self, lsh_threshold: float = 0.85, num_perm: int = 128):
        self.lsh_threshold = lsh_threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.seen_hashes: Set[str] = set()

    def clean_text(self, text: str) -> str:
        """Alias for clean_html_to_markdown to clean plain or HTML text."""
        return self.clean_html_to_markdown(text)

    def clean_html_to_markdown(self, raw_html: str) -> str:
        """Converts raw HTML snippet into clean markdown with standardized LaTeX math."""
        if not raw_html or not raw_html.strip():
            return ""

        # Pre-replace known HTML entities before BS4 parses them out
        text = raw_html
        for ent, ltx in self.ENTITY_TO_LATEX.items():
            text = text.replace(ent, f" ${ltx}$ ")

        soup = BeautifulSoup(text, "html.parser")

        # Convert line breaks and list items
        for br in soup.find_all("br"):
            br.replace_with("\n")

        for p in soup.find_all("p"):
            p.insert_before("\n")
            p.insert_after("\n")

        for li in soup.find_all("li"):
            li.insert_before("\n- ")

        # Convert <sup> and <sub> tags to LaTeX equivalents
        for sup in soup.find_all("sup"):
            sup_text = sup.get_text().strip()
            sup.replace_with(f"^{{{sup_text}}}")

        for sub in soup.find_all("sub"):
            sub_text = sub.get_text().strip()
            sub.replace_with(f"_{{{sub_text}}}")

        # Convert code blocks
        for pre in soup.find_all("pre"):
            code_text = pre.get_text().strip()
            pre.replace_with(f"\n```c\n{code_text}\n```\n")

        for code in soup.find_all("code"):
            # If not inside pre
            if code.parent and code.parent.name != "pre":
                code_text = code.get_text().strip()
                code.replace_with(f"`{code_text}`")

        # Process mathjax / katex elements
        for math_elem in soup.find_all(["span", "div"], class_=re.compile(r"mathjax|katex|math", re.I)):
            tex = math_elem.get_text().strip()
            if not (tex.startswith("$") and tex.endswith("$")):
                tex = f"${tex}$"
            math_elem.replace_with(f" {tex} ")

        # Handle tables
        for table in soup.find_all("table"):
            md_table = self._convert_html_table_to_markdown(table)
            table.replace_with(md_table)

        # Extract text
        cleaned_text = soup.get_text(separator=" ")

        # Unescape any residual entities
        cleaned_text = html.unescape(cleaned_text)

        # Post-clean LaTeX equations & spacing
        cleaned_text = self._standardize_math(cleaned_text)
        cleaned_text = self._strip_noise(cleaned_text)

        return cleaned_text.strip()

    def _convert_html_table_to_markdown(self, table_soup) -> str:
        """Converts an HTML <table> element into GitHub Flavored Markdown table."""
        rows = table_soup.find_all("tr")
        if not rows:
            return ""

        table_data = []
        for row in rows:
            cols = [col.get_text().strip().replace("\n", " ") for col in row.find_all(["th", "td"])]
            if cols:
                table_data.append(cols)

        if not table_data:
            return ""

        max_cols = max(len(r) for r in table_data)
        # Pad shorter rows
        for r in table_data:
            while len(r) < max_cols:
                r.append("")

        header = table_data[0]
        separator = ["---"] * max_cols
        md_lines = ["\n| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]

        for row in table_data[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n" + "\n".join(md_lines) + "\n"

    def _standardize_math(self, text: str) -> str:
        """Fixes LaTeX formatting, cleans math boundaries, and avoids double-dollar nesting."""
        # Convert \(...\) to $...$
        text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)
        # Convert \[...\] to $$...$$
        text = re.sub(r"\\\[(.*?)\\\]", r"\n$$\1$$\n", text, flags=re.DOTALL)

        # Standardize asymptotic notation for plain text (e.g. Theta(n) -> $\Theta(n)$)
        def fix_plain_asymptotics(match):
            sym = match.group(1)
            inner = re.sub(r"\s+", " ", match.group(2).strip())
            tex_sym = r"\Theta" if sym.lower() == "theta" else (r"\Omega" if sym.lower() == "omega" else "O")
            return f"${tex_sym}({inner})$"

        text = re.sub(
            r"(?<![\$\\\w])(O|Theta|Omega)\s*\(([a-zA-Z0-9\s\+\-\*\/\^\_]+)\)(?![\$\w])",
            fix_plain_asymptotics,
            text,
            flags=re.IGNORECASE,
        )

        # Remove duplicate/nested math formatting artifacts
        text = re.sub(r"\$\s*\$", "", text)
        text = re.sub(r"\${3,}", "$$", text)

        # Ensure space before $ if preceded by alphanumeric/punctuation
        text = re.sub(r"([a-zA-Z0-9\.\,\:\)])\$", r"\1 $", text)
        # Ensure space after $ if followed by alphanumeric/punctuation
        text = re.sub(r"(\$[^$\n]+\$)([a-zA-Z0-9])", r"\1 \2", text)

        # Normalize consecutive spaces
        text = re.sub(r"[ \t]+", " ", text)

        return text

    def _strip_noise(self, text: str) -> str:
        """Removes web platform artifacts, voting counters, and banner text."""
        # Common noise patterns from forum exports
        noise_patterns = [
            r"(?i)vote\s+up\s+\d+",
            r"(?i)vote\s+down\s+\d+",
            r"(?i)selected\s+as\s+best\s+answer",
            r"(?i)asked\s+by\s+[\w\s]+",
            r"(?i)gate\s*overflow\s*for\s*gate\s*cse",
            r"(?i)click\s+here\s+to\s+view\s+discussion",
            r"(?i)report\s+an\s+error",
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, "", text)

        # Normalize consecutive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def compute_sha256(self, text: str) -> str:
        """Computes exact SHA-256 hash of normalized text."""
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def compute_minhash(self, text: str) -> MinHash:
        """Generates MinHash object from 3-gram character shingles for fuzzy deduplication."""
        m = MinHash(num_perm=self.num_perm)
        tokens = re.sub(r"\s+", " ", text.strip().lower())
        # 3-gram shingles
        for i in range(max(1, len(tokens) - 2)):
            shingle = tokens[i : i + 3].encode("utf-8")
            m.update(shingle)
        return m

    def is_duplicate(self, doc_id: str, text: str) -> bool:
        """Checks both exact hash and fuzzy MinHash similarity.

        Returns True if duplicate.
        """
        sha = self.compute_sha256(text)
        if sha in self.seen_hashes:
            return True

        minhash = self.compute_minhash(text)
        near_matches = self.lsh.query(minhash)
        if near_matches:
            return True

        # Register new item
        self.seen_hashes.add(sha)
        self.lsh.insert(doc_id, minhash)
        return False

    def clean_raw_question(self, raw: RawQuestion) -> Optional[CleanedQuestion]:
        """Runs end-to-end cleaning and deduplication on a raw question."""
        cleaned_body = self.clean_html_to_markdown(raw.raw_question_html)
        if not cleaned_body or len(cleaned_body.strip()) < 10:
            logger.warning(f"Question {raw.id} rejected: empty or too short body.")
            return None

        # Check deduplication on the question body
        if self.is_duplicate(raw.id, cleaned_body):
            logger.info(f"Question {raw.id} skipped: duplicate detected.")
            return None

        # Clean options
        cleaned_options = {}
        for opt_key, opt_html in raw.raw_options_html.items():
            cleaned_opt = self.clean_html_to_markdown(opt_html)
            cleaned_options[opt_key.upper().strip()] = cleaned_opt

        # Clean explanation
        cleaned_explanation = ""
        if raw.raw_explanation_html:
            cleaned_explanation = self.clean_html_to_markdown(raw.raw_explanation_html)

        # Normalize subject to enum
        try:
            subject_enum = Subject(raw.subject)
        except ValueError:
            # Fallback mapping
            subject_map = {
                "algo": Subject.ALGORITHMS,
                "algorithms": Subject.ALGORITHMS,
                "os": Subject.OPERATING_SYSTEMS,
                "operating systems": Subject.OPERATING_SYSTEMS,
                "dbms": Subject.DBMS,
                "database": Subject.DBMS,
                "cn": Subject.COMPUTER_NETWORKS,
                "computer networks": Subject.COMPUTER_NETWORKS,
                "networks": Subject.COMPUTER_NETWORKS,
                "toc": Subject.THEORY_OF_COMPUTATION,
                "theory of computation": Subject.THEORY_OF_COMPUTATION,
                "automata": Subject.THEORY_OF_COMPUTATION,
                "compiler": Subject.COMPILER_DESIGN,
                "compiler design": Subject.COMPILER_DESIGN,
                "cd": Subject.COMPILER_DESIGN,
                "digital": Subject.DIGITAL_LOGIC,
                "digital logic": Subject.DIGITAL_LOGIC,
                "dld": Subject.DIGITAL_LOGIC,
            }
            sub_key = raw.subject.lower().strip()
            subject_enum = subject_map.get(sub_key, Subject.ALGORITHMS)

        content_hash = self.compute_sha256(cleaned_body)

        return CleanedQuestion(
            id=raw.id,
            source_url=raw.source_url,
            source_name=raw.source_name,
            subject=subject_enum,
            topic=raw.topic or "General",
            year=raw.year,
            question_type=raw.question_type,
            marks=raw.marks or 1,
            question_text=cleaned_body,
            options=cleaned_options,
            correct_answer=raw.correct_answer.strip().upper(),
            raw_explanation=cleaned_explanation,
            content_hash=content_hash,
        )
