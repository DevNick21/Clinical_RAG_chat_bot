"""Content shaping for the clinical RAG pipeline.

Holds the post-retrieval / pre-LLM and post-LLM text transforms:

  * SECTION_RULES — per-section keyword + regex + line-cap policy
  * extract_structured_content / extract_clinical_content — turn raw
    chunked docs into a section-titled, deduplicated context block
  * validate_and_fix_response — post-LLM patch for missing disclaimers
    and wrong "Source Citations: None provided" claims
  * no_records_text / preflight_rejection_text — uniform "we found
    nothing" messages used by the orchestrator and the streaming path

All methods are stateless. Pulled out of ClinicalRAGBot during the
god-class decomposition so the orchestrator stops owning text rules.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Union

from langchain.schema import Document

from RAG_chat_pipeline.core.retriever import Retriever


IdLike = Union[int, List[int], None]


class ContentProcessor:
    """Stateless content shaper: extract → structure → patch."""

    # Standard MIMIC-IV research disclaimer. Used by no_records_text and
    # preflight_rejection_text so all "no answer" paths end the same way.
    DISCLAIMER = "ℹ️ Data from MIMIC-IV database for research/education only."

    # Section-aware extraction rules. Each entry decides which lines from
    # a chunk are "interesting" for that section and caps the count so a
    # single noisy doc can't blow the prompt budget.
    SECTION_RULES = {
        "diagnoses": {
            "title": "ADMISSION {hadm_id} DIAGNOSES:",
            "keywords": ["icd", "diagnosis", "diagnoses", "condition", "disorder"],
            "regexes": [r"\b[A-Z]?\d{2,5}(?:\.\d+)?\b"],
            "max_lines": 12,
        },
        "prescriptions": {
            "title": "ADMISSION {hadm_id} MEDICATIONS:",
            "keywords": ["medication", "drug", "prescription", "dose", "mg", "tablet", "capsule"],
            "regexes": [r"\d+\s*(mg|g|ml|units?)"],
            "max_lines": 10,
        },
        "labs": {
            "title": "ADMISSION {hadm_id} LAB RESULTS:",
            "keywords": ["lab", "test", "result", "value", "normal", "abnormal", "high", "low"],
            "regexes": [r"\d+\.?\d*\s*[a-zA-Z/%]*"],
            "max_lines": 12,
        },
        "labevents": {  # alias for labs
            "title": "ADMISSION {hadm_id} LAB RESULTS:",
            "keywords": ["lab", "test", "result", "value", "normal", "abnormal", "high", "low"],
            "regexes": [r"\d+\.?\d*\s*[a-zA-Z/%]*"],
            "max_lines": 12,
        },
        "default": {
            "title": "ADMISSION {hadm_id} {section}:",
            "keywords": [],
            "regexes": [],
            "max_lines": 8,
        },
    }

    # ------------------------------------------------------------------ #
    # Message formatters (called when we have no answer to give)
    # ------------------------------------------------------------------ #
    @classmethod
    def no_records_text(cls, entity_type: str, entity_id: IdLike,
                        section: Optional[str] = None) -> str:
        """Filter-miss message: index returned no docs for the requested IDs.

        `entity_type` is the singular noun ("admission" / "patient/subject");
        `entity_id` is int OR list[int]; `section` is an optional string.
        """
        ids = Retriever.to_id_list(entity_id)
        section_msg = f" in section '{section}'" if section else ""
        if len(ids) > 1:
            label = f"{entity_type}s"
            id_str = ", ".join(str(i) for i in ids)
        elif ids:
            label = entity_type
            id_str = str(ids[0])
        else:
            return f"No records found{section_msg}"
        return f"No records found for {label} {id_str}{section_msg}"

    @classmethod
    def preflight_rejection_text(cls, missing_ids: Iterable[int]) -> str:
        """Post-retrieval rejection: docs came back but for different IDs.

        Different from the filter-miss path because here the user asked
        about specific IDs and global / mixed retrieval gave back
        unrelated admissions; we want to be explicit about why we're
        not answering.
        """
        ids_str = ", ".join(str(i) for i in sorted(missing_ids))
        return (
            f"No records were found for admission/subject ID(s): {ids_str}. "
            "The retrieved documents are about different admissions, so I "
            "can't answer this question reliably. Please verify the ID(s) "
            f"or rephrase the question.\n\n{cls.DISCLAIMER}"
        )

    # ------------------------------------------------------------------ #
    # Pre-LLM: turn raw chunks into a structured context block
    # ------------------------------------------------------------------ #
    @classmethod
    def extract_structured_content(cls, section: Optional[str], content: str, hadm_id) -> str:
        """Score a single chunk's lines by section relevance, keep top-N."""
        section_key = (section or "").lower() or "default"
        rules = cls.SECTION_RULES.get(section_key, cls.SECTION_RULES["default"])

        lines = [ln.strip() for ln in content.split('\n') if ln.strip()]
        scored = []
        for ln in lines:
            ln_lower = ln.lower()
            if any(kw in ln_lower for kw in rules["keywords"]):
                scored.append((2, ln))  # keyword hit
            elif any(re.search(rx, ln) for rx in rules["regexes"]):
                scored.append((1, ln))  # regex hit
            elif len(ln) > 10 and len(scored) < 3:
                scored.append((0, ln))  # fallback: a few substantial lines

        scored.sort(key=lambda x: (-x[0]))
        max_lines = rules.get("max_lines", 10)
        selected = [ln for _, ln in scored[:max_lines]]

        title = rules["title"].format(
            hadm_id=hadm_id, section=(section or "UNKNOWN").upper())
        if selected:
            return f"{title}\n" + "\n".join(selected)
        return f"{title}\n" + content[:600]

    @classmethod
    def extract_clinical_content(cls, docs: List[Document]) -> str:
        """Apply extract_structured_content to every doc and join with blank lines."""
        out: List[str] = []
        for doc in docs:
            section = doc.metadata.get('section', '')
            hadm_id = doc.metadata.get('hadm_id', 'Unknown')
            out.append(cls.extract_structured_content(section, doc.page_content, hadm_id))
        return "\n\n".join(out)

    # ------------------------------------------------------------------ #
    # Post-LLM: patch common formatting issues
    # ------------------------------------------------------------------ #
    @classmethod
    def validate_and_fix_response(cls, answer: str, retrieved_docs: List[Document],
                                  hadm_id=None) -> str:
        """Patch missing disclaimer + wrong "Source Citations: None provided" claims.

        The LLM sometimes claims it had no source citations even when we
        gave it documents. That happens enough to be worth a deterministic
        fix-up rather than a longer prompt.
        """
        if "Data from MIMIC-IV database for research/education only" not in answer:
            answer += "\n\n Data from MIMIC-IV database for research/education only."

        doc_count = len(retrieved_docs)
        if doc_count == 0:
            return answer

        # Both plain and bold variants of the wrong-citations claim.
        for needle in ("Source Citations: None provided",
                       "**Source Citations**: None provided"):
            if needle in answer:
                bold = needle.startswith("**")
                if hadm_id:
                    fix = (f"{'**Source Citations**' if bold else 'Source Citations'}: "
                           f"From {doc_count} documents for admission {hadm_id}")
                else:
                    fix = (f"{'**Source Citations**' if bold else 'Source Citations'}: "
                           f"From {doc_count} retrieved documents")
                answer = answer.replace(needle, fix)

        return answer
