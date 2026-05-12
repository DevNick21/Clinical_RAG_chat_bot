"""Cross-reference generated answers against retrieved documents.

Flags sentences in the answer that contain *specific* claims (numbers,
dosages, ICD-like codes, admission IDs) where those tokens do not appear
in any retrieved document. Heuristic-driven and intentionally tolerant
of false positives — this is a review aid, not a gate. A flagged claim
means "a human should glance at this," not "the answer is wrong."
"""
import re
from typing import Any, List

# Patterns identifying tokens that should be verifiable against retrieved docs
SPECIFIC_PATTERNS = [
    r"\b\d+\.?\d*\s*(?:mg|g|ml|mcg|units?|kg|lb|cm|mm|%|bpm)\b",  # dosages, measurements
    r"\b[A-Z]\d{2,4}(?:\.\d+)?\b",                                  # ICD-like codes (e.g. J18.9)
    r"\b\d{6,9}\b",                                                 # admission/patient IDs
    r"\b\d+\.\d+\b",                                                # generic numeric values with decimals
]
_PATTERN_RE = re.compile("|".join(SPECIFIC_PATTERNS), re.IGNORECASE)

# Boilerplate phrases to skip when scanning sentences
SKIP_PHRASES = (
    "data from mimic-iv",
    "research/education only",
    "source citations:",
    "source:",
)


def _split_sentences(text: str) -> List[str]:
    """Crude sentence split. Adequate for short clinical answers."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _is_boilerplate(sentence: str) -> bool:
    lo = sentence.lower()
    return any(p in lo for p in SKIP_PHRASES)


def check_claims(answer: str, retrieved_docs: List[Any]) -> List[str]:
    """Return sentences whose specific tokens don't appear in any retrieved doc.

    Args:
        answer: The full model-generated answer text.
        retrieved_docs: The Documents used as context for the answer.

    Returns:
        List of sentences flagged for review. Empty if all specific claims
        are grounded, or if there are no specific claims to verify.
    """
    if not retrieved_docs or not answer:
        return []

    doc_text = " ".join(
        getattr(d, "page_content", "") for d in retrieved_docs
    ).lower()

    flagged: List[str] = []
    for sentence in _split_sentences(answer):
        if _is_boilerplate(sentence):
            continue
        matches = _PATTERN_RE.findall(sentence)
        if not matches:
            continue  # no specific claim → nothing to verify
        if not all(m and m.lower() in doc_text for m in matches):
            flagged.append(sentence)

    return flagged
