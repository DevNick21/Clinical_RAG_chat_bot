"""Cross-reference generated answers against retrieved documents.

Two-stage faithfulness check:

  1. **Regex pre-filter** (`check_claims`) — flags sentences containing
     *specific* claims (numbers, dosages, ICD-like codes, admission IDs)
     where those tokens don't literal-match any retrieved doc. Sub-ms,
     deterministic, runs in every audit. Tolerant of false positives —
     a flagged claim means "a human should glance at this," not "the
     answer is wrong."

  2. **LLM-as-judge** (`llm_judge_claims`) — only the regex-flagged
     sentences are sent to the capable audit model (`get_audit_llm`).
     The judge classifies each as supported / not_supported / unclear
     against the actual retrieved documents. Returns the refined set
     PLUS a per-sentence rationale, so unsupported claims surface in
     the audit log with reasoning rather than just a token-mismatch.

Stage 2 is post-stream and bounded (only operates on what stage 1
flagged, typically 0-3 sentences), so it doesn't touch TTFP.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Stage 2: LLM-as-judge over the regex-flagged sentences
# ---------------------------------------------------------------------------

# Tight JSON-only prompt. Capable models (gpt-5-nano / gpt-4.1 / gpt-4o)
# follow this format reliably without examples; small models would need few-shot.
# The "unclear" verdict exists so the judge can flag sentences where
# the source docs neither support nor refute the claim — that's a
# genuinely different audit signal than "unsupported."
_JUDGE_PROMPT = """You are a clinical faithfulness auditor. Decide whether each
candidate claim from a generated answer is supported by the retrieved
source documents.

Source documents:
{sources}

Candidate claims (one per line, indexed):
{claims}

For each claim, output one JSON object per line (JSON Lines), no commentary:
{{"index": <int>, "verdict": "supported" | "not_supported" | "unclear", "reason": "<one short sentence>"}}

Rules:
- "supported": every specific number / code / ID in the claim appears verbatim or as a clear paraphrase in the sources.
- "not_supported": the claim asserts a specific value / code / ID that contradicts or is absent from the sources.
- "unclear": the sources neither confirm nor refute (ambiguous wording, missing context).
- Do not output anything except the JSONL lines."""


def _format_sources(retrieved_docs: List[Any], char_budget: int = 6000) -> str:
    """Concatenate retrieved doc contents under a hard char budget.

    Bounded to keep the audit prompt cheap. 6k chars ≈ 1500 tokens — plenty
    of room for 5-10 short MIMIC chunks. If the budget is exceeded we
    truncate the tail; the regex flags we're judging are already very
    targeted, so the head of each doc almost always contains the relevant
    tokens.
    """
    parts: List[str] = []
    used = 0
    for i, d in enumerate(retrieved_docs):
        content = getattr(d, "page_content", "") or ""
        hadm = (getattr(d, "metadata", {}) or {}).get("hadm_id", "?")
        section = (getattr(d, "metadata", {}) or {}).get("section", "?")
        header = f"[doc {i} hadm_id={hadm} section={section}]\n"
        remaining = char_budget - used - len(header)
        if remaining <= 0:
            break
        body = content[:remaining]
        parts.append(header + body)
        used += len(header) + len(body)
    return "\n\n".join(parts)


def _parse_jsonl_verdicts(raw: str, expected_n: int) -> List[Dict[str, Any]]:
    """Parse JSONL output from the judge. Tolerates surrounding chatter.

    Capable models occasionally prefix with ```json fences or trailing
    explanations even when told not to — we extract per-line JSON
    objects and skip anything unparseable. Length mismatch is logged but
    not fatal: missing entries default to "unclear" so the audit log
    captures the gap.
    """
    verdicts: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip().strip("`")
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "index" in obj and "verdict" in obj:
                verdicts.append(obj)
        except json.JSONDecodeError:
            continue

    if len(verdicts) != expected_n:
        logger.debug(
            "Judge returned %d verdicts for %d claims; will default missing to 'unclear'",
            len(verdicts), expected_n,
        )
    return verdicts


def llm_judge_claims(
    flagged_sentences: List[str],
    retrieved_docs: List[Any],
    audit_llm: Optional[Any],
) -> List[Dict[str, Any]]:
    """Run LLM-as-judge over the regex-flagged sentences.

    Args:
        flagged_sentences: Output of `check_claims` (regex stage).
        retrieved_docs: Same docs that were passed to the answer model.
        audit_llm: A LangChain BaseChatModel from `get_audit_llm`. When
                   None, returns an empty list (judge disabled).

    Returns:
        List of {sentence, verdict, reason} dicts, one per flagged
        sentence, even if the judge call fails (missing entries default
        to verdict='unclear', reason='judge unavailable').
    """
    if not flagged_sentences or audit_llm is None or not retrieved_docs:
        return []

    sources = _format_sources(retrieved_docs)
    claims_block = "\n".join(f"{i}: {s}" for i, s in enumerate(flagged_sentences))
    prompt = _JUDGE_PROMPT.format(sources=sources, claims=claims_block)

    try:
        response = audit_llm.invoke(prompt)
        raw = getattr(response, "content", None) or str(response)
    except Exception as e:  # noqa: BLE001 — audit must never break the request
        logger.warning("LLM-as-judge call failed: %s", e)
        return [
            {"sentence": s, "verdict": "unclear", "reason": f"judge unavailable: {e}"}
            for s in flagged_sentences
        ]

    verdicts_by_idx = {
        v.get("index"): v for v in _parse_jsonl_verdicts(raw, len(flagged_sentences))
    }

    out: List[Dict[str, Any]] = []
    for i, sentence in enumerate(flagged_sentences):
        v = verdicts_by_idx.get(i)
        if v is None:
            out.append({
                "sentence": sentence,
                "verdict": "unclear",
                "reason": "no verdict returned",
            })
            continue
        verdict = v.get("verdict")
        if verdict not in ("supported", "not_supported", "unclear"):
            verdict = "unclear"
        out.append({
            "sentence": sentence,
            "verdict": verdict,
            "reason": (v.get("reason") or "").strip()[:300],
        })
    return out
