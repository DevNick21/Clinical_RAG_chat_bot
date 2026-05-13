"""Entity extraction utilities.

Two-tier extractor: regex first (fast, free, deterministic), LLM fallback
only when regex finds nothing useful. The LLM call is intentionally rare
because most clinical lookups contain explicit IDs or section keywords
that regex catches outright.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from RAG_chat_pipeline.config.settings import get_settings

logger = logging.getLogger(__name__)
SECTION_KEYWORDS = get_settings().section_keywords

# Sections the LLM is allowed to predict. Anything else gets dropped.
_VALID_SECTIONS = {
    "diagnoses",
    "procedures",
    "labs",
    "prescriptions",
    "microbiology",
    "header",
    "transfers",
}

# Tightly-scoped JSON-only prompt. Keeps the LLM's job small so even at
# minimal reasoning effort it returns a clean, parseable answer in 1-2s.
_LLM_EXTRACTION_PROMPT = """Extract structured entities from this clinical question. Return ONLY valid JSON, no commentary.

Question: {query}

Schema (use null when the field cannot be determined confidently):
{{
  "hadm_id":    integer or null,
  "subject_id": integer or null,
  "section":    one of ["diagnoses","procedures","labs","prescriptions","microbiology","header","transfers"] or null
}}

Return only the JSON object. Do not wrap in markdown fences."""


def _llm_extract_entities(query: str, llm: Any) -> Optional[Dict[str, Any]]:
    """Best-effort LLM-driven extraction. Returns None on any failure.

    Strips common code-fence wrapping that chat models add even when told
    not to. Validates the section against the allowed enum so a
    hallucinated section name doesn't poison downstream filtering.
    """
    try:
        prompt = _LLM_EXTRACTION_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        text = text.strip()

        # Strip ```json ... ``` or ``` ... ``` if the model ignored the
        # 'no fences' instruction.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None

        section = parsed.get("section")
        if section is not None and section not in _VALID_SECTIONS:
            section = None

        return {
            "hadm_id":    parsed.get("hadm_id") if isinstance(parsed.get("hadm_id"), int) else None,
            "subject_id": parsed.get("subject_id") if isinstance(parsed.get("subject_id"), int) else None,
            "section":    section,
        }
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.debug("LLM entity extraction failed (non-fatal): %s", e)
        return None
    except Exception as e:
        # Catch-all: never block the main flow on an extraction-LLM hiccup.
        logger.warning("Unexpected LLM entity extraction error: %s", e)
        return None


def extract_entities(query: str, use_llm_fallback: bool = True, llm=None) -> Dict[str, Any]:
    """Extract entities from a user query.

    Two-tier:
      1. Regex first — catches explicit IDs ("admission 21342515",
         "patient 10006508", any 8-digit number) and section keywords
         from SECTION_KEYWORDS. Sub-millisecond, free, deterministic.
      2. LLM fallback — only fires when regex finds NOTHING useful
         AND use_llm_fallback=True AND a usable llm is provided.
         Handles indirect phrasing like "the diabetic patient's most
         recent admission" or "the cardiac case's medications".
         Strict JSON-only prompt, validated against the allowed
         section enum.
    """
    logger.debug("Extracting entities from query")

    result = {
        "hadm_id": None,
        "hadm_ids": [],         # ALL hadm_ids found, in order. hadm_id is hadm_ids[0] when present.
        "subject_id": None,
        "subject_ids": [],
        "section": None,
        "confidence": "low",
        "reasoning": "",
        "query_type": "single_admission"  # single_admission or patient_history
    }

    query_lower = query.lower()

    # Check for patient/subject queries first
    patient_keywords = ["patient", "subject", "all admissions",
                        "patient history", "previous admissions"]
    if any(keyword in query_lower for keyword in patient_keywords):
        result["query_type"] = "patient_history"
        result["confidence"] = "medium"
        result["reasoning"] = "Query appears to request patient-level information"

    # Extract ALL subject_ids mentioned (not just the last). Handles
    # plural / multi-ID queries like "compare patient X and patient Y".
    # `admissions?\s*` (with optional s) catches "admission 123" and
    # "admissions 123 and 456".
    subj_matches = re.findall(
        r'subjects?\s*(\d+)|patients?\s*(\d+)|subject_id[:\s]*(\d+)', query_lower)
    for match_group in subj_matches:
        for match in match_group:
            if not match:
                continue
            try:
                sid = int(match)
                if sid not in result["subject_ids"]:
                    result["subject_ids"].append(sid)
            except (ValueError, TypeError):
                continue
    if result["subject_ids"]:
        result["subject_id"] = result["subject_ids"][0]  # backward compat: first
        result["confidence"] = "high"
        result["reasoning"] += f" Found {len(result['subject_ids'])} subject_id(s): {result['subject_ids']}"
        result["query_type"] = "patient_history"
        logger.debug("Regex found subject_ids")

    # Extract ALL hadm_ids mentioned. Same multi-ID treatment as above
    # so a comparison query like "admissions 21342515 and 22240591"
    # captures BOTH instead of just the last (which was the original
    # behaviour and caused filter misses).
    hadm_matches = re.findall(
        r'admissions?\s*(\d+)|hadm_id[:\s]*(\d+)|\b(\d{8})\b', query_lower)
    for match_group in hadm_matches:
        for match in match_group:
            if not match or len(match) < 8:  # MIMIC hadm_ids are 8+ digits
                continue
            try:
                hid = int(match)
                if hid not in result["hadm_ids"]:
                    result["hadm_ids"].append(hid)
            except (ValueError, TypeError):
                continue
    # If a number was already captured as subject_id from an explicit
    # "patient"/"subject" prefix, drop it from hadm_ids — the explicit
    # prefix wins over the generic \b\d{8}\b fallback. Avoids the false
    # positive where "Show me labs for patient 10006508" would otherwise
    # set BOTH subject_id=10006508 AND hadm_id=10006508.
    if result["subject_ids"]:
        result["hadm_ids"] = [
            hid for hid in result["hadm_ids"] if hid not in result["subject_ids"]
        ]

    if result["hadm_ids"]:
        result["hadm_id"] = result["hadm_ids"][0]  # backward compat: first
        result["confidence"] = "high"
        result["reasoning"] += f" Found {len(result['hadm_ids'])} hadm_id(s): {result['hadm_ids']}"
        logger.debug("Regex found hadm_ids")
    else:
        # If subject_ids consumed all the 8-digit numbers, clear hadm_id
        # too (we previously set it to the first number, but that number
        # is now known to be a subject_id, not an hadm_id).
        if result["subject_ids"]:
            result["hadm_id"] = None

    # Keyword matching for sections
    for section, keywords in SECTION_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            result["section"] = section
            if result["confidence"] == "low":
                result["confidence"] = "medium"
            result["reasoning"] += f" Found section keywords for '{section}'"
            logger.debug("Regex found section")
            break

    # LLM fallback — only fires when regex found absolutely nothing.
    # By construction this skips the common path (queries with explicit
    # IDs or section keywords), so the LLM round-trip cost is paid only
    # for genuinely ambiguous free-form clinical questions.
    nothing_useful = (
        result["hadm_id"] is None
        and result["subject_id"] is None
        and result["section"] is None
    )
    if nothing_useful and use_llm_fallback and llm is not None:
        llm_result = _llm_extract_entities(query, llm)
        if llm_result and any(llm_result.get(k) for k in ("hadm_id", "subject_id", "section")):
            # Merge LLM-found values, never overwriting a regex hit
            for key in ("hadm_id", "subject_id", "section"):
                if result[key] is None and llm_result.get(key) is not None:
                    result[key] = llm_result[key]
            # LLM-derived results are medium confidence (vs high for regex hits)
            result["confidence"] = "medium"
            result["reasoning"] = "LLM fallback extracted entities from free-form question"
            logger.debug("LLM fallback extracted entities")

    # Set final confidence based on what was found
    if result["hadm_id"] is None and result["section"] is None and result["subject_id"] is None:
        result["reasoning"] = "No entities extracted from query"

    logger.debug("Final extraction result computed")
    return result


def extract_context_from_chat_history(chat_history: List, current_query: str) -> Dict[str, Any]:
    """Extract hadm_id and section context from chat history"""
    context = {"hadm_id": None, "section": None, "confidence": "low"}

    if not chat_history:
        return context

    # Look through recent chat history for hadm_id mentions
    # Last 3 exchanges (user and assistant)
    recent_messages = chat_history[-6:]

    for role, message in reversed(recent_messages):
        if isinstance(message, str):
            # Look for explicit admission IDs
            hadm_matches = re.findall(
                r'admission\s*(\d+)|hadm_id[:\s]*(\d+)|\b(\d{8})\b', message.lower())
            if hadm_matches:
                # Extract the valid hadm_ids found
                valid_hadm_ids = []
                for match_group in hadm_matches:
                    for match in match_group:
                        if match and len(match) >= 8:  # Reasonable hadm_id length
                            try:
                                hadm_id_val = int(match)
                                valid_hadm_ids.append(hadm_id_val)
                            except (ValueError, TypeError):
                                continue
                if valid_hadm_ids:
                    # Use the last found hadm_id
                    # Use the last found hadm_id
                    context["hadm_id"] = valid_hadm_ids[-1]
                    context["confidence"] = "high" if len(
                        valid_hadm_ids) == 1 else "medium"
                    logger.debug("Found hadm_id in chat history")
                    break

    # Look for section context in recent messages
    section_matches = []
    for role, message in reversed(recent_messages):
        if isinstance(message, str):
            message_lower = message.lower()
            for section, keywords in SECTION_KEYWORDS.items():
                if any(keyword in message_lower for keyword in keywords):
                    # Store section and role
                    section_matches.append((section, role))
                    break
    # If multiple sections mentioned, use the most recent one
    if section_matches:
        context["section"] = section_matches[0][0]  # Most recent
        context_role = section_matches[0][1]  # Role that mentioned it
        logger.debug("Found section context in chat history")

    # Use current_query to enhance context if no history context found
    if context["hadm_id"] is None and context["section"] is None:
        # Check if current query contains context clues

        # Look for hadm_id in current query
        query_entities = extract_entities(current_query)
        query_hadm_id = query_entities.get("hadm_id")
        if query_hadm_id:
            context["hadm_id"] = query_hadm_id
            context["confidence"] = "high"
            logger.debug("Found hadm_id in current query")

        # Look for section keywords in current query
        for section, keywords in SECTION_KEYWORDS.items():
            if any(keyword in current_query for keyword in keywords):
                context["section"] = section
                logger.debug("Found section in current query")
                break

    return context
