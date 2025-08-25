"""Entity extraction utilities"""
import re
from typing import Dict, Any, List
from RAG_chat_pipeline.config.config import SECTION_KEYWORDS


def extract_entities(query: str, use_llm_fallback: bool = True, llm=None) -> Dict[str, Any]:
    """Extract entities from a user query using regex only

    Only regex-based extraction for robustness and predictability.
    Now includes subject_id extraction for patient-level queries.
    """
    print(f"Extracting entities from: '{query}'")

    result = {
        "hadm_id": None,
        "subject_id": None,
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

    # Extract subject_id if explicitly mentioned
    subj_matches = re.findall(
        r'subject[_\s]*(\d+)|patient[_\s]*(\d+)|subject_id[:\s]*(\d+)', query_lower)
    if subj_matches:
        for match_group in subj_matches:
            for match in match_group:
                if match:
                    try:
                        result["subject_id"] = int(match)
                        result["confidence"] = "high"
                        result["reasoning"] += f" Found explicit subject_id {match}"
                        result["query_type"] = "patient_history"
                        print(
                            f" Regex found subject_id: {result['subject_id']}")
                        break
                    except (ValueError, TypeError):
                        continue

    # Regex extraction for hadm_id
    hadm_matches = re.findall(
        r'admission\s*(\d+)|hadm_id[:\s]*(\d+)|\b(\d{8})\b', query_lower)
    if hadm_matches:
        for match_group in hadm_matches:
            for match in match_group:
                if match and len(match) >= 8:  # Reasonable hadm_id length
                    try:
                        result["hadm_id"] = int(match)
                        result["confidence"] = "high"
                        result["reasoning"] += f" Found explicit hadm_id {match}"
                        print(f" Regex found hadm_id: {result['hadm_id']}")
                        break
                    except (ValueError, TypeError):
                        continue

    # Keyword matching for sections
    for section, keywords in SECTION_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            result["section"] = section
            if result["confidence"] == "low":
                result["confidence"] = "medium"
            result["reasoning"] += f" Found section keywords for '{section}'"
            print(f" Regex found section: {section}")
            break

    # Set final confidence based on what was found
    if result["hadm_id"] is None and result["section"] is None:
        result["reasoning"] = "No entities extracted from query"

    print(f"Final extraction result: {result}")
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
                    print(
                        f" Found hadm_id {context['hadm_id']} in chat history (from {len(valid_hadm_ids)} candidates)")
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
        print(
            f"Found section '{context['section']}' context from {context_role} message")

    # Use current_query to enhance context if no history context found
    if context["hadm_id"] is None and context["section"] is None:
        # Check if current query contains context clues

        # Look for hadm_id in current query
        query_entities = extract_entities(current_query)
        query_hadm_id = query_entities.get("hadm_id")
        if query_hadm_id:
            context["hadm_id"] = query_hadm_id
            context["confidence"] = "high"
            print(f" Found hadm_id {context['hadm_id']} in current query")

        # Look for section keywords in current query
        for section, keywords in SECTION_KEYWORDS.items():
            if any(keyword in current_query for keyword in keywords):
                context["section"] = section
                print(
                    f" Found section '{context['section']}' in current query")
                break

    return context
