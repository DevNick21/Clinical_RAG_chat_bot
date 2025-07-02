"""Entity extraction utilities"""
import re
import json
from typing import Dict, Any, List
from config import SECTION_KEYWORDS
from langchain.prompts import ChatPromptTemplate
from invoke import safe_llm_invoke

# Entity extraction prompt for LLM
entity_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical entity extraction assistant. Extract admission IDs and medical sections from user queries.

Available sections: "diagnoses", "procedures", "labs", "microbiology", "prescriptions", "header"

Section mapping (flexible):
- "medications", "drugs", "meds" → "prescriptions"
- "laboratory", "lab results", "tests" → "labs"  
- "diagnosis", "conditions", "diseases" → "diagnoses"
- "procedures", "operations", "surgery" → "procedures"
- "micro", "cultures", "infections" → "microbiology"

Extract information and return JSON format:
{{
    "hadm_id": <number or null>,
    "section": "<section_name or null>",
    "confidence": "high|medium|low",
    "reasoning": "<explanation of extraction>",
    "needs_clarification": <boolean>
}}

Rules:
- hadm_id: Extract only explicit admission IDs (numbers)
- section: Map to available sections, null if unclear
- confidence: "high" for explicit mentions, "medium" for probable, "low" for ambiguous
- needs_clarification: true if multiple possibilities or unclear
- reasoning: Explain your extraction logic

Examples:
- "Does admission 12345 have diabetes?" → hadm_id: 12345, section: "diagnoses", confidence: "high"
- "What medications was the patient on?" → hadm_id: null, section: "prescriptions", confidence: "medium"
- "Show me lab results" → hadm_id: null, section: "labs", confidence: "high"
"""),
    ("human", "{query}")
])


def extract_entities(query: str, use_llm_fallback: bool = True, llm=None) -> Dict[str, Any]:
    """Extract entities from a user query using regex and LLM fallback"""
    print(f"Extracting entities from: '{query}'")

    result = {
        "hadm_id": None,
        "section": None,
        "confidence": "low",
        "reasoning": "",
        "needs_clarification": False
    }

    query_lower = query.lower()

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
                        result["reasoning"] = f"Found explicit hadm_id {match} in query"
                        print(f"📝 Regex found hadm_id: {result['hadm_id']}")
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
            print(f"📝 Regex found section: {section}")
            break

    # Use LLM if regex failed completely AND user wants LLM fallback
    if use_llm_fallback and llm and result["hadm_id"] is None and result["section"] is None:
        print("📝 Regex extraction failed, trying LLM fallback...")

        try:
            # Prepare LLM prompt
            extraction_chain = entity_extraction_prompt | llm
            response = safe_llm_invoke(
                extraction_chain,
                {"query": query},
                fallback_message='{"hadm_id": null, "section": null, "confidence": "low", "needs_clarification": true}',
                context="Entity extraction"
            )

            # Parse LLM response
            try:
                if isinstance(response, str):
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        response = json_match.group()
                    llm_entities = json.loads(response)

                # Use LLM results if they're valid
                if llm_entities.get("hadm_id") and result["hadm_id"] is None:
                    result["hadm_id"] = int(llm_entities["hadm_id"])
                    result["reasoning"] += " LLM extracted hadm_id"

                if llm_entities.get("section") and result["section"] is None:
                    result["section"] = llm_entities["section"]
                    result["reasoning"] += f" LLM extracted section '{result['section']}'"

                if result["hadm_id"] or result["section"]:
                    result["confidence"] = "medium"
                else:
                    result["needs_clarification"] = True

            except json.JSONDecodeError:
                print(f"⚠️ LLM response not valid JSON: {response}")
                result["needs_clarification"] = True

        except Exception as e:
            print(f"⚠️ LLM fallback failed: {e}")
            result["needs_clarification"] = True

    # Set needs_clarification if nothing found
    if result["hadm_id"] is None and result["section"] is None:
        result["needs_clarification"] = True
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
                        f"📝 Found hadm_id {context['hadm_id']} in chat history (from {len(valid_hadm_ids)} candidates)")
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
        query_hadm_matches = extract_entities(
            current_query, use_llm_fallback=False).get("hadm_id")
        if query_hadm_matches:
            for match_group in query_hadm_matches:
                for match in match_group:
                    if match and len(match) >= 8:
                        try:
                            context["hadm_id"] = int(match)
                            context["confidence"] = "high"
                            print(
                                f"📝 Found hadm_id {context['hadm_id']} in current query")
                            break
                        except (ValueError, TypeError):
                            continue

        # Look for section keywords in current query
        for section, keywords in SECTION_KEYWORDS.items():
            if any(keyword in current_query for keyword in keywords):
                context["section"] = section
                print(
                    f"📝 Found section '{context['section']}' in current query")
                break

    return context


def ask_for_clarification(entities: Dict[str, Any], available_options: Dict[str, List]) -> Dict[str, Any]:
    """Generate clarification questions based on extracted entities and available options"""
    clarifications = []

    # Check if hadm_id needs clarification
    if entities.get("hadm_id") is None and entities.get("confidence") != "high":
        if available_options.get("hadm_ids"):
            hadm_list = available_options["hadm_ids"][:5]  # Show first 5
            clarifications.append(
                f"Which admission ID? Available: {', '.join(map(str, hadm_list))}")

    # Check if section needs clarification
    if entities.get("section") is None:
        available_sections = ["diagnoses", "procedures",
                              "labs", "microbiology", "prescriptions"]
        clarifications.append(
            f"Which section? Available: {', '.join(available_sections)}")

    return {
        "needs_clarification": len(clarifications) > 0,
        "clarification_questions": clarifications,
        "suggested_format": "Please specify like: 'admission 12345 diagnoses' or 'patient medications'"
    }
