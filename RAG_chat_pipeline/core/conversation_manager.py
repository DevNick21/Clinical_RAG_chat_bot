"""Conversation handling for the clinical RAG pipeline.

Owns everything between "raw HTTP/CLI input arrives" and "retriever is
called with a clean question + filter params":

  * Input validation (question text + filter params)
  * Entity extraction (regex first, LLM fallback)
  * Chat-history shape conversion (dict / tuple / list -> internal tuples)
  * Chat-history truncation (MAX_CHAT_HISTORY)
  * Question rephrasing for short follow-ups (LLM-assisted, safe fallback
    to a template question if rephrasing looks hallucinated)

Pulled out of ClinicalRAGBot during the god-class decomposition. The
orchestrator composes a ConversationManager and delegates rather than
owning these flows directly.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from RAG_chat_pipeline.helper.entity_extraction import (
    extract_context_from_chat_history,
    extract_entities,
)
from RAG_chat_pipeline.helper.invoke import safe_llm_invoke
from RAG_chat_pipeline.utils.logger import ClinicalLogger


# Condense-question prompt. Tightly scoped so the LLM rephrases a short
# follow-up using ONLY the prior chat — no inventing medical terms.
_CONDENSE_Q_TEMPLATE = """Rephrase the follow-up question as a standalone medical question using ONLY the context from chat history.

RULES:
1. Preserve admission IDs (hadm_id), patient IDs (subject_id), and section references
2. Only add context explicitly mentioned in chat history
3. Do NOT add new medical terms, conditions, or test names
4. Keep the question concise and focused

Examples:
Chat: "What medications were prescribed for admission 25282710?"
Follow-up: "What are the diagnoses?"
Output: "What diagnoses are recorded for admission 25282710?"

Chat: "Show labs for patient 12345"
Follow-up: "Any abnormal values?"
Output: "What abnormal lab values are there for patient 12345?"

If no relevant context exists, return the original question unchanged."""


class ConversationManager:
    """Validates, enriches, and rephrases incoming conversation turns.

    Holds two model handles:
      * `llm`      — the clinical answer model (kept for backwards
                     compatibility; not currently invoked from here).
      * `fast_llm` — small non-reasoning model used for the two LLM
                     calls on the TTFP critical path: entity extraction
                     fallback and short-follow-up rephrasing. Falls
                     back to `llm` if not supplied (legacy callers).
    """

    def __init__(self, llm, section_keywords: Dict[str, List[str]],
                 max_chat_history: int = 10,
                 enable_rephrasing: bool = True,
                 enable_entity_extraction: bool = True,
                 fast_llm=None):
        self.llm = llm
        self.fast_llm = fast_llm or llm
        self.section_keywords = section_keywords
        self.max_chat_history = max_chat_history
        self.enable_rephrasing = enable_rephrasing
        self.enable_entity_extraction = enable_entity_extraction

        self.condense_q_prompt = ChatPromptTemplate.from_messages([
            ("system", _CONDENSE_Q_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    # ------------------------------------------------------------------ #
    # Validation (stateless)
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_question(question: Any) -> str:
        """Validate and sanitize user input."""
        if not isinstance(question, str):
            raise ValueError("Question must be a string")
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if len(question) < 3:
            raise ValueError("Question is too short (minimum 3 characters)")
        if len(question) > 2000:
            raise ValueError("Question is too long (maximum 2000 characters)")
        # Strip control chars but keep newlines/tabs for medical formatting.
        return ''.join(c for c in question if ord(c) >= 32 or c in '\n\t')

    @staticmethod
    def validate_parameters(hadm_id=None, subject_id=None, section=None, k=None):
        """Validate search parameters; coerce int-like strings to int."""
        if hadm_id is not None:
            if not isinstance(hadm_id, (int, str)):
                raise ValueError("hadm_id must be an integer or string")
            try:
                hadm_id = int(hadm_id)
                if hadm_id <= 0:
                    raise ValueError("hadm_id must be positive")
            except ValueError:
                raise ValueError("hadm_id must be a valid integer")

        if subject_id is not None:
            if not isinstance(subject_id, (int, str)):
                raise ValueError("subject_id must be an integer or string")
            try:
                subject_id = int(subject_id)
                if subject_id <= 0:
                    raise ValueError("subject_id must be positive")
            except ValueError:
                raise ValueError("subject_id must be a valid integer")

        if section is not None:
            if not isinstance(section, str) or not section.strip():
                raise ValueError("section must be a non-empty string")
            section = section.strip().lower()

        if k is not None:
            if not isinstance(k, int) or k <= 0 or k > 100:
                raise ValueError("k must be an integer between 1 and 100")

        return hadm_id, subject_id, section, k

    # ------------------------------------------------------------------ #
    # Chat history shape + truncation
    # ------------------------------------------------------------------ #
    @staticmethod
    def process_api_chat_history(chat_history) -> List[Tuple[str, str]]:
        """Normalise API chat-history shapes to a list of (role, content) tuples.

        Accepts dicts ({"role":..., "content":...}), tuples/lists, and skips
        anything else with a warning.
        """
        if not chat_history:
            return []

        processed: List[Tuple[str, str]] = []
        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if content.strip():
                    processed.append((role, content))
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                role, content = msg[0], msg[1]
                if content.strip():
                    processed.append((role, content))
            else:
                ClinicalLogger.warning(
                    f"Unexpected chat history format: {type(msg)}")
        return processed

    def truncate_history(self, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Cap history at max_chat_history; warn if we had to drop messages."""
        if history and len(history) > self.max_chat_history:
            ClinicalLogger.warning(
                f"Chat history truncated to {self.max_chat_history} messages")
            return history[-self.max_chat_history:]
        return history

    # ------------------------------------------------------------------ #
    # Entity extraction + parameter resolution
    # ------------------------------------------------------------------ #
    def extract_and_validate_params(self, question, hadm_id=None, subject_id=None,
                                    section=None, k=None):
        """Validate inputs then auto-fill missing filters via entity extraction.

        Returns (question, hadm_id, subject_id, section, k, extracted_entities).
        When the user mentions multiple admission/subject IDs, the FULL list is
        surfaced as the filter param so retrieval unions across all of them.
        """
        question = self.validate_question(question)
        hadm_id, subject_id, section, k = self.validate_parameters(
            hadm_id, subject_id, section, k)

        extracted_entities: Optional[Dict[str, Any]] = None
        if (self.enable_entity_extraction
                and hadm_id is None and subject_id is None and section is None):
            try:
                # Route through the FAST model — this is a JSON-NER task,
                # not a reasoning task. Saves ~1s vs the answer model.
                extracted_entities = extract_entities(question, llm=self.fast_llm)
                if extracted_entities["confidence"] in ("high", "medium"):
                    hadm_id, subject_id, section = self._fold_extracted(
                        extracted_entities, hadm_id, subject_id, section)
                    ClinicalLogger.info(
                        f"Auto-extracted - hadm_id: {hadm_id}, subject_id: {subject_id}, section: {section}")
            except Exception as e:
                ClinicalLogger.warning(f"Entity extraction failed: {e}")

        return question, hadm_id, subject_id, section, k, extracted_entities

    @staticmethod
    def _fold_extracted(entities: Dict[str, Any], hadm_id, subject_id, section):
        """Pick filter params from a successful entity-extraction result."""
        ext_hadm = entities.get("hadm_ids") or []
        ext_subj = entities.get("subject_ids") or []

        if len(ext_hadm) > 1:
            hadm_id = ext_hadm
        elif ext_hadm:
            hadm_id = ext_hadm[0]
        elif entities.get("hadm_id") is not None:
            hadm_id = entities["hadm_id"]

        if len(ext_subj) > 1:
            subject_id = ext_subj
        elif ext_subj:
            subject_id = ext_subj[0]
        elif entities.get("subject_id") is not None:
            subject_id = entities["subject_id"]

        section = entities.get("section") or section
        return hadm_id, subject_id, section

    # ------------------------------------------------------------------ #
    # Chat context: extract IDs from history, rephrase short follow-ups
    # ------------------------------------------------------------------ #
    def process_chat_context(self, chat_history, question,
                             hadm_id=None, subject_id=None, section=None):
        """Use chat-history context to fill missing filters; rephrase if needed.

        Returns (search_question, hadm_id, subject_id, section, chat_context).
        Any failure inside falls back to the input unchanged (chat context
        must never block answering).
        """
        if not chat_history:
            ClinicalLogger.debug("No chat history to process")
            return question, hadm_id, subject_id, section, {}

        try:
            chat_context = extract_context_from_chat_history(chat_history, question)

            old_hadm_id, old_subject_id, old_section = hadm_id, subject_id, section
            hadm_id = hadm_id or chat_context.get("hadm_id")
            subject_id = subject_id or chat_context.get("subject_id")
            section = section or chat_context.get("section")

            if hadm_id != old_hadm_id or subject_id != old_subject_id or section != old_section:
                ClinicalLogger.info(
                    f"Updated parameters from chat history - hadm_id: {hadm_id}, section: {section}")

            original_question = question
            needs_rephrasing = (self.enable_rephrasing
                                and self._should_rephrase(question, chat_history, hadm_id, section))
            if needs_rephrasing:
                ClinicalLogger.info("Rephrasing question using chat history...")
                question = self._rephrase_safely(question, chat_history, hadm_id, original_question)

            return question, hadm_id, subject_id, section, chat_context
        except Exception as e:
            ClinicalLogger.error(f"Chat context processing failed: {e}")
            return question, hadm_id, subject_id, section, {}

    def _should_rephrase(self, question, chat_history, hadm_id, section) -> bool:
        """Heuristic: short follow-up with no explicit admission/section context."""
        has_admission_context = "admission" in question.lower() and (
            hadm_id is not None and str(hadm_id) in question
        )
        has_section_context = section and any(
            kw in question.lower() for kw in self.section_keywords.get(section, [])
        )
        return (
            len(chat_history) > 0
            and not (has_admission_context or has_section_context)
            and len(question.split()) < 6
        )

    def _rephrase_safely(self, question, chat_history, hadm_id, original_question) -> str:
        """LLM rephrase with sanity checks; falls back to a template question.

        Uses the FAST model — condense-question is a short-output task
        that small non-reasoning models handle indistinguishably from
        large ones, at ~300ms instead of ~1.5s.
        """
        try:
            rephrased = safe_llm_invoke(
                self.fast_llm,
                self.condense_q_prompt.format_messages(
                    chat_history=chat_history, input=question),
                fallback_message=question,
                context="Question rephrasing",
            )
            if not isinstance(rephrased, str) or len(rephrased.strip()) <= 5:
                return question

            rephrased = re.sub(
                r'^(The standalone medical question is:?\s*|Standalone question:?\s*|'
                r'Rephrased question:?\s*|The question is:?\s*)',
                '', rephrased, flags=re.IGNORECASE
            ).strip('" \t\n\'')

            if self._is_rephrasing_valid(rephrased, original_question):
                return rephrased
            return self._template_question(hadm_id, original_question)

        except Exception as e:
            ClinicalLogger.warning(f"Rephrasing failed: {e}")
            return question

    @staticmethod
    def _is_rephrasing_valid(rephrased: str, original: str) -> bool:
        """Only reject extreme cases (5x original length suggests hallucination)."""
        return (len(rephrased) / max(1, len(original))) <= 5.0

    @staticmethod
    def _template_question(hadm_id, original_question: str) -> str:
        """Safe template-based rephrase keyed on question intent."""
        if not hadm_id:
            return original_question
        q = original_question.lower()
        if any(w in q for w in ('diagnose', 'diagnosis', 'condition')):
            return f"What diagnoses are recorded for admission {hadm_id}?"
        if any(w in q for w in ('medication', 'drug', 'prescription', 'med')):
            return f"What medications were prescribed for admission {hadm_id}?"
        if any(w in q for w in ('lab', 'test', 'result')):
            return f"What lab results are available for admission {hadm_id}?"
        if any(w in q for w in ('procedure', 'surgery', 'operation')):
            return f"What procedures were performed for admission {hadm_id}?"
        if any(w in q for w in ('microbiology', 'culture', 'organism')):
            return f"What microbiology results are available for admission {hadm_id}?"
        return f"For admission {hadm_id}, {original_question}"
