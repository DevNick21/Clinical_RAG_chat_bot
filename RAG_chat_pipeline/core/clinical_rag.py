"""Main clinical RAG chatbot"""
import re
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from RAG_chat_pipeline.inference.azure_client import get_llm
from RAG_chat_pipeline.config.settings import get_settings
from RAG_chat_pipeline.core.retriever import Retriever
from RAG_chat_pipeline.helper.entity_extraction import extract_entities, extract_context_from_chat_history
from RAG_chat_pipeline.helper.invoke import safe_llm_invoke
from RAG_chat_pipeline.audit.audit_log import log_request
from RAG_chat_pipeline.audit.claim_checker import check_claims

from RAG_chat_pipeline.utils.logger import ClinicalLogger

_settings = get_settings()
ClinicalLogger.set_level(_settings.log_level)

DEFAULT_K = _settings.default_k
MAX_CHAT_HISTORY = _settings.max_chat_history
SECTION_KEYWORDS = _settings.section_keywords
ENABLE_REPHRASING = _settings.enable_rephrasing
ENABLE_ENTITY_EXTRACTION = _settings.enable_entity_extraction
RETRIEVAL_MAX_K = _settings.retrieval_max_k
GLOBAL_SEARCH_MAX_K = _settings.global_search_max_k
CANDIDATE_DOC_LIMIT = _settings.candidate_doc_limit
FINAL_DOCS_LIMIT = _settings.final_docs_limit
STREAMING_CANDIDATE_DOC_LIMIT = _settings.streaming_candidate_doc_limit
STREAMING_GLOBAL_SEARCH_MAX_K = _settings.streaming_global_search_max_k
STREAMING_FINAL_DOCS_LIMIT = _settings.streaming_final_docs_limit


class ClinicalRAGBot:
    # Simple, config-like rules for section-aware extraction
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
        "labevents": {  # alias
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

    def __init__(self, vectorstore: FAISS, clinical_emb, chunked_docs: List):
        self.vectorstore = vectorstore
        self.clinical_emb = clinical_emb
        self.chunked_docs = chunked_docs

        # Initialize LLM from Azure AI Foundry (deployment/key/endpoint via .env).
        # Reasoning effort, model, and credentials all come from .env via the
        # factory. Streaming is decided per-call in the Responses API, not at
        # client construction.
        self.llm = get_llm()

        # Retrieval layer: indices, filter, semantic search, ID preflight, and
        # the embedding cache all live behind this single object.
        self.retriever = Retriever(vectorstore, clinical_emb, chunked_docs)

        # Setup prompts
        self.condense_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Rephrase the follow-up question as a standalone medical question using ONLY the context from chat history.

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

If no relevant context exists, return the original question unchanged."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Clinical QA prompt. The strictness around "asked-about IDs not
        # in retrieved docs" was added after the cloud deploy surfaced a
        # case where the model honestly admitted the asked admissions
        # weren't present but then dumped unrelated retrieved docs as
        # "Available data" — confusing for clinicians.
        self.base_clinical_qa_prompt = """You are a clinical RAG assistant answering questions strictly from the retrieved MIMIC-IV documents below.

{context_instruction}

ANSWERING RULES (apply in order):

1. If the user asks about specific admission IDs, patient IDs, or subject
   IDs, and those IDs do NOT appear in the retrieved documents' metadata,
   reply with: "No records were found for [the IDs they asked about] in
   the available data." Then STOP. Do NOT surface unrelated admissions
   as 'available data' — the user wanted those specific records, not a
   tour of the database.

2. If the retrieved documents DO contain the asked-about IDs, answer the
   question concisely. Include:
   - Specific medical details from the docs (codes, values, dates, units)
   - Citation per claim: "(Admission <hadm_id>, <section>)"

3. Never invent values, codes, dosages, or dates. If a specific value the
   user asked about isn't in the documents, say so explicitly rather than
   substituting a similar value from a different admission.

4. End every answer with: "ℹ️ Data from MIMIC-IV database for research/education only."

Context:
{context}"""

        # Create chains
        self.question_answer_chain = None

    def _create_clinical_prompt(self, hadm_id=None, subject_id=None):
        """Create dynamic clinical prompt with admission/patient context.

        `hadm_id` and `subject_id` may be int OR list[int] (multi-ID
        comparison query). The list is rendered into the instruction
        text so the model knows the full scope of the filter.
        """
        hadm_ids = Retriever.to_id_list(hadm_id)
        subject_ids = Retriever.to_id_list(subject_id)

        if hadm_ids:
            ids_str = ", ".join(str(i) for i in hadm_ids)
            label = "admission IDs" if len(hadm_ids) > 1 else "admission ID"
            context_instruction = f"""You are analyzing documents specifically for {label} {ids_str}. The provided documents are filtered for {"these admissions" if len(hadm_ids) > 1 else "this admission"}, so they ARE relevant to the query.

IMPORTANT: ALWAYS include source citations from each document, even if they don't explicitly repeat the admission ID. Never state "Source Citations: None provided" unless no documents were found."""
        elif subject_ids:
            ids_str = ", ".join(str(i) for i in subject_ids)
            label = "patient/subject IDs" if len(subject_ids) > 1 else "patient/subject ID"
            context_instruction = f"""You are analyzing documents specifically for {label} {ids_str}. The provided documents are filtered for {"these patients" if len(subject_ids) > 1 else "this patient"}, so they ARE relevant to the query.

IMPORTANT: ALWAYS include source citations from each document, even if they don't explicitly repeat the patient ID. Never state "Source Citations: None provided" unless no documents were found."""
        else:
            context_instruction = "You are analyzing medical documents from the MIMIC-IV database. ALWAYS include source citations for any information provided."

        prompt_text = self.base_clinical_qa_prompt.format(
            context_instruction=context_instruction,
            context="{context}"
        )

        return ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{input}")
        ])

    # Standard "no records found" disclaimer string. Used both in the
    # filter-miss path (we asked the index, got nothing) and the
    # preflight rejection path (post-retrieval ID mismatch). Centralised
    # so all four sites pluralise + format consistently.
    _DISCLAIMER = "ℹ️ Data from MIMIC-IV database for research/education only."

    def _no_records_text(self, entity_type, entity_id, section=None):
        """Filter-miss message: index returned no docs for the requested IDs.

        `entity_type` is "admission" / "patient/subject" (singular form);
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

    def _preflight_rejection_text(self, missing_ids):
        """Post-retrieval rejection: docs came back but for different IDs.

        Different from the filter-miss path because here the user asked
        about specific IDs and global / mixed retrieval gave back unrelated
        admissions; we want to be explicit about why we're not answering.
        """
        ids_str = ", ".join(str(i) for i in sorted(missing_ids))
        return (
            f"No records were found for admission/subject ID(s): {ids_str}. "
            "The retrieved documents are about different admissions, so I "
            "can't answer this question reliably. Please verify the ID(s) "
            f"or rephrase the question.\n\n{self._DISCLAIMER}"
        )

    @staticmethod
    def _safe_audit_log(**kwargs):
        """Best-effort audit log write. Returns audit_id or None on failure.

        Audit must never block answering a clinical query — if the disk
        fill, blob is unreachable, etc., we log a warning and return None
        so the caller can continue.
        """
        try:
            return log_request(**kwargs)
        except Exception as exc:
            ClinicalLogger.warning("Audit log write failed", error=str(exc))
            return None

    def _try_preflight_short_circuit(self, question_text, retrieved_docs,
                                     hadm_id, subject_id, section, k,
                                     elapsed_seconds, *, streaming=False):
        """Run the preflight ID-mismatch check.

        Returns (answer, audit_id) when all asked-about IDs are absent
        from the retrieved docs (audit log already written). Caller
        builds its endpoint-specific response shape and returns/yields.

        Returns None when either no IDs were mentioned in the question
        or at least one IS present in retrieved_docs — caller continues
        with the normal LLM flow.

        Used by both clinical_search() (return-style) and chat_stream()
        (yield-style); the endpoint-specific differences are confined
        to those two callers' wrapping code.
        """
        missing_ids = self.retriever.check_id_mismatch(question_text, retrieved_docs)
        if not missing_ids:
            return None

        ids_str = ", ".join(str(i) for i in sorted(missing_ids))
        ClinicalLogger.info(
            f"Pre-flight{' (stream)' if streaming else ''}: asked IDs {ids_str} "
            f"not in retrieved docs; short-circuiting before LLM call.")

        answer = self._preflight_rejection_text(missing_ids)
        metadata = {
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            "section": section,
            "k": k,
            "preflight": "id_mismatch",
            "asked_ids": sorted(missing_ids),
        }
        if streaming:
            metadata["streaming"] = True
        else:
            metadata["search_time"] = elapsed_seconds

        audit_id = self._safe_audit_log(
            question=question_text,
            retrieved_docs=retrieved_docs,
            reasoning_summaries=[],
            answer=answer,
            unsupported_claims=[],
            response_id=None,
            metadata=metadata,
        )
        return answer, audit_id

    def _no_documents_result(self, entity_type, entity_id, section, start_time):
        """Filter-miss result dict; message text delegated to _no_records_text."""
        return {
            "answer": self._no_records_text(entity_type, entity_id, section),
            "source_documents": [],
            "citations": [],
            "search_time": time.time() - start_time,
            "documents_found": 0
        }

    def _validate_and_fix_response(self, answer, retrieved_docs, hadm_id=None):
        """Post-process response to fix common citation and format issues"""
        # Fix missing disclaimer
        if "Data from MIMIC-IV database for research/education only" not in answer:
            answer += "\n\n Data from MIMIC-IV database for research/education only."

        # Fix incorrect citation claims when documents were found
        if "Source Citations: None provided" in answer and len(retrieved_docs) > 0:
            if hadm_id:
                fix_text = f"Source Citations: From {len(retrieved_docs)} documents for admission {hadm_id}"
            else:
                fix_text = f"Source Citations: From {len(retrieved_docs)} retrieved documents"
            answer = answer.replace(
                "Source Citations: None provided", fix_text)
        elif "**Source Citations**: None provided" in answer and len(retrieved_docs) > 0:
            if hadm_id:
                fix_text = f"**Source Citations**: From {len(retrieved_docs)} documents for admission {hadm_id}"
            else:
                fix_text = f"**Source Citations**: From {len(retrieved_docs)} retrieved documents"
            answer = answer.replace(
                "**Source Citations**: None provided", fix_text)

        return answer

    def _extract_structured_content(self, section: str, content: str, hadm_id):
        """Unified, rules-based content extraction for any section"""
        section_key = (section or "").lower() or "default"
        rules = self.SECTION_RULES.get(
            section_key, self.SECTION_RULES["default"])

        lines = [ln.strip() for ln in content.split('\n') if ln.strip()]
        scored = []
        for ln in lines:
            ln_lower = ln.lower()
            # Highest priority: keyword hit
            if any(kw in ln_lower for kw in rules["keywords"]):
                scored.append((2, ln))
            # Next: regex pattern hit
            elif any(re.search(rx, ln) for rx in rules["regexes"]):
                scored.append((1, ln))
            # Fallback: take a few substantial lines
            elif len(ln) > 10 and len(scored) < 3:
                scored.append((0, ln))

        # Sort by priority and keep top-N
        scored.sort(key=lambda x: (-x[0]))
        max_lines = rules.get("max_lines", 10)
        selected = [ln for _, ln in scored[:max_lines]]

        title = rules["title"].format(
            hadm_id=hadm_id, section=(section or "UNKNOWN").upper())
        if selected:
            return f"{title}\n" + "\n".join(selected)
        else:
            return f"{title}\n" + content[:600]

    def _extract_clinical_content(self, docs, query_type="general"):
        """Extract and structure relevant clinical content from documents using unified rules"""
        extracted_content = []
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata
            section = metadata.get('section', '')
            hadm_id = metadata.get('hadm_id', 'Unknown')
            structured_content = self._extract_structured_content(
                section, content, hadm_id)
            extracted_content.append(structured_content)
        return "\n\n".join(extracted_content)

    def clinical_search(self, question, hadm_id=None, subject_id=None, section=None, k=DEFAULT_K, chat_history=None, original_question=None):
        """Clinical search function"""
        start_time = time.time()

        if original_question and original_question != question:
            ClinicalLogger.debug("Original query differs from processed query")
        ClinicalLogger.info(
            "Query received",
            hadm_id=hadm_id,
            subject_id=subject_id,
            section=section,
        )

        if chat_history is None:
            chat_history = []

        try:
            # Performance optimization: limit k to reasonable values
            k = min(k, RETRIEVAL_MAX_K)  # Configurable cap

            # Get filtered documents with performance limits
            candidate_docs = self.retriever.filter_documents(
                hadm_id, subject_id, section, limit=CANDIDATE_DOC_LIMIT)  # Configurable candidate pool

            if candidate_docs is not None:
                # Filtered search
                if not candidate_docs:
                    # No documents found
                    entity_type = "admission" if hadm_id else "patient/subject"
                    entity_id = hadm_id or subject_id
                    return self._no_documents_result(entity_type, entity_id, section, start_time)

                ClinicalLogger.info(
                    f"Filtering documents by {'admission' if hadm_id else 'subject'} ID...")
                ClinicalLogger.debug(
                    f"Filtered to {len(candidate_docs)} documents")

                # Early termination if we have very few documents
                if len(candidate_docs) <= k:
                    retrieved_docs = candidate_docs
                    ClinicalLogger.debug(
                        f"Using all {len(retrieved_docs)} available documents")
                else:
                    retrieved_docs = self.retriever.semantic_search(
                        candidate_docs, question, k)
            else:
                # Global semantic search with tighter limits
                ClinicalLogger.info(
                    "Performing optimized semantic search across all records...")
                # Configurable global cap
                k_global = min(k, GLOBAL_SEARCH_MAX_K)
                retrieved_docs = self.retriever.global_semantic_search(
                    question, k_global, section=section)

            # Performance check: limit final docs to configured value
            if len(retrieved_docs) > FINAL_DOCS_LIMIT:
                ClinicalLogger.debug(
                    f"Processing {len(retrieved_docs)} documents - reducing to top {FINAL_DOCS_LIMIT}")
                retrieved_docs = retrieved_docs[:FINAL_DOCS_LIMIT]

            # Pre-flight: if the user named admission/subject IDs and NONE of
            # the retrieved docs match those IDs, short-circuit with a clean
            # rejection instead of asking the LLM to interpret mismatched
            # context.
            preflight = self._try_preflight_short_circuit(
                original_question or question, retrieved_docs,
                hadm_id, subject_id, section, k,
                time.time() - start_time)
            if preflight is not None:
                preflight_answer, audit_id = preflight
                return {
                    "answer": preflight_answer,
                    "source_documents": [],
                    "citations": [],
                    "search_time": time.time() - start_time,
                    "documents_found": 0,
                    "search_method": "preflight_id_mismatch",
                    "performance_optimized": True,
                    "audit_id": audit_id,
                    "unsupported_claims": [],
                    "reasoning_summaries": [],
                }

            # INTELLIGENT CONTENT EXTRACTION: Extract structured clinical data
            original_citations = [(doc.metadata.get('hadm_id'), doc.metadata.get(
                'section')) for doc in retrieved_docs]
            original_doc_count = len(retrieved_docs)
            # Keep a reference to the unstructured docs for audit/claim-check;
            # `retrieved_docs` is about to be replaced with a single merged doc.
            audit_source_docs = list(retrieved_docs)
            if len(retrieved_docs) > 0:
                ClinicalLogger.debug(
                    f"Extracting clinical content from {len(retrieved_docs)} documents...")
                extracted_content = self._extract_clinical_content(
                    retrieved_docs)

                # Create a single document with structured content
                structured_doc = Document(
                    page_content=extracted_content,
                    metadata={"combined": True,
                              "doc_count": len(retrieved_docs)}
                )
                retrieved_docs = [structured_doc]

            # Generate answer with simplified approach
            ClinicalLogger.info("Generating clinical response...")
            try:
                # Create dynamic prompt with admission/patient context
                clinical_prompt = self._create_clinical_prompt(
                    hadm_id, subject_id)
                dynamic_qa_chain = create_stuff_documents_chain(
                    self.llm, clinical_prompt)

                answer = safe_llm_invoke(
                    dynamic_qa_chain,
                    {
                        "input": question,
                        "context": retrieved_docs,
                        "chat_history": chat_history
                    },
                    fallback_message="Unable to generate clinical response due to system error.",
                    context="Clinical QA"
                )
            except Exception as llm_error:
                ClinicalLogger.warning(
                    f"LLM response generation failed: {llm_error}")
                answer = f"I found relevant medical records but encountered an error generating the response. Please try rephrasing your question. Error: {str(llm_error)}"

            # Post-process response to fix common issues
            answer = self._validate_and_fix_response(
                answer, retrieved_docs, hadm_id)

            # Audit layer: capture reasoning summaries and flag any unsupported
            # specific claims (numbers, dosages, codes) against the retrieved docs.
            # Both steps are best-effort — never block the answer on audit failure.
            reasoning_summaries = getattr(
                self.llm, "last_reasoning_summaries", []) or []
            response_id = getattr(self.llm, "last_response_id", None)
            try:
                unsupported_claims = check_claims(answer, audit_source_docs)
            except Exception as claim_err:
                ClinicalLogger.warning(f"Claim check failed: {claim_err}")
                unsupported_claims = []
            search_time = time.time() - start_time
            audit_id = self._safe_audit_log(
                question=question,
                retrieved_docs=audit_source_docs,
                reasoning_summaries=reasoning_summaries,
                answer=answer,
                unsupported_claims=unsupported_claims,
                response_id=response_id,
                metadata={
                    "hadm_id": hadm_id,
                    "subject_id": subject_id,
                    "section": section,
                    "k": k,
                    "search_time": search_time,
                },
            )
            if unsupported_claims:
                ClinicalLogger.warning(
                    f"Flagged {len(unsupported_claims)} unsupported claim(s) for review; audit_id={audit_id}")

            # Prepare result with comprehensive metadata (preserve true counts/citations)
            result = {
                "answer": answer,
                "source_documents": retrieved_docs,
                "citations": [{"hadm_id": hadm, "section": section} for hadm, section in original_citations],
                "search_time": search_time,
                "documents_found": original_doc_count,
                "search_method": "filtered" if hadm_id is not None or subject_id is not None else "global_semantic",
                "performance_optimized": True,
                "audit_id": audit_id,
                "unsupported_claims": unsupported_claims,
                "reasoning_summaries": reasoning_summaries,
            }

            ClinicalLogger.info(f"Search completed in {search_time:.3f}s")
            return result

        except Exception as e:
            ClinicalLogger.error(f"Critical error in clinical_search: {e}")
            return self._handle_search_fallback(question, hadm_id, section, k, f"Critical search error: {str(e)}")

    def _extract_and_validate_params(self, question, hadm_id=None, subject_id=None, section=None, k=DEFAULT_K):
        """Centralized parameter extraction and validation"""
        # Validate inputs
        question = self._validate_question(question)
        hadm_id, subject_id, section, k = self._validate_parameters(
            hadm_id, subject_id, section, k)

        # Extract entities if needed. When the user mentions multiple
        # admission/subject IDs (e.g., "compare admissions A and B"),
        # surface the FULL list as the hadm_id / subject_id parameter so
        # the downstream filter retrieves docs for any of them. The filter
        # accepts both int and list[int]; downstream prompt + output code
        # paths normalise the list when they need a scalar.
        extracted_entities = None
        if ENABLE_ENTITY_EXTRACTION and hadm_id is None and subject_id is None and section is None:
            try:
                extracted_entities = extract_entities(question, llm=self.llm)
                if extracted_entities["confidence"] in ["high", "medium"]:
                    extracted_hadm_ids = extracted_entities.get("hadm_ids") or []
                    extracted_subject_ids = extracted_entities.get("subject_ids") or []
                    if len(extracted_hadm_ids) > 1:
                        hadm_id = extracted_hadm_ids
                    elif extracted_hadm_ids:
                        hadm_id = extracted_hadm_ids[0]
                    elif extracted_entities.get("hadm_id") is not None:
                        hadm_id = extracted_entities["hadm_id"]
                    if len(extracted_subject_ids) > 1:
                        subject_id = extracted_subject_ids
                    elif extracted_subject_ids:
                        subject_id = extracted_subject_ids[0]
                    elif extracted_entities.get("subject_id") is not None:
                        subject_id = extracted_entities["subject_id"]
                    section = extracted_entities.get("section") or section
                    ClinicalLogger.info(
                        f"Auto-extracted - hadm_id: {hadm_id}, subject_id: {subject_id}, section: {section}")
            except Exception as e:
                ClinicalLogger.warning(f"Entity extraction failed: {e}")

        return question, hadm_id, subject_id, section, k, extracted_entities

    def _process_chat_context(self, chat_history, question, hadm_id=None, subject_id=None, section=None):
        """Centralized chat history processing with minimal debugging"""
        if not chat_history:
            ClinicalLogger.debug("No chat history to process")
            return question, hadm_id, subject_id, section, {}

        try:
            # Extract context from chat history
            chat_context = extract_context_from_chat_history(
                chat_history, question)

            # Use chat context if parameters not explicitly provided
            old_hadm_id, old_subject_id, old_section = hadm_id, subject_id, section
            hadm_id = hadm_id or chat_context.get("hadm_id")
            subject_id = subject_id or chat_context.get("subject_id")
            section = section or chat_context.get("section")

            # Log parameter updates from chat context
            if hadm_id != old_hadm_id or subject_id != old_subject_id or section != old_section:
                ClinicalLogger.info(
                    f"Updated parameters from chat history - hadm_id: {hadm_id}, section: {section}")

            # Store original question for validation
            original_question = question

            # Check if rephrasing is needed
            needs_rephrasing = ENABLE_REPHRASING and self._should_rephrase_question(
                question, chat_history, hadm_id, section)

            if needs_rephrasing:
                ClinicalLogger.info(
                    "Rephrasing question using chat history...")
                question = self._rephrase_question_safely(
                    question, chat_history, hadm_id, original_question)

            return question, hadm_id, subject_id, section, chat_context
        except Exception as e:
            ClinicalLogger.error(f"Chat context processing failed: {e}")
            return question, hadm_id, subject_id, section, {}

    def _should_rephrase_question(self, question, chat_history, hadm_id, section):
        """Determine if question needs rephrasing"""
        has_admission_context = "admission" in question.lower() and (
            hadm_id is not None and str(hadm_id) in question
        )
        has_section_context = section and any(
            kw in question.lower() for kw in SECTION_KEYWORDS.get(section, [])
        )
        is_likely_followup = len(
            question.split()) < 8 and not has_admission_context

        return (
            len(chat_history) > 0 and
            is_likely_followup and
            not (has_admission_context or has_section_context) and
            len(question.split()) < 6
        )

    def _rephrase_question_safely(self, question, chat_history, hadm_id, original_question):
        """Safely rephrase question with validation"""
        try:
            rephrased = safe_llm_invoke(
                self.llm,
                self.condense_q_prompt.format_messages(
                    chat_history=chat_history, input=question),
                fallback_message=question,
                context="Question rephrasing"
            )

            if not isinstance(rephrased, str) or len(rephrased.strip()) <= 5:
                return question

            # Clean up the rephrased question
            rephrased = re.sub(
                r'^(The standalone medical question is:?\s*|Standalone question:?\s*|Rephrased question:?\s*|The question is:?\s*)',
                '', rephrased, flags=re.IGNORECASE
            ).strip('" \t\n\'')

            # Validate rephrasing quality
            if self._is_rephrasing_valid(rephrased, original_question, chat_history):
                return rephrased
            else:
                return self._create_template_question(hadm_id, original_question)

        except Exception as e:
            ClinicalLogger.warning(f"Rephrasing failed: {e}")
            return question

    def _is_rephrasing_valid(self, rephrased, original, chat_history):
        """Simplified rephrasing validation - only check extreme cases"""
        # Only reject extremely long rephrasings (likely hallucinated)
        length_ratio = len(rephrased) / max(1, len(original))
        if length_ratio > 5.0:  # Increased threshold
            return False

        # Accept all other rephrasings - removed medical term filtering
        # as it was blocking legitimate medical queries
        return True

    def _create_template_question(self, hadm_id, original_question):
        """Create safe template-based question"""
        if not hadm_id:
            return original_question

        # Create contextual rephrasing based on question type
        question_lower = original_question.lower()
        if any(word in question_lower for word in ['diagnose', 'diagnosis', 'condition']):
            return f"What diagnoses are recorded for admission {hadm_id}?"
        elif any(word in question_lower for word in ['medication', 'drug', 'prescription', 'med']):
            return f"What medications were prescribed for admission {hadm_id}?"
        elif any(word in question_lower for word in ['lab', 'test', 'result']):
            return f"What lab results are available for admission {hadm_id}?"
        elif any(word in question_lower for word in ['procedure', 'surgery', 'operation']):
            return f"What procedures were performed for admission {hadm_id}?"
        elif any(word in question_lower for word in ['microbiology', 'culture', 'organism']):
            return f"What microbiology results are available for admission {hadm_id}?"
        else:
            return f"For admission {hadm_id}, {original_question}"

    def _validate_question(self, question):
        """Validate and sanitize user input"""
        if not isinstance(question, str):
            raise ValueError("Question must be a string")

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        if len(question) < 3:
            raise ValueError("Question is too short (minimum 3 characters)")

        if len(question) > 2000:
            raise ValueError("Question is too long (maximum 2000 characters)")

        # Remove control characters but preserve medical symbols
        sanitized = ''.join(char for char in question if ord(
            char) >= 32 or char in '\n\t')

        return sanitized

    def _validate_parameters(self, hadm_id=None, subject_id=None, section=None, k=None):
        """Validate search parameters"""
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

    def _handle_search_fallback(self, question, hadm_id=None, section=None, k=DEFAULT_K, error_msg=""):
        """Graceful fallback for failed searches"""
        ClinicalLogger.warning(f"Search fallback triggered: {error_msg}")

        try:
            # Try basic semantic search as fallback
            retrieved_docs = self.retriever.global_semantic_search(
                question, k, section=section)

            if retrieved_docs:
                # Create dynamic prompt for fallback too
                clinical_prompt = self._create_clinical_prompt(hadm_id, None)
                fallback_qa_chain = create_stuff_documents_chain(
                    self.llm, clinical_prompt)

                answer = safe_llm_invoke(
                    fallback_qa_chain,
                    {"input": question, "context": retrieved_docs, "chat_history": []},
                    fallback_message="I found some relevant information, but cannot provide a detailed analysis due to technical limitations.",
                    context="Fallback search"
                )

                return {
                    "answer": answer,
                    "source_documents": retrieved_docs,
                    "citations": [{"hadm_id": doc.metadata.get('hadm_id'), "section": doc.metadata.get('section')} for doc in retrieved_docs],
                    "fallback_used": True,
                    "fallback_reason": error_msg
                }
        except Exception as fallback_error:
            ClinicalLogger.error(
                f"Fallback search also failed: {fallback_error}")

        # Ultimate fallback
        return {
            "answer": f"I apologize, but I encountered technical difficulties processing your question. {error_msg}",
            "source_documents": [],
            "citations": [],
            "error": True,
            "fallback_used": True,
            "fallback_reason": error_msg
        }

    def ask_question(self, question, chat_history=None, hadm_id=None, subject_id=None, section=None, k=DEFAULT_K):
        """Unified question method - handles both single and conversational queries with performance optimization"""
        is_conversational = chat_history is not None

        is_from_chat = chat_history is not None

        # Log differently based on context
        if is_from_chat:
            source = "API/CLI" if is_conversational else "Direct Query"
            ClinicalLogger.info(f"=== {source} CONVERSATIONAL MODE ===")
        else:
            ClinicalLogger.info(
                f"=== {'CONVERSATIONAL' if is_conversational else 'SINGLE QUESTION'} MODE ===")

        # Performance optimization: validate k early and set reasonable limits
        k = min(k, RETRIEVAL_MAX_K)  # Configurable cap

        try:
            performance_start = time.time()
            original_hadm_id, original_subject_id, original_section = hadm_id, subject_id, section
            chat_history = chat_history or []

            # Validate and extract parameters
            question, hadm_id, subject_id, section, k, extracted_entities = self._extract_and_validate_params(
                question, hadm_id, subject_id, section, k)

            # Performance check: Skip expensive chat processing for simple questions
            processing_time = time.time() - performance_start

            # Process chat context if conversational
            original_question = question
            if is_conversational and chat_history and processing_time < 1.0:  # Skip if already slow
                search_question, hadm_id, subject_id, section, chat_context = self._process_chat_context(
                    chat_history, question, hadm_id, subject_id, section)
            else:
                search_question = question
                chat_context = {}
                if processing_time >= 1.0:
                    ClinicalLogger.info(
                        "Skipping chat processing due to performance constraints")

            # Perform search
            result = self.clinical_search(
                search_question, hadm_id, subject_id, section, k, chat_history, original_question)

            # Update chat history if conversational
            if is_conversational:
                chat_history.extend(
                    [("human", question), ("assistant", result["answer"])]
                )
                if len(chat_history) > MAX_CHAT_HISTORY:
                    chat_history = chat_history[-MAX_CHAT_HISTORY:]

            total_time = time.time() - performance_start
            result.update({
                "mode": "conversational" if is_conversational else "single_question",
                "chat_history": chat_history if is_conversational else None,
                "extracted_entities": extracted_entities,
                "chat_context": chat_context if is_conversational else {},
                "manual_override": {"hadm_id": original_hadm_id, "subject_id": original_subject_id, "section": original_section} if is_conversational else None,
                "parameters": {"hadm_id": hadm_id, "subject_id": subject_id, "section": section, "k": k} if not is_conversational else None,
                "total_processing_time": total_time,
                "performance_optimized": True
            })

            if total_time > 30:  # Warn about slow queries
                ClinicalLogger.warning(
                    f"Slow query detected: {total_time:.2f}s - consider reducing document scope")

            return result

        except Exception as e:
            return self._handle_search_fallback(question, hadm_id, section, k, str(e))

    def chat(self, message, chat_history=None):
        """Main chat interface for API - handles chat history format conversion"""
        is_api_call = False
        is_cli_call = False
        is_evaluation = False

        # Log context appropriately
        if is_api_call:
            ClinicalLogger.debug("Processing API request...")
        elif is_cli_call:
            ClinicalLogger.debug("Processing CLI request...")
        elif is_evaluation:
            ClinicalLogger.debug("Running in evaluation mode...")

        # Convert and validate chat history
        processed_chat_history = self._process_api_chat_history(chat_history)

        # Truncate if too long
        if processed_chat_history and len(processed_chat_history) > MAX_CHAT_HISTORY:
            processed_chat_history = processed_chat_history[-MAX_CHAT_HISTORY:]
            ClinicalLogger.warning(
                f"Chat history truncated to {MAX_CHAT_HISTORY} messages")

        # Basic input validation
        if not is_evaluation and (not message or not isinstance(message, str) or len(message.strip()) < 2):
            ClinicalLogger.warning("Empty or invalid message received")
            return "I couldn't understand your message. Please provide a valid question."

        # Call main processing method
        response = self.ask_question(message, processed_chat_history)
        return response.get('answer', 'No answer generated')

    def chat_stream(self, message, chat_history=None):
        """Streaming chat interface for API - yields response chunks"""
        is_api_call = False
        is_cli_call = False
        is_evaluation = False

        # Log context appropriately
        if is_api_call:
            ClinicalLogger.debug("Processing streaming API request...")
        elif is_cli_call:
            ClinicalLogger.debug("Processing streaming CLI request...")
        elif is_evaluation:
            ClinicalLogger.debug("Running in streaming evaluation mode...")

        # Convert and validate chat history
        processed_chat_history = self._process_api_chat_history(chat_history)

        # Truncate if too long
        if processed_chat_history and len(processed_chat_history) > MAX_CHAT_HISTORY:
            processed_chat_history = processed_chat_history[-MAX_CHAT_HISTORY:]
            ClinicalLogger.warning(
                f"Chat history truncated to {MAX_CHAT_HISTORY} messages")

        # Basic input validation
        if not is_evaluation and (not message or not isinstance(message, str) or len(message.strip()) < 2):
            ClinicalLogger.warning("Empty or invalid message received")
            yield {"error": "I couldn't understand your message. Please provide a valid question."}
            return

        try:
            # Process question and get context
            performance_start = time.time()
            chat_history_processed = processed_chat_history or []

            # Validate and extract parameters
            question, hadm_id, subject_id, section, k, extracted_entities = self._extract_and_validate_params(
                message, None, None, None, DEFAULT_K)

            # Process chat context if conversational
            original_question = question
            if chat_history_processed:
                search_question, hadm_id, subject_id, section, chat_context = self._process_chat_context(
                    chat_history_processed, question, hadm_id, subject_id, section)
            else:
                search_question = question
                chat_context = {}

            # Get relevant documents for context. Streaming uses the same
            # retrieval limits as non-streaming - the dissertation-era
            # STREAMING_* constants got collapsed because gpt-5-nano's
            # latency is reasoning-bound, not prompt-size-bound.
            k = min(k, RETRIEVAL_MAX_K)
            candidate_docs = self.retriever.filter_documents(
                hadm_id, subject_id, section, limit=CANDIDATE_DOC_LIMIT)

            if candidate_docs is not None:
                if not candidate_docs:
                    _entity_type = "admission" if hadm_id else "patient/subject"
                    _entity_id = hadm_id if hadm_id else subject_id
                    yield {"error": self._no_records_text(_entity_type, _entity_id)}
                    return

                retrieved_docs = self.retriever.semantic_search(
                    candidate_docs, search_question, k) if len(candidate_docs) > k else candidate_docs
            else:
                k_global = min(k, GLOBAL_SEARCH_MAX_K)
                retrieved_docs = self.retriever.global_semantic_search(
                    search_question, k_global, section=section)

            if len(retrieved_docs) > FINAL_DOCS_LIMIT:
                retrieved_docs = retrieved_docs[:FINAL_DOCS_LIMIT]

            # Pre-flight: if the user named admission/subject IDs and NONE of
            # the retrieved docs match, yield a clean SSE rejection instead
            # of feeding the LLM mismatched context.
            preflight = self._try_preflight_short_circuit(
                message, retrieved_docs,
                hadm_id, subject_id, section, k,
                time.time() - performance_start, streaming=True)
            if preflight is not None:
                preflight_answer, audit_id = preflight
                yield {"content": preflight_answer, "done": False}
                yield {
                    "done": True,
                    "metadata": {
                        "search_time": time.time() - performance_start,
                        "documents_found": 0,
                        "citations": [],
                        "audit_id": audit_id,
                        "unsupported_claims": [],
                        "reasoning_summaries": [],
                        "search_method": "preflight_id_mismatch",
                        "final_answer": preflight_answer,
                    },
                }
                return

            # Keep the unstructured docs for audit/claim-check; the merged
            # structured_doc loses per-source citations.
            audit_source_docs = list(retrieved_docs)

            # Extract clinical content
            if retrieved_docs:
                extracted_content = self._extract_clinical_content(
                    retrieved_docs)
                structured_doc = Document(
                    page_content=extracted_content,
                    metadata={"combined": True,
                              "doc_count": len(retrieved_docs)}
                )
                context_docs = [structured_doc]
            else:
                context_docs = []

            # Create dynamic prompt
            clinical_prompt = self._create_clinical_prompt(hadm_id, subject_id)
            dynamic_qa_chain = create_stuff_documents_chain(
                self.llm, clinical_prompt)

            # Stream the response
            ClinicalLogger.info("Starting streaming clinical response...")

            full_response = ""
            for chunk in dynamic_qa_chain.stream({
                "input": search_question,
                "context": context_docs,
                "chat_history": chat_history_processed
            }):
                if isinstance(chunk, str):
                    full_response += chunk
                    yield {"content": chunk, "done": False}
                elif isinstance(chunk, dict) and "answer" in chunk:
                    chunk_text = chunk["answer"]
                    full_response += chunk_text
                    yield {"content": chunk_text, "done": False}

            # Post-process and finalize
            final_answer = self._validate_and_fix_response(
                full_response, retrieved_docs, hadm_id)

            # Audit layer (same flow as clinical_search). Best-effort: any
            # failure logs a warning but never blocks the streamed answer.
            reasoning_summaries = getattr(
                self.llm, "last_reasoning_summaries", []) or []
            response_id = getattr(self.llm, "last_response_id", None)
            try:
                unsupported_claims = check_claims(
                    final_answer, audit_source_docs)
            except Exception as claim_err:
                ClinicalLogger.warning(
                    f"Claim check failed: {claim_err}")
                unsupported_claims = []
            audit_id = self._safe_audit_log(
                question=search_question,
                retrieved_docs=audit_source_docs,
                reasoning_summaries=reasoning_summaries,
                answer=final_answer,
                unsupported_claims=unsupported_claims,
                response_id=response_id,
                metadata={
                    "hadm_id": hadm_id,
                    "subject_id": subject_id,
                    "section": section,
                    "k": k,
                    "streaming": True,
                },
            )
            if unsupported_claims:
                ClinicalLogger.warning(
                    f"Flagged {len(unsupported_claims)} unsupported claim(s) for review; audit_id={audit_id}")

            # Update chat history
            if chat_history_processed:
                chat_history_processed.extend(
                    [("human", message), ("assistant", final_answer)])
                if len(chat_history_processed) > MAX_CHAT_HISTORY:
                    chat_history_processed = chat_history_processed[-MAX_CHAT_HISTORY:]

            # Send final metadata
            total_time = time.time() - performance_start
            yield {
                "done": True,
                "metadata": {
                    "search_time": total_time,
                    "documents_found": len(retrieved_docs),
                    "citations": [{"hadm_id": doc.metadata.get('hadm_id'), "section": doc.metadata.get('section')} for doc in retrieved_docs],
                    "audit_id": audit_id,
                    "unsupported_claims": unsupported_claims,
                    "reasoning_summaries": reasoning_summaries,
                    "final_answer": final_answer
                }
            }

        except Exception as e:
            ClinicalLogger.error(f"Streaming error: {e}")
            yield {"error": f"An error occurred while processing your request: {str(e)}", "done": True}

    def _process_api_chat_history(self, chat_history):
        """Convert API chat history format to internal format"""
        if not chat_history:
            return []

        processed = []
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
