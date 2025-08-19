"""Main clinical RAG chatbot"""
import sys
import re
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from RAG_chat_pipeline.config.config import LLM_MODEL, DEFAULT_K, MAX_CHAT_HISTORY, SECTION_KEYWORDS, ENABLE_REPHRASING, ENABLE_ENTITY_EXTRACTION, LOG_LEVEL
from RAG_chat_pipeline.helper.entity_extraction import extract_entities, extract_context_from_chat_history
from RAG_chat_pipeline.helper.invoke import safe_llm_invoke
from collections import defaultdict
from functools import lru_cache

# Robust import for logger with fallback
try:
    from RAG_chat_pipeline.utils.logger import ClinicalLogger  # type: ignore
except Exception:  # pragma: no cover
    class ClinicalLogger:
        """Centralized logging utility with verbosity control (fallback)"""
        LEVELS = {"quiet": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}
        level = "info"

        @classmethod
        def set_level(cls, level: str):
            if level in cls.LEVELS:
                cls.level = level

        @classmethod
        def _enabled(cls, lvl: str) -> bool:
            return cls.LEVELS.get(cls.level, 3) >= cls.LEVELS.get(lvl, 3)

        @staticmethod
        def info(msg):
            if ClinicalLogger._enabled("info"):
                print(f"ℹ️ {msg}")

        @staticmethod
        def warning(msg):
            if ClinicalLogger._enabled("warning"):
                print(f"⚠️ {msg}")

        @staticmethod
        def error(msg):
            if ClinicalLogger._enabled("error"):
                print(f"❌ {msg}")

        @staticmethod
        def success(msg):
            if ClinicalLogger._enabled("info"):
                print(f"✅ {msg}")

        @staticmethod
        def debug(msg):
            if ClinicalLogger._enabled("debug"):
                print(f"🔍 {msg}")

# Initialize logger level from config
ClinicalLogger.set_level(LOG_LEVEL)

# Common medical term fragments for hallucination detection
MEDICAL_KEYWORDS = [
    'cardio', 'neuro', 'hepat', 'renal', 'pulmon', 'gastro', 'endo', 'immuno',
    'oncol', 'hemat', 'psych', 'ortho', 'derm', 'ophthalm', 'oto', 'gyneco',
    'obste', 'urolog', 'nephro', 'ather', 'ischemi', 'infarc', 'stenosis',
    'arteri', 'ventric', 'systolic', 'diastol', 'hypertensi', 'fibrill',
    'tachyc', 'bradyc', 'thromb', 'embol', 'aneurysm', 'failure', 'insuffic',
    'pneumonia', 'diabetes', 'hypertension', 'sepsis', 'myocardial', 'cerebral',
    'respiratory', 'cardiac', 'acute', 'chronic', 'syndrome', 'disease'
]


class ErrorHandler:
    """Centralized error handling utility"""
    @staticmethod
    def safe_operation(func, *args, fallback=None, error_msg="Operation failed", **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ClinicalLogger.warning(f"{error_msg}: {type(e).__name__}: {e}")
            return fallback


class _EmbCache:
    """Lightweight embedding cache"""

    def __init__(self, embedder, max_items: int = 512):
        self.embedder = embedder
        self.max_items = max_items
        self._cache = {}
        self._order = []

    def get(self, text: str):
        key = text
        if key in self._cache:
            # move to end (LRU)
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return self._cache[key]
        # compute and store
        vec = self.embedder.embed_query(text)
        self._cache[key] = vec
        self._order.append(key)
        if len(self._order) > self.max_items:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        return vec


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

        # Initialize LLM with proper local configuration
        self.llm = Ollama(
            model=LLM_MODEL,
            temperature=0.1,  # Slight randomness for better responses
            repeat_penalty=1.1  # Standard repeat penalty
        )

        # Embedding cache
        self._emb_cache = _EmbCache(self.clinical_emb, max_items=512)

        # Performance optimization: Create metadata indices for faster lookups
        ClinicalLogger.info("Initializing performance optimizations...")
        self._build_metadata_indices()

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

        # Create base clinical prompt template - SIMPLIFIED FOR SPEED
        self.base_clinical_qa_prompt = """Analyze the MIMIC-IV medical records and provide a concise, accurate response.

{context_instruction}

Based on the provided documents, answer the question directly and include:
- Specific medical details (codes, values, dates)
- Source: admission ID and section type

Use only information from the documents. End with: "ℹ️ Data from MIMIC-IV database for research/education only."

Context: {context}"""

        # Create chains - these will be updated dynamically
        self.question_answer_chain = None

    def _create_clinical_prompt(self, hadm_id=None, subject_id=None):
        """Create dynamic clinical prompt with admission/patient context"""
        if hadm_id:
            context_instruction = f"""You are analyzing documents specifically for admission ID {hadm_id}. The provided documents are filtered for this admission, so they ARE relevant to the query about this admission.

IMPORTANT: ALWAYS include source citations from each document, even if they don't explicitly repeat the admission ID. Never state "Source Citations: None provided" unless no documents were found. Each document contains relevant information for this admission."""
        elif subject_id:
            context_instruction = f"""You are analyzing documents specifically for patient/subject ID {subject_id}. The provided documents are filtered for this patient, so they ARE relevant to the query about this patient.

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

    def _build_metadata_indices(self):
        """Optimized indices building for faster metadata-based filtering"""
        ClinicalLogger.info("Building optimized metadata indices...")
        start_time = time.time()

        self.hadm_id_index = defaultdict(list)
        self.subject_id_index = defaultdict(list)
        self.section_index = defaultdict(list)
        self.hadm_section_index = defaultdict(list)

        # Process documents in batches for better memory management
        batch_size = 1000
        total_docs = len(self.chunked_docs)
        processed = 0

        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch = self.chunked_docs[batch_start:batch_end]

            for i, doc in enumerate(batch):
                doc_idx = batch_start + i
                hadm_id = doc.metadata.get('hadm_id')
                subject_id = doc.metadata.get('subject_id')
                section = doc.metadata.get('section')

                # Robust type handling for hadm_id
                if hadm_id is not None:
                    try:
                        hadm_id_int = int(hadm_id)
                        self.hadm_id_index[hadm_id_int].append(doc_idx)

                        # Handle subject_id
                        if subject_id is not None:
                            try:
                                subject_id_int = int(subject_id)
                                self.subject_id_index[subject_id_int].append(
                                    doc_idx)
                            except (ValueError, TypeError):
                                if processed < 10:  # Limit warning messages
                                    ClinicalLogger.warning(
                                        f"Invalid subject_id in document {doc_idx}: {subject_id}")

                        if section:
                            section_str = str(section).lower()
                            self.section_index[section_str].append(doc_idx)
                            self.hadm_section_index[(
                                hadm_id_int, section_str)].append(doc_idx)

                    except (ValueError, TypeError):
                        if processed < 10:  # Limit warning messages
                            ClinicalLogger.warning(
                                f"Invalid hadm_id in document {doc_idx}: {hadm_id}")
                        continue

                elif section:
                    section_str = str(section).lower()
                    self.section_index[section_str].append(doc_idx)

                processed += 1

            # Progress indicator for large datasets
            if total_docs > 5000:
                progress = (batch_end / total_docs) * 100
                ClinicalLogger.debug(
                    f"Index building progress: {progress:.1f}% ({batch_end}/{total_docs})")

        build_time = time.time() - start_time
        ClinicalLogger.info(f"Metadata indices built in {build_time:.2f}s")
        ClinicalLogger.info(
            f"Index stats: {len(self.hadm_id_index)} admissions, {len(self.subject_id_index)} subjects, {len(self.section_index)} sections")

    def _filter_candidate_documents(self, hadm_id=None, subject_id=None, section=None, limit=50):
        """Centralized document filtering logic"""
        if hadm_id is not None:
            candidate_indices = self.hadm_id_index.get(hadm_id, [])
            if section is not None:
                key = (hadm_id, section)
                candidate_indices = self.hadm_section_index.get(key, [])

            # Apply limit to prevent excessive document processing
            if len(candidate_indices) > limit:
                ClinicalLogger.info(
                    f"Limiting documents from {len(candidate_indices)} to {limit} for performance")
                candidate_indices = candidate_indices[:limit]

            return [self.chunked_docs[i] for i in candidate_indices]

        elif subject_id is not None:
            candidate_indices = self.subject_id_index.get(subject_id, [])

            # Apply limit early to reduce processing
            if len(candidate_indices) > limit:
                ClinicalLogger.info(
                    f"Limiting documents from {len(candidate_indices)} to {limit} for performance")
                candidate_indices = candidate_indices[:limit]

            candidate_docs = [self.chunked_docs[i] for i in candidate_indices]
            if section is not None:
                candidate_docs = [doc for doc in candidate_docs
                                  if doc.metadata.get('section', '').lower() == section.lower()]
            return candidate_docs

        return None  # Indicates global search needed

    def _no_documents_result(self, entity_type, entity_id, section, start_time):
        """Helper method for no documents found result"""
        section_msg = f" in section '{section}'" if section else ""
        return {
            "answer": f"No records found for {entity_type} {entity_id}{section_msg}",
            "source_documents": [],
            "citations": [],
            "search_time": time.time() - start_time,
            "documents_found": 0
        }

    def _semantic_search_on_docs(self, candidate_docs, question, k):
        """Optimized semantic search using FAISS vectorstore efficiently"""
        if len(candidate_docs) <= k:
            ClinicalLogger.debug(
                f"Returning all {len(candidate_docs)} documents (less than k={k})")
            return candidate_docs

        try:
            # For moderate-sized candidate sets, use direct similarity calculation
            if len(candidate_docs) <= 100:
                # Get question embedding once (cached)
                question_embedding = self._emb_cache.get(question)

                # Calculate similarities efficiently
                scored_docs = []
                for doc in candidate_docs:
                    # Use first 500 chars for consistency with vectorstore
                    doc_text = doc.page_content[:500]
                    doc_embedding = self._emb_cache.get(doc_text)

                    similarity = cosine_similarity(
                        [question_embedding], [doc_embedding])[0][0]
                    scored_docs.append((similarity, doc))

                # Sort and return top k
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                top_docs = [doc for _, doc in scored_docs[:k]]

                ClinicalLogger.debug(
                    f"Selected top {len(top_docs)} documents by direct similarity")
                return top_docs

            else:
                # For larger sets, create temporary FAISS index (still faster than before)
                ClinicalLogger.info(
                    f"Creating temporary index for {len(candidate_docs)} documents")
                faiss_temp = FAISS.from_documents(
                    candidate_docs, self.clinical_emb)
                top_docs = faiss_temp.similarity_search(question, k=k)

                ClinicalLogger.debug(
                    f"Selected top {len(top_docs)} documents via temporary FAISS")
                return top_docs

        except Exception as similarity_error:
            ClinicalLogger.warning(
                f"Similarity search failed: {similarity_error}, using first {k} documents")
            return candidate_docs[:k]

    def _validate_and_fix_response(self, answer, retrieved_docs, hadm_id=None):
        """Post-process response to fix common citation and format issues"""
        # Fix missing disclaimer
        if "Data from MIMIC-IV database for research/education only" not in answer:
            answer += "\n\nℹ️ Data from MIMIC-IV database for research/education only."

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
            ClinicalLogger.debug(f"Original query: '{original_question}'")
        ClinicalLogger.info(
            f"Query: '{question}' | hadm_id: {hadm_id} | subject_id: {subject_id} | section: {section}")

        if chat_history is None:
            chat_history = []

        try:
            # Performance optimization: limit k to reasonable values
            k = min(k, 5)  # Focus on most relevant documents

            # Get filtered documents with performance limits
            candidate_docs = self._filter_candidate_documents(
                hadm_id, subject_id, section, limit=20)  # Smaller candidate pool

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
                    retrieved_docs = self._semantic_search_on_docs(
                        candidate_docs, question, k)
            else:
                # Global semantic search with tighter limits
                ClinicalLogger.info(
                    "Performing optimized semantic search across all records...")
                k_global = min(k, 20)  # Even tighter limit for global search
                retrieved_docs = self.vectorstore.similarity_search(
                    question, k=k_global)
                ClinicalLogger.debug(
                    f"Retrieved {len(retrieved_docs)} documents")

                # Post-filter by section if specified
                if section is not None:
                    original_count = len(retrieved_docs)
                    retrieved_docs = [doc for doc in retrieved_docs
                                      if doc.metadata.get('section', '').lower() == section.lower()]
                    ClinicalLogger.debug(
                        f"Section filtering: {original_count} → {len(retrieved_docs)} documents")

            # Performance check: warn if too many documents
            if len(retrieved_docs) > 5:
                ClinicalLogger.warning(
                    f"Processing {len(retrieved_docs)} documents - reducing to top 3")
                retrieved_docs = retrieved_docs[:3]

            # INTELLIGENT CONTENT EXTRACTION: Extract structured clinical data
            original_citations = [(doc.metadata.get('hadm_id'), doc.metadata.get(
                'section')) for doc in retrieved_docs]
            original_doc_count = len(retrieved_docs)
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
                    fallback_message="Unable to generate clinical response due to system error (Ollama).",
                    context="Clinical QA"
                )
            except Exception as llm_error:
                ClinicalLogger.warning(
                    f"LLM response generation failed: {llm_error}")
                answer = f"I found relevant medical records but encountered an error generating the response. Please try rephrasing your question. Error: {str(llm_error)}"

            # Post-process response to fix common issues
            answer = self._validate_and_fix_response(
                answer, retrieved_docs, hadm_id)

            # Prepare result with comprehensive metadata (preserve true counts/citations)
            search_time = time.time() - start_time
            result = {
                "answer": answer,
                "source_documents": retrieved_docs,
                "citations": [{"hadm_id": hadm, "section": section} for hadm, section in original_citations],
                "search_time": search_time,
                "documents_found": original_doc_count,
                "search_method": "filtered" if hadm_id is not None or subject_id is not None else "global_semantic",
                "performance_optimized": True
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

        # Extract entities if needed
        extracted_entities = None
        if ENABLE_ENTITY_EXTRACTION and hadm_id is None and subject_id is None and section is None:
            try:
                extracted_entities = extract_entities(
                    question, use_llm_fallback=True, llm=self.llm)
                if extracted_entities["confidence"] in ["high", "medium"]:
                    hadm_id = extracted_entities.get("hadm_id") or hadm_id
                    subject_id = extracted_entities.get(
                        "subject_id") or subject_id
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

            # Check for hallucinations
            if self._has_hallucination(rephrased):
                ClinicalLogger.warning(
                    "Hallucination detected, using template-based approach")
                return self._create_template_question(hadm_id, original_question)

            # Validate rephrasing quality
            if self._is_rephrasing_valid(rephrased, original_question, chat_history):
                return rephrased
            else:
                return self._create_template_question(hadm_id, original_question)

        except Exception as e:
            ClinicalLogger.warning(f"Rephrasing failed: {e}")
            return question

    def _has_hallucination(self, text):
        """Check for common hallucination patterns"""
        patterns = [
            r'\b[A-Z]\d{4,5}\b',  # ICD-like codes
            r'\bcode\s*[A-Z]?\d+',  # "code K5080"
            r'\b(pneumonia|diabetes|hypertension|sepsis|myocardial|stroke)\b'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def _is_rephrasing_valid(self, rephrased, original, chat_history):
        """Validate rephrasing quality"""
        length_ratio = len(rephrased) / max(1, len(original))
        if length_ratio > 3.0:
            return False

        # Check for suspicious medical terms
        original_tokens = set(original.lower().split())
        rephrased_tokens = set(rephrased.lower().split())
        new_tokens = rephrased_tokens - original_tokens

        # Get chat tokens for context
        chat_tokens = set()
        for role, msg in chat_history[-4:]:
            if isinstance(msg, str):
                chat_tokens.update(msg.lower().split())

        suspicious_tokens = new_tokens - chat_tokens
        medical_terms_added = [token for token in suspicious_tokens
                               if any(med_term in token for med_term in MEDICAL_KEYWORDS) or len(token) > 8]

        return len(medical_terms_added) <= 1 and len(suspicious_tokens) <= 8

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
            retrieved_docs = self.vectorstore.similarity_search(question, k=k)

            if section and retrieved_docs:
                retrieved_docs = [
                    doc for doc in retrieved_docs if doc.metadata.get('section') == section]

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

        # Detect call context
        caller_info = sys._getframe(1)
        caller_function = caller_info.f_code.co_name
        is_from_chat = caller_function == "chat"

        # Log differently based on context
        if is_from_chat:
            source = "API/CLI" if is_conversational else "Direct Query"
            ClinicalLogger.info(f"=== {source} CONVERSATIONAL MODE ===")
        else:
            ClinicalLogger.info(
                f"=== {'CONVERSATIONAL' if is_conversational else 'SINGLE QUESTION'} MODE ===")

        # Performance optimization: validate k early and set reasonable limits
        k = min(k, 3)  # Focus on top 3 most relevant documents

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

    # Removed redundant methods - use ask_question() directly

    def chat(self, message, chat_history=None):
        """Main chat interface for API - handles chat history format conversion"""
        # Identify the context of the call
        caller_info = sys._getframe(1)
        caller_filename = caller_info.f_code.co_filename

        is_api_call = "app.py" in caller_filename
        is_cli_call = "main.py" in caller_filename
        is_evaluation = "rag_evaluator.py" in caller_filename or "evaluator" in caller_filename

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
