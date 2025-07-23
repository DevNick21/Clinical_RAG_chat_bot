"""Main clinical RAG chatbot"""
import sys
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from typing import List
from RAG_chat_pipeline.config.config import LLM_MODEL, DEFAULT_K, MAX_CHAT_HISTORY
from RAG_chat_pipeline.helper.entity_extraction import extract_entities, extract_context_from_chat_history
from RAG_chat_pipeline.helper.invoke import safe_llm_invoke
from collections import defaultdict
import time


class ClinicalRAGBot:
    def __init__(self, vectorstore: FAISS, clinical_emb, chunked_docs: List):
        self.vectorstore = vectorstore
        self.clinical_emb = clinical_emb
        self.chunked_docs = chunked_docs

        # Initialize LLM
        self.llm = Ollama(model=LLM_MODEL)

        # Create metadata indices for faster lookups
        self._build_metadata_indices()

        # Setup prompts
        self.condense_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the chat history and follow-up question, rephrase the follow-up question as a standalone medical question.

CRITICAL: Preserve all medical identifiers and context:
- Keep admission IDs (hadm_id) explicit: "admission 12345"
- Maintain section references: diagnoses, procedures, labs, microbiology, prescriptions
- Preserve temporal context: dates, times, sequences
- Keep medical terminology precise

Example:
History: "What diagnoses does admission 25282710 have?"
Follow-up: "How serious are they?"
Standalone: "How serious are the diagnoses for admission 25282710?"

The standalone question must be answerable without chat history while preserving all medical context."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.clinical_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical AI assistant analyzing structured medical records from the MIMIC database.

DOCUMENT STRUCTURE - Each document contains one of these sections:
• header: admission info (admit/discharge times, type, expire flag)
• diagnoses: ICD diagnosis codes with descriptions
• procedures: ICD procedure codes with descriptions
• labs: laboratory tests (values, times, categories, flags)
• microbiology: culture tests (specimen types, dates, comments)
• prescriptions: medications (dosages, administration times, order status)

PATIENT CONTEXT:
- Each document has hadm_id (admission ID) and subject_id (patient ID)
- Multiple admissions may exist for the same patient (subject_id)
- When relevant, reference patient history across admissions
- Always cite specific admission IDs when providing information

RESPONSE GUIDELINES:
1. **ACCURACY IS CRITICAL**: ONLY provide information that is explicitly in the provided documents
2. **Direct answers**: Start with the specific answer to the question
3. **Cite sources**: Reference admission IDs and sections when available
4. **Handle missing data**: If no relevant data found, state CLEARLY: "No records found for [patient/admission] [ID]" - NEVER make up information
5. **Patient context**: If multiple admissions exist for same patient, mention when relevant
6. **Cross-admission insights**: When appropriate, note patterns across patient's admissions
7. **Use section structure**: Reference the specific section types above
8. **Include details**: For labs/meds, include values, times, and flags when relevant
9. **ICD codes**: When discussing diagnoses/procedures, include ICD codes if available
10. **NEVER HALLUCINATE**: If a patient ID or admission ID is not found in the documents, clearly state this fact

RESPONSE FORMAT:
- Lead with direct answer
- Include relevant details from the documents
- Cite admission ID and section
- Note patient context when relevant (e.g., "This patient has 3 admissions in the database")
- End with disclaimer

IMPORTANT: Always end your response with:
"⚠️ MEDICAL DISCLAIMER: This information is for educational purposes only and should not be used for medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice."

Context: {context}"""),
            ("human", "{input}")
        ])

        # Create chains
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, self.clinical_qa_prompt)

    def _build_metadata_indices(self):
        """Building indices for faster metadata-based filtering"""
        self.hadm_id_index = defaultdict(list)
        self.subject_id_index = defaultdict(list)
        self.section_index = defaultdict(list)
        self.hadm_section_index = defaultdict(list)

        for i, doc in enumerate(self.chunked_docs):
            hadm_id = doc.metadata.get('hadm_id')
            subject_id = doc.metadata.get('subject_id')
            section = doc.metadata.get('section')

            # Robust type handling for hadm_id
            if hadm_id is not None:
                try:
                    hadm_id_int = int(hadm_id)
                    self.hadm_id_index[hadm_id_int].append(i)

                    # Handle subject_id
                    if subject_id is not None:
                        try:
                            subject_id_int = int(subject_id)
                            self.subject_id_index[subject_id_int].append(i)
                        except (ValueError, TypeError):
                            print(
                                f"⚠️ Invalid subject_id in document {i}: {subject_id}")

                    if section:
                        section_str = str(section).lower()
                        self.section_index[section_str].append(i)
                        self.hadm_section_index[(
                            hadm_id_int, section_str)].append(i)

                except (ValueError, TypeError):
                    print(f"⚠️ Invalid hadm_id in document {i}: {hadm_id}")
                    continue

            elif section:
                section_str = str(section).lower()
                self.section_index[section_str].append(i)

    def _filter_candidate_documents(self, hadm_id=None, subject_id=None, section=None):
        """Centralized document filtering logic"""
        if hadm_id is not None:
            candidate_indices = self.hadm_id_index.get(hadm_id, [])
            if section is not None:
                key = (hadm_id, section)
                candidate_indices = self.hadm_section_index.get(key, [])
            return [self.chunked_docs[i] for i in candidate_indices]

        elif subject_id is not None:
            candidate_indices = self.subject_id_index.get(subject_id, [])
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
        """Perform semantic search on filtered documents"""
        if len(candidate_docs) <= k:
            return candidate_docs
        else:
            try:
                faiss_temp = FAISS.from_documents(
                    candidate_docs, self.clinical_emb)
                return faiss_temp.similarity_search(question, k=k)
            except Exception as faiss_error:
                print(
                    f"⚠️ FAISS search failed: {faiss_error}, using top {k} documents")
                return candidate_docs[:k]

    def clinical_search(self, question, hadm_id=None, subject_id=None, section=None, k=DEFAULT_K, chat_history=None):
        """Simplified clinical search function"""
        start_time = time.time()

        print(
            f"Query: '{question}' | hadm_id: {hadm_id} | subject_id: {subject_id} | section: {section}")

        if chat_history is None:
            chat_history = []

        try:
            # Get filtered documents
            candidate_docs = self._filter_candidate_documents(
                hadm_id, subject_id, section)

            if candidate_docs is not None:
                # Filtered search
                if not candidate_docs:
                    # No documents found
                    entity_type = "admission" if hadm_id else "patient/subject"
                    entity_id = hadm_id or subject_id
                    return self._no_documents_result(entity_type, entity_id, section, start_time)

                print(
                    f"Filtering documents by {'admission' if hadm_id else 'subject'} ID...")
                print(f"Filtered to {len(candidate_docs)} documents")
                retrieved_docs = self._semantic_search_on_docs(
                    candidate_docs, question, k)
            else:
                # Global semantic search
                print("Performing semantic search across all records...")
                retrieved_docs = self.vectorstore.similarity_search(
                    question, k=k)
                print(f"Retrieved {len(retrieved_docs)} documents")

                # Post-filter by section if specified
                if section is not None:
                    original_count = len(retrieved_docs)
                    retrieved_docs = [doc for doc in retrieved_docs
                                      if doc.metadata.get('section', '').lower() == section.lower()]
                    print(
                        f"Section filtering: {original_count} → {len(retrieved_docs)} documents")

            # Generate answer with error handling
            print("Generating clinical response...")
            try:
                answer = safe_llm_invoke(
                    self.question_answer_chain,
                    {
                        "input": question,
                        "context": retrieved_docs,
                        "chat_history": chat_history
                    },
                    fallback_message="Unable to generate clinical response due to system error (Ollama).",
                    context="Clinical QA"
                )
            except Exception as llm_error:
                print(f"⚠️ LLM response generation failed: {llm_error}")
                answer = f"I found relevant medical records but encountered an error generating the response. Please try rephrasing your question. Error: {str(llm_error)}"

            # Prepare result with comprehensive metadata
            search_time = time.time() - start_time
            result = {
                "answer": answer,
                "source_documents": retrieved_docs,
                "citations": [{"hadm_id": doc.metadata.get('hadm_id'), "section": doc.metadata.get('section')} for doc in retrieved_docs],
                "search_time": search_time,
                "documents_found": len(retrieved_docs),
                "search_method": "filtered" if hadm_id is not None or subject_id is not None else "global_semantic"
            }

            print(f"Search completed in {search_time:.3f}s")
            return result

        except Exception as e:
            print(f"❌ Critical error in clinical_search: {e}")
            return self._handle_search_fallback(question, hadm_id, section, k, f"Critical search error: {str(e)}")

    def _extract_and_validate_params(self, question, hadm_id=None, subject_id=None, section=None, k=DEFAULT_K):
        """Centralized parameter extraction and validation"""
        # Validate inputs
        question = self._validate_question(question)
        hadm_id, subject_id, section, k = self._validate_parameters(
            hadm_id, subject_id, section, k)

        # Extract entities if needed
        extracted_entities = None
        if hadm_id is None and subject_id is None and section is None:
            try:
                extracted_entities = extract_entities(
                    question, use_llm_fallback=True, llm=self.llm)
                if extracted_entities["confidence"] in ["high", "medium"]:
                    hadm_id = extracted_entities.get("hadm_id") or hadm_id
                    subject_id = extracted_entities.get(
                        "subject_id") or subject_id
                    section = extracted_entities.get("section") or section
                    print(
                        f"Auto-extracted - hadm_id: {hadm_id}, subject_id: {subject_id}, section: {section}")
            except Exception as e:
                print(f"⚠️ Entity extraction failed: {e}")

        return question, hadm_id, subject_id, section, k, extracted_entities

    def _process_chat_context(self, chat_history, question, hadm_id=None, subject_id=None, section=None):
        """Centralized chat history processing"""
        if not chat_history:
            return question, hadm_id, subject_id, section, {}

        try:
            # Extract context from chat history
            chat_context = extract_context_from_chat_history(
                chat_history, question)

            # Use chat context if parameters not explicitly provided
            hadm_id = hadm_id or chat_context.get("hadm_id")
            subject_id = subject_id or chat_context.get("subject_id")
            section = section or chat_context.get("section")

            # Rephrase question using chat history
            if len(chat_history) > 0:
                rephrased = safe_llm_invoke(
                    self.llm,
                    self.condense_q_prompt.format_messages(
                        chat_history=chat_history, input=question),
                    fallback_message=question,
                    context="Question rephrasing"
                )
                question = rephrased if isinstance(rephrased, str) and len(
                    rephrased.strip()) > 5 else question

            return question, hadm_id, subject_id, section, chat_context
        except Exception as e:
            print(f"⚠️ Chat context processing failed: {e}")
            return question, hadm_id, subject_id, section, {}

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
        print(f"⚠️ Search fallback triggered: {error_msg}")

        try:
            # Try basic semantic search as fallback
            retrieved_docs = self.vectorstore.similarity_search(question, k=k)

            if section and retrieved_docs:
                retrieved_docs = [
                    doc for doc in retrieved_docs if doc.metadata.get('section') == section]

            if retrieved_docs:
                answer = safe_llm_invoke(
                    self.question_answer_chain,
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
            print(f"❌ Fallback search also failed: {fallback_error}")

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
        """Unified question method - handles both single and conversational queries"""
        is_conversational = chat_history is not None

        # Detect call context
        caller_info = sys._getframe(1)
        caller_function = caller_info.f_code.co_name
        is_from_chat = caller_function == "chat"

        # Log differently based on context
        if is_from_chat:
            source = "API/CLI" if is_conversational else "Direct Query"
            print(f"=== {source} CONVERSATIONAL MODE ===")
        else:
            print(
                f"=== {'CONVERSATIONAL' if is_conversational else 'SINGLE QUESTION'} MODE ===")

        try:
            original_hadm_id, original_subject_id, original_section = hadm_id, subject_id, section
            chat_history = chat_history or []

            # Validate and extract parameters
            question, hadm_id, subject_id, section, k, extracted_entities = self._extract_and_validate_params(
                question, hadm_id, subject_id, section, k)

            # Process chat context if conversational
            if is_conversational and chat_history:
                search_question, hadm_id, subject_id, section, chat_context = self._process_chat_context(
                    chat_history, question, hadm_id, subject_id, section)
            else:
                search_question = question
                chat_context = {}

            # Perform search
            result = self.clinical_search(
                search_question, hadm_id, subject_id, section, k, chat_history)

            # Update chat history if conversational
            if is_conversational:
                chat_history.extend(
                    [("human", question), ("assistant", result["answer"])])
                if len(chat_history) > MAX_CHAT_HISTORY:
                    chat_history = chat_history[-MAX_CHAT_HISTORY:]

            result.update({
                "mode": "conversational" if is_conversational else "single_question",
                "chat_history": chat_history if is_conversational else None,
                "extracted_entities": extracted_entities,
                "chat_context": chat_context if is_conversational else {},
                "manual_override": {"hadm_id": original_hadm_id, "subject_id": original_subject_id, "section": original_section} if is_conversational else None,
                "parameters": {"hadm_id": hadm_id, "subject_id": subject_id, "section": section, "k": k} if not is_conversational else None
            })
            return result

        except Exception as e:
            return self._handle_search_fallback(question, hadm_id, section, k, str(e))

    def ask_single_question(self, question, hadm_id=None, subject_id=None, section=None, k=DEFAULT_K, search_strategy="auto"):
        """Backward compatibility wrapper for single questions"""
        return self.ask_question(question, None, hadm_id, subject_id, section, k)

    def ask_with_chat_history(self, question, chat_history=None, hadm_id=None, subject_id=None, section=None, k=DEFAULT_K):
        """Backward compatibility wrapper for conversational mode"""
        return self.ask_question(question, chat_history, hadm_id, subject_id, section, k)

    def chat(self, message, chat_history=None):
        """Main chat interface for API - handles chat history format conversion"""
        # Check the source of the call to ensure proper usage
        caller_info = sys._getframe(1)
        caller_filename = caller_info.f_code.co_filename
        caller_function = caller_info.f_code.co_name

        # Identify the context of the call
        is_api_call = "app.py" in caller_filename  # Called from Flask API
        is_cli_call = "main.py" in caller_filename  # Called from CLI
        is_evaluation = "rag_evaluator.py" in caller_filename or "evaluator" in caller_filename

        # Log appropriate usage information
        if is_api_call:
            print("📱 Processing API request...")
        elif is_cli_call:
            print("💻 Processing CLI request...")
        elif is_evaluation:
            print("🧪 Running in evaluation mode...")
        else:
            print(
                "⚠️ Warning: chat() method called outside of API, CLI or evaluation context")
            print(
                f"Called from: {caller_filename} in function {caller_function}")

        # Convert chat history format if needed
        processed_chat_history = self._process_api_chat_history(chat_history)

        # Truncate chat history if too long
        if processed_chat_history and len(processed_chat_history) > MAX_CHAT_HISTORY:
            processed_chat_history = processed_chat_history[-MAX_CHAT_HISTORY:]

        # For non-evaluation contexts, verify message is likely from user input
        if not is_evaluation:
            if not message or not isinstance(message, str) or len(message.strip()) < 2:
                print("⚠️ Warning: Empty or invalid message received")
                return "I couldn't understand your message. Please provide a valid question."

            # Basic check for hardcoded test messages in non-evaluation contexts
            test_patterns = ["test", "hello", "demo", "example"]
            if not is_evaluation and len(message.split()) == 1 and message.lower() in test_patterns:
                print("⚠️ Possible test message detected in production context")

        # Call the main ask_question method
        response = self.ask_question(message, processed_chat_history)

        # Return in the format expected by the API
        return response.get('answer', 'No answer generated')

    def _process_api_chat_history(self, chat_history):
        """Convert API chat history format to internal format"""
        if not chat_history:
            return []

        processed = []
        for msg in chat_history:
            if isinstance(msg, dict):
                # Handle API format: {"role": "user", "content": "message"}
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if content.strip():
                    processed.append((role, content))
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                # Handle tuple format: (role, message)
                role, content = msg[0], msg[1]
                if content.strip():
                    processed.append((role, content))
            else:
                print(f"Warning: Unexpected chat history format: {type(msg)}")

        return processed
