"""Main clinical RAG chatbot"""
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from typing import List
from config import LLM_MODEL, DEFAULT_K, MAX_CHAT_HISTORY
from entity_extraction import extract_entities, extract_context_from_chat_history, ask_for_clarification
from invoke import safe_llm_invoke


class ClinicalRAGBot:
    def __init__(self, vectorstore: FAISS, clinical_emb, chunked_docs: List):
        self.vectorstore = vectorstore
        self.clinical_emb = clinical_emb
        self.chunked_docs = chunked_docs

        # Initialize LLM
        self.llm = Ollama(model=LLM_MODEL)

        # Setup prompts
        self.condense_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and follow-up question, rephrase the follow-up question as a standalone medical question that can be understood without the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.clinical_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical AI assistant analyzing medical records. 

Based on the provided medical context, answer the question accurately and concisely.
- Focus on specific medical findings, diagnoses, lab values, and treatments
- If asking about severity, reference ICD codes, lab values, or clinical indicators
- If information is not available in the context, clearly state this
- Provide citations to specific admission IDs when possible
- Use medical terminology appropriately but explain complex terms

Context: {context}"""),
            ("human", "{input}")
        ])

        # Create chains
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, self.clinical_qa_prompt)

    def clinical_search(self, question, hadm_id=None, section=None, k=DEFAULT_K, chat_history=None):
        """Clinical search function"""
        print(f"Query: '{question}' | hadm_id: {hadm_id} | section: {section}")

        if chat_history is None:
            chat_history = []

        # Single search logic that handles all cases
        # If hadm_id is provided, filter by it; otherwise, do semantic search
        if hadm_id is not None:
            # Direct filter approach
            candidate_docs = [doc for doc in self.chunked_docs if doc.metadata.get(
                'hadm_id') == int(hadm_id)]
            # If section is specified, filter further
            if section is not None:
                candidate_docs = [
                    doc for doc in candidate_docs if doc.metadata.get('section') == section]

            # If no documents found, return early as the hadm_id may not exist
            if not candidate_docs:
                return {"answer": f"No records found for admission {hadm_id}", "source_documents": [], "citations": []}

            # If candidate_docs is small enough, return them directly
            # Otherwise, use FAISS for similarity search
            retrieved_docs = candidate_docs[:k] if len(candidate_docs) <= k else \
                FAISS.from_documents(
                    candidate_docs, self.clinical_emb).similarity_search(question, k=k)
        else:
            # Semantic search approach
            retrieved_docs = self.vectorstore.similarity_search(question, k=k)
            if section is not None:
                retrieved_docs = [
                    doc for doc in retrieved_docs if doc.metadata.get('section') == section]

        # Single LLM invocation
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

        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "citations": [{"hadm_id": doc.metadata.get('hadm_id'), "section": doc.metadata.get('section')} for doc in retrieved_docs]
        }

    def ask_with_sources_clinical_chatbot(self, question, chat_history=None, hadm_id=None, section=None, k=DEFAULT_K, auto_extract=True):
        """Main clinical RAG chatbot function with conversation history"""
        print(f"=== CLINICAL RAG CHATBOT ===")
        print(f"Question: '{question}'")

        if chat_history is None:
            chat_history = []

        original_hadm_id = hadm_id
        original_section = section

        # Extract context from chat history if available using extract_context_from_chat_history function
        chat_context = {"hadm_id": None, "section": None}
        if len(chat_history) > 0:
            chat_context = extract_context_from_chat_history(
                chat_history, question)

            # Use chat history context
            if chat_context["hadm_id"]:
                hadm_id = chat_context["hadm_id"]
                print(f"Using hadm_id from chat history: {hadm_id}")

            if chat_context["section"]:
                section = chat_context["section"]
                print(f"Using section from chat history: {section}")

        extracted_entities = None
        if auto_extract and hadm_id is None and section is None:
            extracted_entities = extract_entities(
                question, use_llm_fallback=True, llm=self.llm)

            # Use extracted entities if confidence is high or medium
            if extracted_entities["confidence"] in ["high", "medium"]:
                if extracted_entities["hadm_id"] is not None:
                    hadm_id = extracted_entities["hadm_id"]
                    print(f"Using extracted hadm_id from question: {hadm_id}")

                if extracted_entities["section"] is not None:
                    section = extracted_entities["section"]
                    print(f"Using extracted section from question: {section}")

            # Ask for clarification if needed
            elif extracted_entities["needs_clarification"]:
                # Get available options for clarification
                #! I am going to implement a similarity search to find possible hadm_ids from the question
                available_hadm_ids = list(set(
                    [doc.metadata["hadm_id"] for doc in self.chunked_docs[:100]]))  # Sample for speed

                clarification = ask_for_clarification(
                    extracted_entities,
                    {"hadm_ids": available_hadm_ids}
                )

                if clarification["needs_clarification"]:
                    return {
                        "answer": f"I need clarification to better help you:\n\n" +
                        "\n".join([f"• {q}" for q in clarification["clarification_questions"]]) +
                        f"\n\n{clarification['suggested_format']}",
                        "source_documents": [],
                        "citations": [],
                        "needs_clarification": True,
                        "extracted_entities": extracted_entities,
                        "clarification_questions": clarification["clarification_questions"]
                    }

        print(f"Final filters - hadm_id: {hadm_id}, section: {section}")

        # History-aware retriever for follow-up questions
        question_to_search = question
        if len(chat_history) > 0:
            # For follow-up questions, first rephrase using chat history
            standalone_question = safe_llm_invoke(
                self.llm,
                self.condense_q_prompt.format_messages(
                    chat_history=chat_history,
                    input=question
                ),
                fallback_message=question,  # Use original question as fallback
                context="Question rephrasing"
            )
            if isinstance(standalone_question, str) and len(standalone_question.strip()) > 5:
                print(f"Rephrased question: {standalone_question}")
                question_to_search = standalone_question
            else:
                print(
                    "⚠️ LLM rephrasing produced invalid result, using original question")
                question_to_search = question

        result = self.clinical_search(
            question_to_search,
            hadm_id=hadm_id,
            section=section,
            k=k,
            chat_history=chat_history
        )

        # Handling empty results
        if not result.get("source_documents"):
            fallback_message = "No relevant medical records found."
            if hadm_id:
                fallback_message += f" Admission {hadm_id} may not exist in the database."
            if section:
                fallback_message += f" Section '{section}' may not have data for this admission."

        # Update chat history
        chat_history.append(("human", question))
        chat_history.append(("assistant", result["answer"]))

        # Keep only last 30 exchanges to prevent memory issues
        if len(chat_history) > MAX_CHAT_HISTORY:
            chat_history = chat_history[-MAX_CHAT_HISTORY:]

        # Add metadata for clinical context
        result["chat_history"] = chat_history
        result["original_question"] = question
        result["search_question"] = question_to_search
        result["extracted_entities"] = extracted_entities
        result["chat_context"] = chat_context
        result["used_extraction"] = auto_extract and extracted_entities is not None
        result["manual_override"] = {
            "hadm_id": original_hadm_id, "section": original_section}

        return result

    def clinical_rag_query(self, question, top_k=DEFAULT_K):
        """Debug chatbot without chat history"""
        print(f"=== CLINICAL RAG QUERY ===")

        # Use the new clinical chatbot function
        result = self.ask_with_sources_clinical_chatbot(
            question=question,
            chat_history=[],
            k=top_k
        )

        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "source_documents": result["source_documents"]
        }


if __name__ == "__main__":
    from embeddings_manager import load_or_create_vectorstore

    vectorstore, clinical_emb, chunked_docs = load_or_create_vectorstore()

    # Initialize chatbot
    chatbot = ClinicalRAGBot(vectorstore, clinical_emb, chunked_docs)

    # Test queries
    chat_history = []

