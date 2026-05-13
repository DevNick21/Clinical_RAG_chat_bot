"""Main execution script"""
from RAG_chat_pipeline.core.embeddings_manager import load_or_create_vectorstore
from RAG_chat_pipeline.core.clinical_rag import ClinicalRAGBot
from RAG_chat_pipeline.utils.logger import ClinicalLogger


def main():
    """Main execution function"""
    ClinicalLogger.info("Starting Clinical RAG System")

    # Setup embeddings and vectorstore
    ClinicalLogger.info("Setting up embeddings and vectorstore")
    vectorstore, clinical_emb, chunked_docs = load_or_create_vectorstore()

    # Initialize chatbot
    ClinicalLogger.info("Initializing Clinical RAG Bot")
    chatbot = ClinicalRAGBot(vectorstore, clinical_emb, chunked_docs)

    ClinicalLogger.info("Clinical RAG System Ready")
    return chatbot


def initialize_clinical_rag():
    """Initialize and return the clinical RAG chatbot - used by evaluator"""
    return main()


if __name__ == "__main__":
    chatbot = main()

    # Interactive loop (optional)
    ClinicalLogger.info("Clinical RAG Chatbot Ready")
    ClinicalLogger.info("Type 'quit' to exit")

    chat_history = []
    while True:
        try:
            question = input("Ask a question from this database: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            response = chatbot.ask_question(
                question,
                chat_history=chat_history,
                k=5
            )

            print(f"\nAnswer: {response['answer']}")
            print(f"Citations: {len(response.get('citations', []))} sources")
            print("-" * 50)

        except KeyboardInterrupt:
            ClinicalLogger.info("Goodbye")
            break
        except Exception as e:
            ClinicalLogger.error("CLI error", error=str(e))
