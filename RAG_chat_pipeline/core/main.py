"""Main execution script"""
from RAG_chat_pipeline.core.embeddings_manager import load_or_create_vectorstore
from RAG_chat_pipeline.core.clinical_rag import ClinicalRAGBot


def main():
    """Main execution function"""
    print("🚀 Starting Clinical RAG System...")

    # Setup embeddings and vectorstore
    print("\nSetting up embeddings and vectorstore...")
    vectorstore, clinical_emb, chunked_docs = load_or_create_vectorstore()

    # Initialize chatbot
    print("\nInitializing Clinical RAG Bot...")
    chatbot = ClinicalRAGBot(vectorstore, clinical_emb, chunked_docs)

    print("✅ Clinical RAG System Ready!")
    return chatbot


def initialize_clinical_rag():
    """Initialize and return the clinical RAG chatbot - used by evaluator"""
    return main()


if __name__ == "__main__":
    chatbot = main()

    # Interactive loop (optional)
    print("\n🤖 Clinical RAG Chatbot Ready!")
    print("Type 'quit' to exit\n")

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

            print(f"\n💡 Answer: {response['answer']}")
            print(f"📊 Citations: {len(response['citations'])} sources")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
