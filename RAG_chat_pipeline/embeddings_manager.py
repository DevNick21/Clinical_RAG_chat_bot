"""Embedding model and vectorstore management"""
import pickle
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import CLINICAL_MODEL_NAME, LOCAL_MODEL_PATH, VECTORSTORE_PATH, CHUNKED_DOCS_PATH


def setup_clinical_embeddings():
    """Setup clinical embeddings with local model saving/loading"""

    # Check if model exists locally
    if LOCAL_MODEL_PATH.exists() and any(LOCAL_MODEL_PATH.iterdir()):
        # Attempt to load local model
        try:
            clinical_emb = HuggingFaceEmbeddings(
                model_name=str(LOCAL_MODEL_PATH),
                encode_kwargs={"batch_size": 16}
            )
            # Test the model to ensure it's working
            test_vector = clinical_emb.embed_query("test medical query")
            print(
                f"Local model loaded successfully (test vector dim: {len(test_vector)})")
            return clinical_emb

        except Exception as e:
            print(f"⚠️ Error loading local model: {e}")
            print("Downloading model...")

    # Download and save model locally

    # Create directory
    LOCAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Download using SentenceTransformer first
    model = SentenceTransformer(CLINICAL_MODEL_NAME)
    model.save(str(LOCAL_MODEL_PATH))
    print(f"Model saved to: {LOCAL_MODEL_PATH}")

    # LangChain embedding wrapper for SentenceTransformers (STMs)
    clinical_emb = HuggingFaceEmbeddings(
        model_name=str(LOCAL_MODEL_PATH),
        encode_kwargs={"batch_size": 16}
    )

    # Test the model
    test_vector = clinical_emb.embed_query("test medical query")
    print(f"Model setup complete (test vector dim: {len(test_vector)})")

    return clinical_emb


def load_or_create_vectorstore():
    """Load existing vectorstore or create new one"""
    clinical_emb = setup_clinical_embeddings()

    with open(CHUNKED_DOCS_PATH, "rb") as f:
        chunked_docs = pickle.load(f)

    # Try to load existing vectorstore
    try:
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            clinical_emb,
            allow_dangerous_deserialization=True
        )
        print("Vectorstore loaded successfully")
        return vectorstore, clinical_emb, chunked_docs

    except Exception as e:
        print(f"⚠️ Error loading vectorstore: {e}")

        if chunked_docs is None:
            raise ValueError(
                "No existing vectorstore found and no chunked_docs provided to create new one")

        print("Creating new vectorstore...")
        vectorstore = FAISS.from_documents(chunked_docs, clinical_emb)
        vectorstore.save_local(VECTORSTORE_PATH)

        # Save chunked docs
        with open(CHUNKED_DOCS_PATH, "wb") as f:
            pickle.dump(chunked_docs, f)

        print("✅ New vectorstore created and saved")
        return vectorstore, clinical_emb, chunked_docs


if __name__ == "__main__":
    vectorstore, clinical_emb, chunked_docs = load_or_create_vectorstore()
    print("Embeddings setup complete!")
