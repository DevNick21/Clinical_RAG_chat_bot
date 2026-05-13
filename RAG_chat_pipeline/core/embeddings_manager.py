"""Embedding model and vectorstore management"""
import os
import tempfile
from pathlib import Path

# Import compatibility fix before sentence-transformers
from RAG_chat_pipeline.utils import huggingface_compat  # Auto-patches on import

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from RAG_chat_pipeline.config import config as cfg
from RAG_chat_pipeline.utils.data_provider import DataProvider
from RAG_chat_pipeline.utils.logger import ClinicalLogger

load_dotenv()


# Cache directory for the winner FAISS index pulled from Blob.
# tempfile.gettempdir() resolves to /tmp on Linux (incl. ACA containers)
# and the user's temp dir on Windows. Files are downloaded once per cold
# start; subsequent loads in the same container reuse the cache.
_FAISS_CACHE_DIR = Path(tempfile.gettempdir()) / "faiss_winner"
_BLOB_FAISS_PREFIX = "prod-index/faiss_winner/"


def _ensure_blob_index_cached() -> Path:
    """Ensure the winner FAISS index from Blob is present at _FAISS_CACHE_DIR.

    Lists every blob under prod-index/faiss_winner/ and downloads any that
    aren't already cached. Returns the local cache path, ready to be
    passed to FAISS.load_local().

    Manual cache refresh: rm -rf <cache dir> and call again.
    """
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    account = os.getenv("AZURE_STORAGE_ACCOUNT")
    container = os.getenv("AZURE_BLOB_CONTAINER")
    if not account or not container:
        raise RuntimeError(
            "USE_BLOB_INDEX=true but AZURE_STORAGE_ACCOUNT/AZURE_BLOB_CONTAINER "
            "are not set in .env."
        )

    _FAISS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    container_client = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    ).get_container_client(container)

    blobs = list(container_client.list_blobs(name_starts_with=_BLOB_FAISS_PREFIX))
    if not blobs:
        raise RuntimeError(
            f"No blobs found at {container}/{_BLOB_FAISS_PREFIX} — "
            "has Step 1 (seed_upload) been run?"
        )

    downloaded = 0
    for blob in blobs:
        rel = blob.name[len(_BLOB_FAISS_PREFIX):]
        if not rel:
            continue  # the prefix itself
        target = _FAISS_CACHE_DIR / rel
        if target.exists() and target.stat().st_size == blob.size:
            continue  # already cached
        target.parent.mkdir(parents=True, exist_ok=True)
        size_mb = (blob.size or 0) / (1024 * 1024)
        ClinicalLogger.info("Downloading FAISS blob", blob=rel, size_mb=f"{size_mb:.1f}")
        with target.open("wb") as f:
            container_client.get_blob_client(blob.name).download_blob().readinto(f)
        downloaded += 1

    if downloaded == 0:
        ClinicalLogger.info("FAISS index cache hit", path=str(_FAISS_CACHE_DIR))
    else:
        ClinicalLogger.info(
            "FAISS index ready",
            path=str(_FAISS_CACHE_DIR),
            files_downloaded=downloaded,
        )
    return _FAISS_CACHE_DIR


def setup_clinical_embeddings():
    """Setup clinical embeddings with local model saving/loading"""

    # Check if model exists locally
    if cfg.LOCAL_MODEL_PATH.exists() and any(cfg.LOCAL_MODEL_PATH.iterdir()):
        # Attempt to load local model
        try:
            clinical_emb = HuggingFaceEmbeddings(
                model_name=str(cfg.LOCAL_MODEL_PATH),
                encode_kwargs={"batch_size": 16}
            )
            # Test the model to ensure it's working
            test_vector = clinical_emb.embed_query("test medical query")
            ClinicalLogger.info(
                "Local model loaded successfully",
                test_vector_dim=len(test_vector),
            )
            return clinical_emb

        except Exception as e:
            ClinicalLogger.warning("Error loading local model", error=str(e))
            ClinicalLogger.info("Downloading model")

    # Download and save model locally

    # Create directory
    cfg.LOCAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Download using SentenceTransformer first
    model = SentenceTransformer(cfg.CLINICAL_MODEL_NAME)
    model.save(str(cfg.LOCAL_MODEL_PATH))
    ClinicalLogger.info("Model saved", path=str(cfg.LOCAL_MODEL_PATH))

    # LangChain embedding wrapper for SentenceTransformers (STMs)
    clinical_emb = HuggingFaceEmbeddings(
        model_name=str(cfg.LOCAL_MODEL_PATH),
        encode_kwargs={"batch_size": 16}
    )

    # Test the model
    test_vector = clinical_emb.embed_query("test medical query")
    ClinicalLogger.info("Model setup complete", test_vector_dim=len(test_vector))

    return clinical_emb


def load_or_create_vectorstore():
    """Load existing vectorstore or create new one.

    Chunked docs always come through DataProvider (which decides between
    Blob / local-real / local-synthetic per its own priority order).

    Vectorstore source is env-toggled:
      USE_BLOB_INDEX=true  -> download prod-index/faiss_winner/ from Blob,
                              cache to /tmp/faiss_winner/ on first call,
                              load from cache. No re-download policy:
                              `rm -rf /tmp/faiss_winner` to refresh.
      otherwise            -> load from cfg.VECTORSTORE_PATH (local).
    """
    clinical_emb = setup_clinical_embeddings()

    data_provider = DataProvider()

    try:
        chunked_docs = data_provider.load_chunked_docs()
    except Exception as e:
        ClinicalLogger.warning("Error loading chunked documents", error=str(e))
        chunked_docs = None

    use_blob_index = os.getenv("USE_BLOB_INDEX", "").lower() in ("true", "1", "yes")
    if use_blob_index:
        index_path = _ensure_blob_index_cached()
        ClinicalLogger.info("Loading vectorstore from Blob cache", path=str(index_path))
    else:
        index_path = cfg.VECTORSTORE_PATH
        ClinicalLogger.info("Loading vectorstore from local", path=str(index_path))

    try:
        vectorstore = FAISS.load_local(
            str(index_path),
            clinical_emb,
            allow_dangerous_deserialization=True
        )
        ClinicalLogger.info("Vectorstore loaded successfully")
        return vectorstore, clinical_emb, chunked_docs

    except Exception as e:
        ClinicalLogger.warning("Error loading vectorstore", error=str(e))

        if chunked_docs is None:
            raise ValueError(
                "No existing vectorstore found and no chunked_docs provided to create new one")

        # Rebuild path: only writes to local. The Blob copy of the winner
        # index is a release artefact, not something we regenerate at runtime.
        ClinicalLogger.info("Creating new vectorstore")
        vectorstore = FAISS.from_documents(chunked_docs, clinical_emb)
        vectorstore.save_local(cfg.VECTORSTORE_PATH)

        ClinicalLogger.info("New vectorstore created and saved")
        return vectorstore, clinical_emb, chunked_docs


if __name__ == "__main__":
    vectorstore, clinical_emb, chunked_docs = load_or_create_vectorstore()
    ClinicalLogger.info("Embeddings setup complete")
