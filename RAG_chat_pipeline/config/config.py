from pathlib import Path

# Initializes the models, embedings, and vectorstore for the RAG chat pipeline
"""Configuration and constants"""
BASE = Path(__file__).resolve().parents[2]  # Base directory for the project
CHUNKED_DOCS_PATH = BASE / "mimic_sample_1000" / "chunked_docs.pkl"

# Default model to use, can be changed to "multi-qa", "mini-lm", or "static-retr"
model_in_use = "mini-lm"

# Model configuration
# Clinical model for entity extraction and question answering
model_names = {"ms-marco": ["S-PubMedBert-MS-MARCO", "pritamdeka/S-PubMedBert-MS-MARCO", "faiss_mimic_sample1000_ms-marco"],
               "multi-qa": ["multi-qa-mpnet-base-cos-v1", "sentence-transformers/multi-qa-mpnet-base-cos-v1", "faiss_mimic_sample1000_multi-qa"],
               "mini-lm": ["all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "faiss_mimic_sample1000_mini-lm"],
               "biomedbert": ["BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "faiss_mimic_sample1000_biomedbert"],
               "mpnet-v2": ["all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", "faiss_mimic_sample1000_mpnet-v2"],
               "e5-base": ["e5-base-v2", "intfloat/e5-base-v2", "faiss_mimic_sample1000_e5-base"],
               "BioLORD": ["BioLORD-2023-C", "sentence-transformers/BioLORD-2023-C", "faiss_mimic_sample1000_BioLORD"],
               "BioBERT": ["BioBERT-mnli-snli-scinli-scitail-mednli-stsb", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", "faiss_mimic_sample1000_BioBERT"],
               "MedQuAD": ["S-PubMedBert-MedQuAD", "TimKond/S-PubMedBert-MedQuAD", "faiss_mimic_sample1000_MedQuAD"]
               }

llms = {
    "deepseek": "deepseek-r1:1.5b",
    "qwen": "qwen3:1.7b",
    "llama": "llama3.2:1b",
    "gemma": "gemma3:1b",
    "phi3": "phi3:3.8b",
    "tinyllama": "tinyllama:1.1b",
}


# Model name for downloading and saving locally
CLINICAL_MODEL_NAME = model_names[model_in_use][1]

# Local path to save the clinical model
MODELS_DIR = BASE / "models"

# Path where the model will be saved locally or existing model will be loaded from
LOCAL_MODEL_PATH = MODELS_DIR / model_names[model_in_use][0]
# Changed to fastest model for better performance
LLM_MODEL = llms["tinyllama"]

# Embedding model for vector embeddings
EMBEDDING_MODEL = model_names[model_in_use][0]

# Path to save the vectorstore
VECTORSTORE_PATH = BASE / "vector_stores" / model_names[model_in_use][2]

# Vector stores mapping for API
vector_stores = {key: val[2] for key, val in model_names.items()}

DEFAULT_K = 5  # Slightly higher for better recall during evaluation

# Retrieval and candidate limits (configurable)
# - RETRIEVAL_MAX_K: caps the number of documents considered in focused searches
# - GLOBAL_SEARCH_MAX_K: caps the number of docs returned by global vector search
# - CANDIDATE_DOC_LIMIT: max docs to consider after metadata filtering
# - FINAL_DOCS_LIMIT: final number of docs passed to LLM after structuring
# Streaming variants are tuned for lower latency in streaming endpoints
RETRIEVAL_MAX_K = 5
GLOBAL_SEARCH_MAX_K = 20
CANDIDATE_DOC_LIMIT = 20
FINAL_DOCS_LIMIT = 5

# Streaming-specific limits (lower for responsiveness)
STREAMING_CANDIDATE_DOC_LIMIT = 10
STREAMING_GLOBAL_SEARCH_MAX_K = 5
STREAMING_FINAL_DOCS_LIMIT = 2

# Feature flags (centralized) - optimized for performance
ENABLE_REPHRASING = False  # disabled for performance and to avoid validation issues
ENABLE_ENTITY_EXTRACTION = False  # disabled for performance

# Logging (set to quiet for max evaluation speed)
LOG_LEVEL = "quiet"  # options: quiet, error, warning, info, debug

# Maximum number of chat history messages to keep (reduced for performance)
# This is used to limit the context size for the LLM
MAX_CHAT_HISTORY = 60

# Section keywords for entity extraction
SECTION_KEYWORDS = {
    "diagnoses": ["diagnoses", "diagnosis", "conditions", "diseases", "dx", "icd", "icd codes", "diagnosis icd"],
    "procedures": ["procedures", "operations", "surgery", "interventions", "procedures icd"],
    "labs": ["labs", "laboratory", "test results", "lab results", "tests", "lab", "laboratory results", "lab tests"],
    "prescriptions": ["medications", "drugs", "prescriptions", "meds", "orders", "emars", "poe", "pharmacy", "medication"],
    "microbiology": ["microbiology", "cultures", "infections", "micro"],
    "header": ["header", "admission", "discharge", "admit", "admittime", "dischtime", "admission type"]
}

# =============================================
# RAG EVALUATION CONFIGURATION
# =============================================

# Semantic similarity configuration for BioBERT-based evaluation
SEMANTIC_EVALUATION_CONFIG = {
    "biobert_model_path": "models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "similarity_threshold": 0.60,  # Adjusted threshold for clinical domain (was 0.75)
    "batch_size": 32,              # Embedding batch size
    "max_sequence_length": 512     # Max tokens for BioBERT
}

# Core evaluation parameters
EVALUATION_DEFAULT_PARAMS = {
    "default_k": 5,  # Number of documents to retrieve for evaluation
    "search_strategy": "fast",
    "short_evaluation_limit": 5,
    "quick_test_limit": 3
}

# Output formatting (keep for display purposes)
EVALUATION_OUTPUT_CONFIG = {
    "answer_preview_length": 200,
    "question_preview_length": 60,
    "separator_length": 50,
    "long_separator_length": 70
}

# =============================================
# DYNAMIC CONFIGURATION MANAGEMENT
# =============================================


def set_models(embedding_model: str = "ms-marco", llm_model: str = "deepseek"):
    """
    Dynamically set the embedding and LLM models for the current session.
    This updates the module-level variables without modifying the config file.

    Args:
        embedding_model: Embedding model nickname (ms-marco, multi-qa, mini-lm, static-retr)
        llm_model: LLM model nickname (deepseek, qwen, llama)

    Returns:
        tuple: (embedding_model, llm_model) that were set

    Raises:
        ValueError: If invalid model names are provided
    """
    global model_in_use, LLM_MODEL, CLINICAL_MODEL_NAME, LOCAL_MODEL_PATH, VECTORSTORE_PATH

    # Validate embedding model
    if embedding_model not in model_names:
        raise ValueError(
            f"Invalid embedding model: {embedding_model}. Available: {list(model_names.keys())}")

    # Validate LLM model
    if llm_model not in llms:
        raise ValueError(
            f"Invalid LLM model: {llm_model}. Available: {list(llms.keys())}")

    # Update global variables
    model_in_use = embedding_model
    LLM_MODEL = llms[llm_model]

    # Update embedding and vectorstore paths to match selected embedding model
    try:
        selected = model_names[model_in_use]
        # selected -> [local_dir_name, hf_id, vectorstore_dir]
        MODELS_DIR_LOCAL = BASE / "models"
        CLINICAL_MODEL_NAME = selected[1]  # HF repo id
        LOCAL_MODEL_PATH = MODELS_DIR_LOCAL / selected[0]
        VECTORSTORE_PATH = BASE / "vector_stores" / selected[2]
    except Exception:
        # If anything goes wrong, leave previous paths as-is
        pass

    return embedding_model, llm_model

# =============================================
# CONFIGURATION UTILITIES
# =============================================


def get_config_summary():
    """Get a summary of current configuration."""
    llm_nickname = next(k for k, v in llms.items() if v == LLM_MODEL)
    return f"""
 Current Configuration:
   Embedding Model: {model_in_use} ({CLINICAL_MODEL_NAME})
   LLM Model: {llm_nickname} ({LLM_MODEL})
   Vector Store: {VECTORSTORE_PATH.name}
   Model Path: {LOCAL_MODEL_PATH.name}
"""


# Initialize with defaults (this happens when config is imported)
# This ensures the config is always in a valid state
try:
    set_models()
except Exception:
    pass  # Fallback to original static values if something goes wrong
