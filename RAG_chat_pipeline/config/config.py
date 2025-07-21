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
    "llama": "llama3.2:latest",
    "gemma": "gemma:2b",
    "phi3": "phi3:3.8b",
    "tinyllama": "tinyllama:1.1b",
}


# Model name for downloading and saving locally
CLINICAL_MODEL_NAME = model_names[model_in_use][1]

# Local path to save the clinical model
MODELS_DIR = BASE / "models"

# Path where the model will be saved locally or existing model will be loaded from
LOCAL_MODEL_PATH = MODELS_DIR / model_names[model_in_use][0]
LLM_MODEL = llms["deepseek"]


# Path to save the vectorstore
VECTORSTORE_PATH = BASE / "vector_stores" / model_names[model_in_use][2]


DEFAULT_K = 5

# Maximum number of chat history messages to keep
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

# Scoring weights for overall evaluation
EVALUATION_SCORING_WEIGHTS = {
    "factual_accuracy": 0.6,
    "behavior": 0.3,
    "performance": 0.1
}

# Pass thresholds by question category
EVALUATION_PASS_THRESHOLDS = {
    "header": 0.70,
    "diagnoses": 0.75,
    "procedures": 0.75,
    "labs": 0.65,
    "microbiology": 0.65,
    "prescriptions": 0.70,
    "comprehensive": 0.60,
    "default": 0.70
}

# Performance evaluation thresholds
EVALUATION_PERFORMANCE_THRESHOLDS = {
    "slow_search_time": 5.0,
    "moderate_search_time": 2.0,
    "slow_penalty": 0.4,
    "moderate_penalty": 0.2,
    "no_docs_penalty": 0.3
}

# Response length evaluation
EVALUATION_RESPONSE_LENGTH = {
    "too_short": 10,
    "too_long": 1000,
    "short_penalty": 0.3,
    "long_penalty": 0.7,
    "comprehensive_bonus_threshold": 200,
    "comprehensive_bonus": 0.2
}

# Keywords for different medical categories
EVALUATION_MEDICAL_KEYWORDS = {
    "header": {
        "admission_type": ["emergency", "elective", "urgent", "newborn", "trauma"],
        "expire": ["yes", "no", "expired", "deceased", "alive"],
        "date_pattern": r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}',
        "time_pattern": r'\d{1,2}:\d{2}'
    },
    "diagnoses": {
        "primary": ["icd", "diagnosis", "condition", "disease", "disorder"],
        "code_pattern": r'[A-Z]\d{2}\.?\d*'
    },
    "procedures": {
        "primary": ["procedure", "surgery", "operation", "intervention", "performed"],
        "code_pattern": r'[0-9A-Z]{3,7}'
    },
    "labs": {
        "primary": ["lab", "test", "result", "value", "blood", "urine", "chemistry"],
        "units": ["mg/dl", "mmol/l", "unit", "normal", "abnormal", "high", "low"],
        "value_pattern": r'\d+\.?\d*'
    },
    "microbiology": {
        "primary": ["culture", "specimen", "microbiology", "bacteria", "infection", "swab"],
        "specimen_types": ["blood", "urine", "sputum", "wound", "stool"]
    },
    "prescriptions": {
        "primary": ["medication", "drug", "prescription", "dose", "mg", "tablet", "capsule"],
        "timing": ["daily", "twice", "hours", "morning", "evening", "prn"],
        "dosage_pattern": r'\d+\s*(mg|ml|units|tablets?)'
    },
    "comprehensive": {
        "sections": {
            "admission": ["admitted", "admission", "type"],
            "diagnoses": ["diagnosis", "condition", "icd"],
            "procedures": ["procedure", "surgery", "operation"],
            "labs": ["lab", "test", "result"],
            "medications": ["medication", "drug", "prescription"]
        }
    }
}

# Scoring weights for structured data validation
EVALUATION_STRUCTURED_DATA_SCORING = {
    "keywords_found_weight": 0.5,
    "codes_found_weight": 0.3,
    "additional_keywords_weight": 0.2,
    "no_data_response_score": 0.8
}

# Scoring weights for specific categories
EVALUATION_CATEGORY_SCORING_WEIGHTS = {
    "labs": {
        "lab_terms": 0.4,
        "values": 0.3,
        "units": 0.3
    },
    "prescriptions": {
        "med_terms": 0.5,
        "dosages": 0.3,
        "timing": 0.2
    }
}

# Inappropriate response detection
EVALUATION_INAPPROPRIATE_RESPONSES = {
    "phrases": ["i don't know", "i cannot", "unable to process"],
    "penalty_score": 0.0
}

# Good response indicators
EVALUATION_GOOD_RESPONSE_INDICATORS = {
    "source_citation": ["according to", "based on"],
    "bonus_score": 1.0
}

# No data response patterns
EVALUATION_NO_DATA_PATTERNS = [
    "no records", "not found", "does not exist", "no", "none", "not found"]

# Basic medical scoring
EVALUATION_BASIC_MEDICAL_SCORING = {
    "max_score": 0.8,
    "overlap_scale_factor": 10
}

# Default evaluation parameters
EVALUATION_DEFAULT_PARAMS = {
    "default_k": 5,
    "search_strategy": "auto",
    "short_evaluation_limit": 10,
    "quick_test_limit": 3
}

# Output formatting
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

    return embedding_model, llm_model

# =============================================
# CONFIGURATION UTILITIES
# =============================================


def get_config_summary():
    """Get a summary of current configuration."""
    llm_nickname = next(k for k, v in llms.items() if v == LLM_MODEL)
    return f"""
🔧 Current Configuration:
   Embedding Model: {model_in_use} ({CLINICAL_MODEL_NAME})
   LLM Model: {llm_nickname} ({LLM_MODEL})
   Vector Store: {VECTORSTORE_PATH.name}
   Model Path: {LOCAL_MODEL_PATH.name}
"""


# Initialize with defaults (this happens when config is imported)
# This ensures the config is always in a valid state
try:
    set_models()  # Uses defaults: ms-marco + deepseek
except Exception:
    pass  # Fallback to original static values if something goes wrong
print(BASE)
