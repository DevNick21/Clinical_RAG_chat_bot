from pathlib import Path

# Initializes the models, embedings, and vectorstore for the RAG chat pipeline
"""Configuration and constants"""
CHUNKED_DOCS_PATH = "../mimic_sample_1000/chunked_docs.pkl"

# Default model to use, can be changed to "multi-qa", "mini-lm", or "static-retr"
model_in_use = "ms-marco"

# Model configuration
# Clinical model for entity extraction and question answering
model_names = {"ms-marco": ["S-PubMedBert-MS-MARCO", "pritamdekar/S-PubMedBert-MS-MARCO", "faiss_mimic_sample1000_ms-marco"],
               "multi-qa": ["multi-qa-mpnet-base-cos-v1", "sentence-transformers/multi-qa-mpnet-base-cos-v1", "faiss_mimic_sample1000_multi-qa"],
               "mini-lm": ["all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "faiss_mimic_sample1000_mini-lm"],
               "static-retr": ["multi-qa-mpnet-base-cos-v1", "sentence-transformers/multi-qa-mpnet-base-cos-v1", "faiss_mimic_sample1000_static-retr"]}
llms = {
    "deepseek": "deepseek-r1:1.5b",
    "qwen": "qwen3:1.7b",
    "llama": "llama3.2:latest",
}

# Set the clinical model and LLM to be used in the pipeline
# You can change these to use different models as needed

# Model name for downloading and saving locally
CLINICAL_MODEL_NAME = model_names[model_in_use][1]

# Local path to save the clinical model
MODELS_DIR = Path("../models")

# Path where the model will be saved locally or existing model will be loaded from
LOCAL_MODEL_PATH = MODELS_DIR / model_names[model_in_use][0]
LLM_MODEL = llms["deepseek"]


# Path to save the vectorstore
VECTORSTORE_PATH = f"../vector_stores/{model_names[model_in_use][2]}"
CHUNKED_DOCS_PATH = "../mimic_sample_1000/chunked_docs.pkl"


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
    "microbiology": ["microbiology", "cultures", "infections", "micro"]
}
