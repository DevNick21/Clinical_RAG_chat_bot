from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClinicalRAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRAG_", frozen=True)

    embedding_model: str = "biomedbert"
    llm_model: str = "llama"

    default_k: int = Field(default=5, ge=1, le=100)
    max_chat_history: int = Field(default=60, ge=1, le=200)

    retrieval_max_k: int = Field(default=5, ge=1, le=100)
    global_search_max_k: int = Field(default=20, ge=1, le=200)
    candidate_doc_limit: int = Field(default=20, ge=1, le=500)
    final_docs_limit: int = Field(default=5, ge=1, le=100)

    streaming_candidate_doc_limit: int = Field(default=20, ge=1, le=500)
    streaming_global_search_max_k: int = Field(default=20, ge=1, le=200)
    streaming_final_docs_limit: int = Field(default=5, ge=1, le=100)

    enable_rephrasing: bool = True
    enable_entity_extraction: bool = True

    log_level: str = "info"

    section_keywords: Dict[str, List[str]] = Field(default_factory=lambda: {
        "diagnoses": ["diagnoses", "diagnosis", "conditions", "diseases", "dx", "icd", "icd codes", "diagnosis icd"],
        "procedures": ["procedures", "operations", "surgery", "interventions", "procedures icd"],
        "labs": ["labs", "laboratory", "test results", "lab results", "tests", "lab", "laboratory results", "lab tests"],
        "prescriptions": ["medications", "drugs", "prescriptions", "meds", "orders", "emars", "poe", "pharmacy", "medication"],
        "microbiology": ["microbiology", "cultures", "infections", "micro"],
        "header": ["header", "discharge", "admittime", "dischtime", "admission type"],
    })


@lru_cache(maxsize=1)
def get_settings() -> ClinicalRAGSettings:
    return ClinicalRAGSettings()
