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

    # Entity extraction: regex-tier is always on (sub-ms, deterministic).
    # The LLM fallback fires only when regex finds nothing; routed to
    # the FAST non-reasoning model (`get_fast_llm`), so its cost on the
    # TTFP critical path is ~300ms rather than the ~1.5s the reasoning
    # model used to charge. Default ON now that the fast path exists.
    enable_entity_extraction: bool = True
    enable_llm_entity_extraction: bool = True

    # Question rephrasing for short conversational follow-ups. Also
    # routed through the FAST model when enabled. Template-based
    # fallback still runs as the safe baseline. Default ON for the
    # same reason as above — small model + structured task.
    enable_rephrasing: bool = True

    # LLM-as-judge audit on the final answer. Runs AFTER the stream
    # against the capable audit model (`get_audit_llm`) so it never
    # touches TTFP. The regex faithfulness pre-filter in
    # `audit.claim_checker.check_claims` always runs; the LLM judge
    # narrows / confirms its flagged sentences.
    enable_llm_audit: bool = True

    # Emit an early SSE `retrieval_done` event after retrieval finishes
    # but before the LLM begins streaming. Lets the frontend render a
    # "Retrieved N documents…" badge during the LLM's reasoning phase,
    # cutting perceived TTFP even when the model itself can't start
    # emitting tokens any sooner.
    enable_streaming_warmup_event: bool = True

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
