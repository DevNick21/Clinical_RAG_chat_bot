"""Persistent audit log for clinical RAG requests.

One JSON record per request, written under AUDIT_LOG_DIR (default ./audit).
Each record captures the question, retrieved documents, reasoning summaries
emitted by the model, the final answer, and any unsupported claims flagged
by the claim checker.

Storage is intentionally simple (one file per request). Migrate to Azure
Blob append-only storage in Phase G when the API is deployed.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

AUDIT_LOG_DIR = Path(os.getenv("AUDIT_LOG_DIR", "./audit"))

# How much doc content to persist per record. Full content is verbose;
# 1k chars is enough for a reviewer to recognise the source.
_DOC_PREVIEW_CHARS = 1000


def log_request(
    question: str,
    retrieved_docs: List[Any],
    reasoning_summaries: List[str],
    answer: str,
    unsupported_claims: List[str],
    response_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist an audit record and return its audit_id.

    Audit ID format: `YYYYMMDD_HHMMSS_<8hex>` — also the filename stem.
    """
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    audit_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    record = {
        "audit_id": audit_id,
        "timestamp_utc": ts.isoformat(),
        "question": question,
        "retrieved_docs": [
            {
                "content": getattr(doc, "page_content", "")[:_DOC_PREVIEW_CHARS],
                "metadata": getattr(doc, "metadata", {}),
            }
            for doc in retrieved_docs
        ],
        "reasoning_summaries": reasoning_summaries,
        "answer": answer,
        "unsupported_claims": unsupported_claims,
        "response_id": response_id,
        "metadata": metadata or {},
    }

    path = AUDIT_LOG_DIR / f"{audit_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)

    return audit_id
