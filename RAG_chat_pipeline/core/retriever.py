"""Retrieval layer for the clinical RAG pipeline.

Owns:
  * Per-doc metadata indices for O(1) hadm_id / subject_id / section lookups
  * The polymorphic candidate filter (int | list[int] | None)
  * Cached re-ranking semantic search over a candidate set
  * Global semantic search via the underlying FAISS vectorstore
  * Lightweight ID-mismatch preflight (used to short-circuit before LLM)

Extracted from ClinicalRAGBot during the god-class decomposition. The bot
composes a Retriever instance and delegates all retrieval calls here so
the orchestrator stays focused on prompt construction, audit, and chat
state.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from threading import Lock
from typing import List, Optional, Set, Union

from cachetools import LRUCache
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

from RAG_chat_pipeline.utils.logger import ClinicalLogger


IdLike = Union[int, List[int], None]


class _EmbCache:
    """Thread-safe LRU embedding cache."""

    def __init__(self, embedder, max_items: int = 512):
        self.embedder = embedder
        self._cache = LRUCache(maxsize=max_items)
        self._lock = Lock()

    def get(self, text: str):
        with self._lock:
            vec = self._cache.get(text)
            if vec is not None:
                return vec
        vec = self.embedder.embed_query(text)
        with self._lock:
            self._cache[text] = vec
        return vec


class Retriever:
    """Indexed + cached retrieval over the loaded chunked documents.

    All inputs that take an "id" accept int, list[int], or None. None means
    "no filter on this dimension"; a list means "union across these IDs"
    (used for multi-admission comparison queries).
    """

    # 6-10 digit runs in user text are candidate hadm_id / subject_id mentions.
    # MIMIC-IV uses 8-digit IDs; widening to 6-10 catches typos and the full
    # ID family while staying narrow enough to avoid catching dates.
    _USER_ID_RE = re.compile(r"\b\d{6,10}\b")

    def __init__(self, vectorstore: FAISS, clinical_emb, chunked_docs: List[Document],
                 emb_cache_size: int = 512):
        self.vectorstore = vectorstore
        self.clinical_emb = clinical_emb
        self.chunked_docs = chunked_docs

        self._emb_cache = _EmbCache(clinical_emb, max_items=emb_cache_size)

        ClinicalLogger.info("Initializing performance optimizations...")
        self._build_metadata_indices()

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def _build_metadata_indices(self):
        """Build hadm_id / subject_id / section indices for fast filtering."""
        ClinicalLogger.info("Building optimized metadata indices...")
        start_time = time.time()

        self.hadm_id_index = defaultdict(list)
        self.subject_id_index = defaultdict(list)
        self.section_index = defaultdict(list)
        self.hadm_section_index = defaultdict(list)

        batch_size = 1000
        total_docs = len(self.chunked_docs)
        processed = 0

        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch = self.chunked_docs[batch_start:batch_end]

            for i, doc in enumerate(batch):
                doc_idx = batch_start + i
                hadm_id = doc.metadata.get('hadm_id')
                subject_id = doc.metadata.get('subject_id')
                section = doc.metadata.get('section')

                if hadm_id is not None:
                    try:
                        hadm_id_int = int(hadm_id)
                        self.hadm_id_index[hadm_id_int].append(doc_idx)

                        if subject_id is not None:
                            try:
                                subject_id_int = int(subject_id)
                                self.subject_id_index[subject_id_int].append(doc_idx)
                            except (ValueError, TypeError):
                                if processed < 10:
                                    ClinicalLogger.warning(
                                        f"Invalid subject_id in document {doc_idx}: {subject_id}")

                        if section:
                            section_str = str(section).lower()
                            self.section_index[section_str].append(doc_idx)
                            self.hadm_section_index[(hadm_id_int, section_str)].append(doc_idx)

                    except (ValueError, TypeError):
                        if processed < 10:
                            ClinicalLogger.warning(
                                f"Invalid hadm_id in document {doc_idx}: {hadm_id}")
                        continue

                elif section:
                    section_str = str(section).lower()
                    self.section_index[section_str].append(doc_idx)

                processed += 1

            if total_docs > 5000:
                progress = (batch_end / total_docs) * 100
                ClinicalLogger.debug(
                    f"Index building progress: {progress:.1f}% ({batch_end}/{total_docs})")

        build_time = time.time() - start_time
        ClinicalLogger.info(f"Metadata indices built in {build_time:.2f}s")
        ClinicalLogger.info(
            f"Index stats: {len(self.hadm_id_index)} admissions, "
            f"{len(self.subject_id_index)} subjects, {len(self.section_index)} sections")

    # ------------------------------------------------------------------ #
    # ID utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_id_list(value: IdLike) -> List[int]:
        """Normalise int | list | tuple | set | None to a list of ints (or [])."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [int(v) for v in value if v is not None]
        return [int(value)]

    @classmethod
    def extract_user_ids(cls, question: str) -> Set[int]:
        """Pull hadm_id / subject_id-shaped numbers from the user's text."""
        return {int(m) for m in cls._USER_ID_RE.findall(question or "")}

    # ------------------------------------------------------------------ #
    # Filter / search
    # ------------------------------------------------------------------ #
    def filter_documents(self, hadm_id: IdLike = None, subject_id: IdLike = None,
                         section: Optional[str] = None, limit: int = 50):
        """Return candidate docs for the given filters, or None for global search.

        `hadm_id` and `subject_id` accept int | list[int] | None. When a list
        is supplied (multi-ID query like "compare admissions A and B") we
        union the candidate indices across all IDs, dedupe, and apply the
        limit across the merged set.

        Returns:
          list[Document]  — filter applied, possibly empty
          None            — no filter applicable; caller should do global search
        """
        hadm_ids = self.to_id_list(hadm_id)
        subject_ids = self.to_id_list(subject_id)

        if hadm_ids:
            ClinicalLogger.info(f"Filtering documents for hadm_id(s): {hadm_ids}")
            candidate_indices: List[int] = []
            for hid in hadm_ids:
                if section is not None:
                    indices = self.hadm_section_index.get((hid, section), [])
                    ClinicalLogger.info(
                        f"  hadm_id {hid} + section '{section}': {len(indices)} documents")
                else:
                    indices = self.hadm_id_index.get(hid, [])
                    ClinicalLogger.info(f"  hadm_id {hid}: {len(indices)} documents")
                if not indices:
                    available_keys = list(self.hadm_id_index.keys())[:10]
                    ClinicalLogger.warning(
                        f"No documents for hadm_id {hid}. Available keys (first 10): {available_keys}")
                candidate_indices.extend(indices)

            candidate_indices = list(dict.fromkeys(candidate_indices))

            if len(candidate_indices) > limit:
                ClinicalLogger.info(
                    f"Limiting documents from {len(candidate_indices)} to {limit} for performance")
                candidate_indices = candidate_indices[:limit]

            return [self.chunked_docs[i] for i in candidate_indices]

        if subject_ids:
            ClinicalLogger.info(f"Filtering documents for subject_id(s): {subject_ids}")
            candidate_indices = []
            for sid in subject_ids:
                indices = self.subject_id_index.get(sid, [])
                ClinicalLogger.info(f"  subject_id {sid}: {len(indices)} documents")
                candidate_indices.extend(indices)

            candidate_indices = list(dict.fromkeys(candidate_indices))

            if len(candidate_indices) > limit:
                ClinicalLogger.info(
                    f"Limiting documents from {len(candidate_indices)} to {limit} for performance")
                candidate_indices = candidate_indices[:limit]

            candidate_docs = [self.chunked_docs[i] for i in candidate_indices]
            if section is not None:
                candidate_docs = [doc for doc in candidate_docs
                                  if doc.metadata.get('section', '').lower() == section.lower()]
            return candidate_docs

        return None  # Indicates global search needed

    def semantic_search(self, candidate_docs: List[Document], question: str, k: int) -> List[Document]:
        """Cosine-similarity re-rank over a candidate doc set, top-k.

        Uses the embedding cache to skip recomputation for repeated docs/queries.
        For >100 candidates, processes in batches of 50 to keep memory bounded.
        Falls back to the first k docs on any similarity failure.
        """
        if len(candidate_docs) <= k:
            ClinicalLogger.debug(
                f"Returning all {len(candidate_docs)} documents (less than k={k})")
            return candidate_docs

        try:
            if len(candidate_docs) <= 100:
                question_embedding = self._emb_cache.get(question)

                scored_docs = []
                for doc in candidate_docs:
                    doc_text = doc.page_content[:500]
                    doc_embedding = self._emb_cache.get(doc_text)
                    similarity = cosine_similarity(
                        [question_embedding], [doc_embedding])[0][0]
                    scored_docs.append((similarity, doc))

                scored_docs.sort(key=lambda x: x[0], reverse=True)
                top_docs = [doc for _, doc in scored_docs[:k]]
                ClinicalLogger.debug(
                    f"Selected top {len(top_docs)} documents by direct similarity")
                return top_docs

            ClinicalLogger.info(
                f"Using batch similarity for {len(candidate_docs)} documents (avoiding temporary FAISS)")

            batch_size = 50
            all_scored_docs = []
            question_embedding = self._emb_cache.get(question)
            for i in range(0, len(candidate_docs), batch_size):
                batch = candidate_docs[i:i + batch_size]
                for doc in batch:
                    doc_text = doc.page_content[:500]
                    doc_embedding = self._emb_cache.get(doc_text)
                    similarity = cosine_similarity(
                        [question_embedding], [doc_embedding])[0][0]
                    all_scored_docs.append((similarity, doc))

            all_scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in all_scored_docs[:k]]
            ClinicalLogger.debug(
                f"Selected top {len(top_docs)} documents via batch similarity")
            return top_docs

        except Exception as similarity_error:
            ClinicalLogger.warning(
                f"Similarity search failed: {similarity_error}, using first {k} documents")
            return candidate_docs[:k]

    def global_semantic_search(self, question: str, k: int,
                               section: Optional[str] = None) -> List[Document]:
        """FAISS similarity search over the full corpus, with optional section post-filter."""
        retrieved_docs = self.vectorstore.similarity_search(question, k=k)
        ClinicalLogger.debug(f"Retrieved {len(retrieved_docs)} documents")

        if section is not None:
            original_count = len(retrieved_docs)
            retrieved_docs = [doc for doc in retrieved_docs
                              if doc.metadata.get('section', '').lower() == section.lower()]
            ClinicalLogger.debug(
                f"Section filtering: {original_count} → {len(retrieved_docs)} documents")
        return retrieved_docs

    # ------------------------------------------------------------------ #
    # Preflight
    # ------------------------------------------------------------------ #
    def check_id_mismatch(self, question: str, retrieved_docs: List[Document]) -> Set[int]:
        """Pre-flight check: did the user name IDs that aren't in retrieved docs?

        Returns the set of asked-about IDs that are NOT represented in the
        retrieved docs' metadata. Returns empty set if either:
          - the user didn't mention any IDs (nothing to validate)
          - at least one mentioned ID IS in the retrieved set (mixed case;
            let the prompt handle it)

        A non-empty return value means "all asked-about IDs are missing"
        and the caller should short-circuit with a clean rejection instead
        of feeding mismatched docs to the LLM.
        """
        user_ids = self.extract_user_ids(question)
        if not user_ids:
            return set()

        present: Set[int] = set()
        for doc in retrieved_docs or []:
            md = getattr(doc, "metadata", {}) or {}
            for key in ("hadm_id", "subject_id"):
                val = md.get(key)
                if val is None:
                    continue
                try:
                    present.add(int(val))
                except (TypeError, ValueError):
                    pass

        missing = user_ids - present
        return missing if missing == user_ids else set()
