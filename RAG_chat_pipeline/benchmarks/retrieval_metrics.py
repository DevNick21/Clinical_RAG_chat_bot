"""Module for evaluating retrieval performance with precision/recall metrics"""
import numpy as np
from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


class RetrievalMetricsEvaluator:
    """Evaluates retrieval performance using precision/recall metrics"""

    def __init__(self, gold_standard_docs: Dict[str, Set[str]] = None):
        """
        Initialize with gold standard document IDs for each question

        Args:
            gold_standard_docs: Dictionary mapping question IDs to sets of relevant document IDs
        """
        self.gold_standard = gold_standard_docs or {}

    def build_gold_standard_from_questions(self, gold_questions: List[Dict]) -> Dict[str, Set[str]]:
        """
        Build gold standard from existing gold questions by using hadm_id and section
        as the basis for relevant documents

        Args:
            gold_questions: List of gold questions with hadm_id and category info

        Returns:
            Dictionary mapping question IDs to sets of relevant document identifiers
        """
        gold_standard = {}

        for i, question in enumerate(gold_questions):
            question_id = f"q_{i}"
            hadm_id = question.get("hadm_id")
            category = question.get("category")

            # Create document identifiers that should be relevant for this question
            relevant_docs = set()

            if hadm_id and str(hadm_id) != "abc123":
                # For specific patient questions, relevant docs are from that patient
                if category and category != "comprehensive":
                    # For specific category questions, the relevant section
                    relevant_docs.add(f"{hadm_id}_{category}")
                else:
                    # For comprehensive questions, multiple sections from same patient
                    sections = ["diagnoses", "procedures",
                                "prescriptions", "labevents", "microbiologyevents"]
                    for section in sections:
                        relevant_docs.add(f"{hadm_id}_{section}")

            gold_standard[question_id] = relevant_docs

        return gold_standard

    def extract_doc_id(self, doc: Any) -> str:
        """
        Extract document ID from a retrieved document

        Args:
            doc: Retrieved document object

        Returns:
            String identifier for the document
        """
        try:
            hadm_id = doc.metadata.get('hadm_id', 'unknown')
            section = doc.metadata.get('section', 'unknown')
            return f"{hadm_id}_{section}"
        except (AttributeError, KeyError):
            return "unknown_unknown"

    def evaluate_retrieval_for_question(self, question_id: str, retrieved_docs: List[Any],
                                        k: int = None) -> Dict[str, float]:
        """
        Evaluate retrieval performance for a single question

        Args:
            question_id: Identifier for the question
            retrieved_docs: List of retrieved documents
            k: Number of documents to consider (if None, use all)

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if question_id not in self.gold_standard:
            logger.warning(f"No gold standard for question {question_id}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "relevant_retrieved": 0,
                "total_retrieved": len(retrieved_docs),
                "total_relevant": 0,
                "warning": "No gold standard available"
            }

        gold_doc_ids = self.gold_standard[question_id]

        # Limit retrieved documents if k is specified
        if k is not None:
            retrieved_docs = retrieved_docs[:k]

        # Extract document IDs from retrieved documents
        retrieved_doc_ids = set()
        for doc in retrieved_docs:
            doc_id = self.extract_doc_id(doc)
            retrieved_doc_ids.add(doc_id)

        # Calculate metrics
        relevant_retrieved = gold_doc_ids.intersection(retrieved_doc_ids)

        precision = len(relevant_retrieved) / \
            len(retrieved_docs) if retrieved_docs else 0.0
        recall = len(relevant_retrieved) / \
            len(gold_doc_ids) if gold_doc_ids else 0.0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "relevant_retrieved": len(relevant_retrieved),
            "total_retrieved": len(retrieved_docs),
            "total_relevant": len(gold_doc_ids)
        }

    def evaluate_retrieval_batch(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Evaluate retrieval metrics for a batch of questions

        Args:
            results: Dictionary mapping question IDs to retrieval results
                    Each result should have "source_documents" key

        Returns:
            Dictionary with aggregated metrics
        """
        metrics_by_question = {}
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for question_id, result in results.items():
            question_metrics = self.evaluate_retrieval_for_question(
                question_id,
                result.get("source_documents", [])
            )

            if "warning" not in question_metrics:
                metrics_by_question[question_id] = question_metrics
                precision_scores.append(question_metrics["precision"])
                recall_scores.append(question_metrics["recall"])
                f1_scores.append(question_metrics["f1"])

        # Calculate aggregate statistics
        if not precision_scores:  # No valid metrics calculated
            return {
                "metrics_by_question": metrics_by_question,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_f1": 0.0,
                "median_precision": 0.0,
                "median_recall": 0.0,
                "median_f1": 0.0,
                "questions_evaluated": 0
            }

        return {
            "metrics_by_question": metrics_by_question,
            "mean_precision": np.mean(precision_scores),
            "mean_recall": np.mean(recall_scores),
            "mean_f1": np.mean(f1_scores),
            "median_precision": np.median(precision_scores),
            "median_recall": np.median(recall_scores),
            "median_f1": np.median(f1_scores),
            "questions_evaluated": len(precision_scores),
            "std_precision": np.std(precision_scores),
            "std_recall": np.std(recall_scores),
            "std_f1": np.std(f1_scores)
        }


def format_retrieval_metrics_summary(metrics: Dict[str, Any]) -> str:
    """
    Format retrieval metrics into a readable summary string

    Args:
        metrics: Dictionary returned by evaluate_retrieval_batch

    Returns:
        Formatted string summary of retrieval performance
    """
    if metrics.get("questions_evaluated", 0) == 0:
        return "No retrieval metrics available (no gold standard questions)"

    return f"""
RETRIEVAL PERFORMANCE METRICS
{'='*40}
Questions Evaluated: {metrics['questions_evaluated']}
Mean Precision: {metrics['mean_precision']:.3f} (±{metrics.get('std_precision', 0):.3f})
Mean Recall: {metrics['mean_recall']:.3f} (±{metrics.get('std_recall', 0):.3f})
Mean F1-Score: {metrics['mean_f1']:.3f} (±{metrics.get('std_f1', 0):.3f})

Median Precision: {metrics['median_precision']:.3f}
Median Recall: {metrics['median_recall']:.3f}
Median F1-Score: {metrics['median_f1']:.3f}
"""
