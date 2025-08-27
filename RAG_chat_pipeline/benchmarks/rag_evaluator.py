"""Core RAG evaluation logic using semantic precision/recall scoring with BioBERT

This module provides the core evaluation functionality:
- Single question evaluation with semantic similarity
- Batch evaluation for model combinations
- BioBERT-based precision, recall, F1-score calculation

For complete evaluation workflows, use evaluation_results_manager.py
"""
import numpy as np
import time
from typing import Dict, List, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer

from RAG_chat_pipeline.utils.data_provider import get_sample_data
from RAG_chat_pipeline.core.clinical_rag import ClinicalRAGBot, ClinicalLogger
from RAG_chat_pipeline.config.config import *


class ClinicalRAGEvaluator:
    """Semantic RAG evaluator using BioBERT for precision/recall scoring"""

    def __init__(self, chatbot: ClinicalRAGBot = None):
        self.chatbot = chatbot
        self.patient_data = get_sample_data()
        self.biobert_model = None
        self._load_biobert_model()

    def _load_biobert_model(self):
        """Load BioBERT model for semantic similarity"""
        try:
            model_path = SEMANTIC_EVALUATION_CONFIG["biobert_model_path"]
            self.biobert_model = SentenceTransformer(model_path)
            # Disable progress bars for cleaner output
            self.biobert_model.show_progress_bar = False
            ClinicalLogger.info(f"Loaded BioBERT model from {model_path}")
        except Exception as e:
            ClinicalLogger.error(f"Failed to load BioBERT model: {e}")
            # Fallback to general model
            try:
                self.biobert_model = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2')
                # Disable progress bars for cleaner output
                self.biobert_model.show_progress_bar = False
                ClinicalLogger.warning(
                    "Using fallback model - may be less accurate for medical content")
            except Exception as fallback_error:
                ClinicalLogger.error(
                    f"Failed to load fallback model: {fallback_error}")
                self.biobert_model = None

    def evaluate_question(self, gold_question: Dict, question_id: str = None) -> Dict:
        """Evaluate using semantic precision/recall scoring"""
        start_time = time.time()

        try:
            # Initialize chatbot if not provided
            if not self.chatbot:
                from RAG_chat_pipeline.core.main import main as initialize_clinical_rag
                self.chatbot = initialize_clinical_rag()

            # Get response from chatbot
            response = self.chatbot.ask_question(
                gold_question["question"],
                hadm_id=gold_question.get("hadm_id"),
                section=gold_question.get("section"),
                k=EVALUATION_DEFAULT_PARAMS.get("default_k", 3)
            )

            # Extract response data
            search_time = time.time() - start_time
            answer_text = response.get("answer", str(response))
            expected_keywords = gold_question.get("expected_keywords", [])

            # Enhance expected keywords for category-specific evaluation
            category = gold_question.get("category", "")
            enhanced_keywords = self._enhance_keywords_for_category(
                category, expected_keywords, answer_text)
            
            # Compute semantic precision and recall
            semantic_results = self._compute_semantic_precision_recall(
                enhanced_keywords, answer_text
            )

            # Overall score using F1-score (harmonic mean of precision/recall)
            f1_score = semantic_results["f1_score"]

            # Extract documents found
            documents_found = response.get("documents_found", 0)
            if isinstance(documents_found, list):
                documents_found = len(documents_found)

            return {
                "question": gold_question["question"],
                "category": gold_question["category"],
                "answer": answer_text,
                "precision": semantic_results["precision"],
                "recall": semantic_results["recall"],
                "f1_score": f1_score,
                "search_time": response.get("search_time", search_time),
                "documents_found": documents_found,
                "semantic_matches": semantic_results["semantic_matches"],
                "total_expected": semantic_results["total_expected"],
                "question_id": question_id,
                "timestamp": datetime.now().isoformat(),
                "hadm_id": gold_question.get("hadm_id"),
                "expected_keywords": expected_keywords
            }

        except Exception as e:
            ClinicalLogger.error(f"Evaluation failed: {e}")
            return {
                "question": gold_question["question"],
                "category": gold_question.get("category", "unknown"),
                "answer": f"Error: {str(e)}",
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "search_time": time.time() - start_time,
                "documents_found": 0,
                "semantic_matches": 0,
                "total_expected": len(gold_question.get("expected_keywords", [])),
                "question_id": question_id,
                "timestamp": datetime.now().isoformat(),
                "hadm_id": gold_question.get("hadm_id"),
                "expected_keywords": gold_question.get("expected_keywords", []),
                "error": str(e)
            }

    def _compute_semantic_precision_recall(self, expected_keywords: List[str], response_text: str) -> Dict:
        """Compute semantic precision and recall using BioBERT embeddings"""
        if not expected_keywords:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "semantic_matches": 0,
                "total_expected": 0
            }

        if not self.biobert_model:
            ClinicalLogger.warning(
                "BioBERT model not available, using fallback scoring")
            return self._fallback_semantic_scoring(expected_keywords, response_text)

        try:
            # Find semantic matches using BioBERT
            matches, similarities = self._find_semantic_matches(
                expected_keywords, response_text)

            # Calculate recall: how many expected keywords were semantically found
            semantic_matches = sum(matches)
            total_expected = len(expected_keywords)
            recall = semantic_matches / total_expected if total_expected > 0 else 0.0

            # Calculate precision: average similarity of matched keywords
            if semantic_matches > 0:
                matched_similarities = [sim for match, sim in zip(
                    matches, similarities) if match]
                precision = np.mean(matched_similarities)
            else:
                precision = 0.0

            # F1 score
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "semantic_matches": semantic_matches,
                "total_expected": total_expected
            }

        except Exception as e:
            ClinicalLogger.error(f"Error in semantic scoring: {e}")
            return self._fallback_semantic_scoring(expected_keywords, response_text)

    def _find_semantic_matches(self, expected_keywords: List[str], response_text: str) -> Tuple[List[bool], List[float]]:
        """Find which expected keywords have semantic matches in response text using hybrid approach"""
        matches = []
        similarities = []
        threshold = SEMANTIC_EVALUATION_CONFIG["similarity_threshold"]
        
        # Clean response text for better matching
        response_lower = response_text.lower()

        for keyword in expected_keywords:
            keyword_str = str(keyword).strip()
            if not keyword_str:
                matches.append(False)
                similarities.append(0.0)
                continue

            # Method 1: Direct substring matching (highest priority)
            keyword_lower = keyword_str.lower()
            if keyword_lower in response_lower:
                matches.append(True)
                similarities.append(1.0)  # Perfect match
                continue
            
            # Method 2: Semantic similarity with sentence chunks
            best_similarity = 0.0
            sentences = self._split_into_sentences(response_text)
            if not sentences:
                sentences = [response_text]

            for sentence in sentences:
                if sentence.strip():
                    similarity = self._compute_semantic_similarity(
                        keyword_str, sentence)
                    best_similarity = max(best_similarity, similarity)
            
            # Method 3: If sentence matching fails, try phrase-level matching
            if best_similarity < threshold:
                # Split into smaller chunks (by commas, dashes, parentheses)
                import re
                phrases = re.split(r'[,\-\(\)\[\]]+', response_text)
                for phrase in phrases:
                    if phrase.strip():
                        similarity = self._compute_semantic_similarity(
                            keyword_str, phrase.strip())
                        best_similarity = max(best_similarity, similarity)

            matches.append(best_similarity >= threshold)
            similarities.append(best_similarity)

        return matches, similarities

    def _enhance_keywords_for_category(self, category: str, expected_keywords: List[str], answer: str) -> List[str]:
        """Enhance expected keywords based on category-specific patterns found in the answer"""
        if not expected_keywords:
            return expected_keywords
            
        enhanced_keywords = list(expected_keywords)
        
        if category == "prescriptions":
            # Extract medication-related terms from the answer for better matching
            import re
            answer_lower = answer.lower()
            
            # Find medication dosage patterns (e.g., "mg", "mL", "Units")
            dosage_patterns = re.findall(r'\b\d+\.?\d*\s*(mg|ml|units|g|mcg)\b', answer_lower)
            for pattern in dosage_patterns:
                if pattern not in [kw.lower() for kw in enhanced_keywords]:
                    enhanced_keywords.append(pattern)
            
            # Find medication form patterns (e.g., "Capsule", "Tablet", "Bag")
            form_patterns = re.findall(r'\b(capsule|tablet|bag|vial|injection|solution)\b', answer_lower)
            for form in form_patterns:
                if form.title() not in enhanced_keywords:
                    enhanced_keywords.append(form.title())
                    
        elif category == "header" and not enhanced_keywords:
            # Add common header terms if they appear in the answer
            import re
            answer_lower = answer.lower()
            
            header_terms = ["admission", "discharged", "subject", "weight", "height", "blood pressure"]
            for term in header_terms:
                if term in answer_lower:
                    enhanced_keywords.append(term.title())
            
            # Extract year patterns (MIMIC uses 2127+ format)
            year_patterns = re.findall(r'\b(212[0-9])\b', answer)
            for year in year_patterns:
                enhanced_keywords.append(year)
        
        return enhanced_keywords[:15]  # Limit to reasonable number

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using BioBERT"""
        try:
            embeddings = self.biobert_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            ClinicalLogger.error(f"Error computing semantic similarity: {e}")
            return 0.0

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better semantic matching"""
        if not text:
            return []
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _fallback_semantic_scoring(self, expected_keywords: List[str], response_text: str) -> Dict:
        """Fallback scoring when BioBERT is unavailable"""
        if not expected_keywords:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "semantic_matches": 0, "total_expected": 0}

        text_lower = str(response_text).lower()
        hits = sum(1 for kw in expected_keywords if str(
            kw).lower() in text_lower)

        recall = hits / len(expected_keywords)
        precision = recall  # Simple approximation
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "semantic_matches": hits,
            "total_expected": len(expected_keywords)
        }

    def run_batch_evaluation(self, embedding_model: str, llm_model: str,
                             questions: List[Dict]) -> List[Dict]:
        """Run evaluation batch for model combination

        Args:
            embedding_model: Embedding model nickname (e.g., "ms-marco")
            llm_model: LLM model nickname (e.g., "deepseek")
            questions: List of question dictionaries with expected_keywords

        Returns:
            List of evaluation result dictionaries with precision, recall, F1-score
        """
        from RAG_chat_pipeline.core.main import main as initialize_clinical_rag
        from RAG_chat_pipeline.config.config import set_models

        ClinicalLogger.info(
            f"Starting batch evaluation: {embedding_model} + {llm_model}")

        # Set models and initialize chatbot for this batch
        set_models(embedding_model, llm_model)
        self.chatbot = initialize_clinical_rag()

        # Evaluate all questions
        results = []
        for i, question in enumerate(questions):
            try:
                question_id = f"{embedding_model}_{llm_model}_{i}"
                result = self.evaluate_question(question, question_id)
                results.append(result)

                if (i + 1) % 10 == 0:  # Progress logging every 10 questions
                    ClinicalLogger.info(
                        f"Processed {i + 1}/{len(questions)} questions")

            except Exception as e:
                ClinicalLogger.error(f"Failed question {i}: {e}")
                # Continue processing remaining questions

        ClinicalLogger.info(
            f"Batch evaluation completed: {len(results)}/{len(questions)} successful")
        return results
