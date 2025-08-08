"""Streamlined RAG evaluation framework focused on practical assessment"""
import re
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from RAG_chat_pipeline.utils.data_provider import get_sample_data
from RAG_chat_pipeline.core.clinical_rag import ClinicalRAGBot
from .gold_questions import generate_gold_questions_from_data
from .retrieval_metrics import RetrievalMetricsEvaluator, format_retrieval_metrics_summary
from .visualization import EvaluationVisualizer
from .reporting import EvaluationReporter
from .dashboard import EvaluationDashboard
from RAG_chat_pipeline.config.config import *


class ClinicalRAGEvaluator:
    """Configuration-driven RAG evaluator using config.py parameters"""

    def __init__(self, chatbot: ClinicalRAGBot):
        self.chatbot = chatbot
        self.patient_data = get_sample_data()
        self.results = []

        # Initialize retrieval metrics evaluator (will be set up when questions are available)
        self.retrieval_metrics = None

        # Results storage for comprehensive reporting
        self.detailed_results = []
        self.evaluation_metadata = {
            "start_time": None,
            "end_time": None,
            "model_config": {
                "embedder": getattr(chatbot, 'embedder_name', 'unknown'),
                "llm": getattr(chatbot, 'llm_name', 'unknown')
            }
        }

    def evaluate_question(self, gold_question: Dict, question_id: str = None) -> Dict:
        """Evaluate a single question with enhanced weighted scoring"""
        result = {
            "question": gold_question["question"],
            "category": gold_question["category"],
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id
        }

        try:
            # Get chatbot response
            hadm_id = self._clean_hadm_id(gold_question.get("hadm_id"))

            # Enhanced prompt for better responses
            enhanced_question = self._enhance_prompt(
                gold_question["question"], gold_question["category"])

            response = self.chatbot.ask_question(
                question=enhanced_question,
                hadm_id=hadm_id,
                k=5
            )

            # Apply document filtering if too many docs retrieved
            source_docs = response.get("source_documents", [])
            if len(source_docs) > 10:
                filtered_docs = self.filter_and_rank_documents(
                    source_docs, gold_question["question"], top_k=10
                )
                response["source_documents"] = filtered_docs
                response["documents_found"] = len(filtered_docs)

            # Enhanced scoring with new metrics
            factual_score = self.evaluate_factual_accuracy(
                gold_question, response)
            behavior_score = self.evaluate_behavior(gold_question, response)
            performance_score = self.evaluate_performance(response)

            # New enhanced metrics
            context_relevance = self.evaluate_context_relevance(
                gold_question["question"], source_docs
            )

            # Extract expected entities for completeness check
            expected_entities = self._get_expected_entities(gold_question)
            completeness_score = self.evaluate_completeness(
                response.get("answer", ""), expected_entities
            )

            # Semantic similarity with medical keywords
            medical_keywords = self._get_medical_keywords_for_category(
                gold_question["category"])
            semantic_score = self.evaluate_semantic_similarity(
                response.get("answer", ""), medical_keywords
            )

            # Enhanced weighted overall score
            overall_score = (
                factual_score * EVALUATION_SCORING_WEIGHTS["factual_accuracy"] +
                behavior_score * EVALUATION_SCORING_WEIGHTS["behavior"] +
                performance_score * EVALUATION_SCORING_WEIGHTS["performance"] +
                context_relevance * 0.15 +  # Context relevance weight
                completeness_score * 0.1 +   # Completeness weight
                semantic_score * 0.1         # Semantic similarity weight
            )

            # Normalize the score since we added new weights
            overall_score = min(1.0, overall_score / 1.35)

            # Enhanced pass/fail determination with grades
            pass_grade, grade_value = self.get_pass_grade(
                overall_score, gold_question["category"])
            passed = grade_value >= 0

            # Add retrieval metrics using existing RetrievalMetricsEvaluator
            retrieval_metrics = {}
            if self.retrieval_metrics:
                try:
                    retrieval_metrics = self.retrieval_metrics.evaluate_retrieval_for_question(
                        question_id or f"q_{len(self.results)}",
                        source_docs
                    )
                except Exception as e:
                    print(
                        f"⚠️ Warning: Failed to evaluate retrieval metrics: {e}")
                    retrieval_metrics = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "error": str(e)
                    }

            result.update({
                "response": response.get("answer", ""),
                "enhanced_question": enhanced_question,
                "factual_accuracy_score": factual_score,
                "behavior_score": behavior_score,
                "performance_score": performance_score,
                "context_relevance_score": context_relevance,
                "completeness_score": completeness_score,
                "semantic_similarity_score": semantic_score,
                "overall_score": overall_score,
                "pass_threshold": self.get_pass_threshold(gold_question["category"]),
                "pass_grade": pass_grade,
                "grade_value": grade_value,
                "passed": passed,
                "search_time": response.get("search_time", 0),
                "documents_found": response.get("documents_found", 0),
                "documents_filtered": len(source_docs) > 10,
                "retrieval_metrics": retrieval_metrics,
                "extracted_entities": self.extract_medical_entities(response.get("answer", "")),
                "source_documents": source_docs,
                "error": None
            })

        except Exception as e:
            result.update({
                "error": str(e),
                "passed": False,
                "overall_score": 0.0,
                "response": "",
                "pass_grade": "fail",
                "grade_value": -1,
                "retrieval_metrics": {}
            })

        self.results.append(result)
        self.detailed_results.append(result)
        return result

    def evaluate_factual_accuracy(self, gold_question: Dict, response: Dict) -> float:
        """Evaluate factual accuracy against patient data"""
        answer = response.get("answer", "").lower()
        category = gold_question.get("category", "")

        # Handle special cases first using config patterns
        if gold_question.get("should_return_no_data"):
            return 1.0 if any(phrase in answer for phrase in EVALUATION_NO_DATA_PATTERNS) else 0.0

        # Get patient facts for validation
        hadm_id = self._clean_hadm_id(
            gold_question.get("hadm_id"))  # Fixed field name
        if not hadm_id:
            return 0.5  # Cannot validate without patient ID

        patient_facts = self._get_patient_facts(hadm_id)

        # Category-specific validation based on ACTUAL categories from gold questions
        if category == "header":
            return self._validate_header_answer(answer, patient_facts, gold_question)
        elif category == "diagnoses":
            return self._validate_diagnoses_answer(answer, patient_facts, gold_question)
        elif category == "procedures":
            return self._validate_procedures_answer(answer, patient_facts, gold_question)
        elif category == "labs":
            return self._validate_labs_answer(answer, patient_facts, gold_question)
        elif category == "microbiology":
            return self._validate_microbiology_answer(answer, patient_facts, gold_question)
        elif category == "prescriptions":
            return self._validate_prescriptions_answer(answer, patient_facts, gold_question)
        elif category == "comprehensive":
            return self._validate_comprehensive_answer(answer, patient_facts, gold_question)
        else:
            return self._basic_medical_score(answer, patient_facts)

    def evaluate_behavior(self, gold_question: Dict, response: Dict) -> float:
        """Evaluate if system behaves appropriately using config parameters"""
        answer = response.get("answer", "").lower()

        # Check for inappropriate responses
        if any(phrase in answer for phrase in EVALUATION_INAPPROPRIATE_RESPONSES["phrases"]) and not gold_question.get("should_return_no_data"):
            return EVALUATION_INAPPROPRIATE_RESPONSES["penalty_score"]

        # Check for good response indicators
        if any(indicator in answer for indicator in EVALUATION_GOOD_RESPONSE_INDICATORS["source_citation"]):
            return EVALUATION_GOOD_RESPONSE_INDICATORS["bonus_score"]

        # Check response length appropriateness
        if len(answer) < EVALUATION_RESPONSE_LENGTH["too_short"]:
            return EVALUATION_RESPONSE_LENGTH["short_penalty"]
        elif len(answer) > EVALUATION_RESPONSE_LENGTH["too_long"]:
            return EVALUATION_RESPONSE_LENGTH["long_penalty"]

        return 0.8  # Default reasonable score

    def evaluate_performance(self, response: Dict) -> float:
        """Evaluate system performance metrics using config thresholds"""
        search_time = response.get("search_time", 0)
        docs_found = response.get("documents_found", 0)

        score = 1.0
        if search_time > EVALUATION_PERFORMANCE_THRESHOLDS["slow_search_time"]:
            score -= EVALUATION_PERFORMANCE_THRESHOLDS["slow_penalty"]
        elif search_time > EVALUATION_PERFORMANCE_THRESHOLDS["moderate_search_time"]:
            score -= EVALUATION_PERFORMANCE_THRESHOLDS["moderate_penalty"]

        if docs_found == 0:
            score -= EVALUATION_PERFORMANCE_THRESHOLDS["no_docs_penalty"]

        return max(0.0, score)

    def _clean_hadm_id(self, hadm_id) -> int:
        """Clean and validate HADM ID"""
        if hadm_id is None or str(hadm_id) == "abc123":
            return None
        try:
            hadm_id = int(hadm_id)
            return hadm_id if hadm_id > 0 else None
        except (ValueError, TypeError):
            return None

    def _get_patient_facts(self, hadm_id: int) -> Dict:
        """Extract patient facts from dataset"""
        facts = {"diagnoses": [], "medications": [],
                 "lab_tests": [], "procedures": []}

        if not self.patient_data:
            return facts

        try:
            # Access the actual data structure from get_sample_data()
            link_tables = self.patient_data.get("link_tables", {})

            # Get diagnoses from diagnoses_icd table
            if "diagnoses_icd" in link_tables:
                df = link_tables["diagnoses_icd"]
                if 'hadm_id' in df.columns:
                    patient_rows = df[df['hadm_id'] == hadm_id]
                    if 'long_title' in df.columns:
                        facts["diagnoses"].extend(
                            patient_rows['long_title'].dropna().tolist())

            # Get medications from prescriptions table
            if "prescriptions" in link_tables:
                df = link_tables["prescriptions"]
                if 'hadm_id' in df.columns:
                    patient_rows = df[df['hadm_id'] == hadm_id]
                    if 'drug' in df.columns:
                        facts["medications"].extend(
                            patient_rows['drug'].dropna().tolist())

            # Get lab tests from labevents table
            if "labevents" in link_tables:
                df = link_tables["labevents"]
                if 'hadm_id' in df.columns:
                    patient_rows = df[df['hadm_id'] == hadm_id]
                    if 'label' in df.columns:
                        facts["lab_tests"].extend(
                            patient_rows['label'].dropna().tolist())

            # Get procedures from procedures_icd table
            if "procedures_icd" in link_tables:
                df = link_tables["procedures_icd"]
                if 'hadm_id' in df.columns:
                    patient_rows = df[df['hadm_id'] == hadm_id]
                    if 'long_title' in df.columns:
                        facts["procedures"].extend(
                            patient_rows['long_title'].dropna().tolist())

        except Exception as e:
            print(f"Warning: Error extracting patient facts: {e}")
            pass  # Return empty facts on error

        return facts

    def evaluate_semantic_similarity(self, generated_answer: str, expected_keywords: List[str]) -> float:
        """Evaluate semantic similarity using clinical embeddings"""
        if not expected_keywords or not generated_answer.strip():
            return 0.0

        try:
            # Get embeddings for generated answer
            gen_embedding = self.chatbot.clinical_emb.embed_query(
                generated_answer)

            # Get embeddings for expected keywords (combined)
            expected_text = " ".join(expected_keywords)
            exp_embedding = self.chatbot.clinical_emb.embed_query(
                expected_text)

            # Calculate cosine similarity
            similarity = cosine_similarity(
                [gen_embedding], [exp_embedding])[0][0]
            return float(similarity)

        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            return 0.0

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        entities = {
            "medications": [],
            "lab_values": [],
            "procedures": [],
            "diagnoses": [],
            "dates": []
        }

        # Extract lab values with units
        lab_pattern = r'(\d+\.?\d*)\s*(mg/dL|mmol/L|U/L|mEq/L|ng/mL|%)'
        lab_matches = re.findall(lab_pattern, text, re.IGNORECASE)
        entities["lab_values"] = [
            f"{value} {unit}" for value, unit in lab_matches]

        # Extract dates
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{1,2}/\d{1,2}/\d{2,4}'
        ]
        for pattern in date_patterns:
            entities["dates"].extend(re.findall(pattern, text))

        # Extract ICD codes
        icd_pattern = r'\b[A-Z]\d{2}\.?\d*\b'
        entities["diagnoses"].extend(re.findall(icd_pattern, text))

        # Extract medication dosages
        med_pattern = r'\b\d+\s*mg\b|\b\d+\s*mcg\b|\b\d+\s*units?\b'
        entities["medications"].extend(
            re.findall(med_pattern, text, re.IGNORECASE))

        return entities

    def evaluate_completeness(self, answer: str, expected_entities: Dict[str, List[str]]) -> float:
        """Evaluate answer completeness based on expected entities"""
        if not expected_entities:
            return 1.0

        found_entities = self.extract_medical_entities(answer)
        total_expected = sum(len(entities)
                             for entities in expected_entities.values())

        if total_expected == 0:
            return 1.0

        total_found = 0
        for category, expected_list in expected_entities.items():
            found_list = found_entities.get(category, [])
            # Check intersection
            matches = len(set(str(e).lower() for e in expected_list).intersection(
                set(str(f).lower() for f in found_list)))
            total_found += matches

        completeness = min(1.0, total_found / total_expected)
        return completeness

    def evaluate_context_relevance(self, question: str, retrieved_docs: List) -> float:
        """Evaluate relevance of retrieved documents to the question"""
        if not retrieved_docs:
            return 0.0

        try:
            question_embedding = self.chatbot.clinical_emb.embed_query(
                question)
            doc_scores = []

            # Limit to top 10 docs for efficiency
            for doc in retrieved_docs[:10]:
                doc_text = getattr(doc, 'page_content', str(doc))
                doc_embedding = self.chatbot.clinical_emb.embed_query(
                    doc_text[:500])  # Limit text length
                relevance = cosine_similarity(
                    [question_embedding], [doc_embedding])[0][0]
                doc_scores.append(relevance)

            return float(np.mean(doc_scores)) if doc_scores else 0.0

        except Exception as e:
            print(f"Warning: Context relevance calculation failed: {e}")
            return 0.0

    def filter_and_rank_documents(self, docs: List, question: str, top_k: int = 5) -> List:
        """Filter and rank documents by relevance to question"""
        if not docs or len(docs) <= top_k:
            return docs

        try:
            question_embedding = self.chatbot.clinical_emb.embed_query(
                question)
            doc_scores = []

            for i, doc in enumerate(docs):
                doc_text = getattr(doc, 'page_content', str(doc))
                doc_embedding = self.chatbot.clinical_emb.embed_query(
                    doc_text[:500])
                relevance = cosine_similarity(
                    [question_embedding], [doc_embedding])[0][0]
                doc_scores.append((i, relevance, doc))

            # Sort by relevance score and return top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return [doc for _, _, doc in doc_scores[:top_k]]

        except Exception as e:
            print(f"Warning: Document filtering failed: {e}")
            return docs[:top_k]

    def get_pass_grade(self, score: float, category: str) -> Tuple[str, int]:
        """Get graded pass level instead of binary pass/fail"""
        threshold = self.get_pass_threshold(category)

        if score >= threshold + 0.2:
            return "excellent", 2
        elif score >= threshold:
            return "pass", 1
        elif score >= threshold - 0.1:
            return "borderline", 0
        else:
            return "fail", -1

    def _enhance_prompt(self, question: str, category: str) -> str:
        """Enhance the prompt based on question category for better responses"""
        category_prompts = {
            "header": f"""
            Based on the patient admission records, answer this question about admission details PRECISELY and CONCISELY.
            Focus specifically on admission metadata (dates, types, discharge information).
            
            Question: {question}
            
            Please provide a clear, factual answer based on the admission records.
            """,

            "diagnoses": f"""
            Based on the patient's diagnostic records, answer this question about medical diagnoses.
            Include relevant ICD codes when available and focus on clinical accuracy.
            
            Question: {question}
            
            Provide specific diagnostic information with codes when possible.
            """,

            "labs": f"""
            Based on the laboratory results, answer this question about lab values.
            Include specific values, units, and normal ranges when available.
            
            Question: {question}
            
            Provide precise lab values with appropriate units.
            """,

            "prescriptions": f"""
            Based on the medication records, answer this question about prescriptions.
            Include drug names, dosages, frequencies, and administration routes when available.
            
            Question: {question}
            
            Provide specific medication information with dosages and timing.
            """,

            "procedures": f"""
            Based on the procedure records, answer this question about medical procedures.
            Include procedure names, dates, and relevant codes when available.
            
            Question: {question}
            
            Provide specific procedure information with codes when possible.
            """,

            "microbiology": f"""
            Based on the microbiology reports, answer this question about cultures and infections.
            Include organism names, specimen types, and sensitivity results when available.
            
            Question: {question}
            
            Provide specific microbiology findings with organism details.
            """,

            "comprehensive": f"""
            Based on all available patient records, provide a comprehensive answer to this question.
            Include information from multiple record types (diagnoses, procedures, medications, labs).
            
            Question: {question}
            
            Provide a thorough summary covering relevant clinical aspects.
            """
        }

        return category_prompts.get(category, question)

    def _get_expected_entities(self, gold_question: Dict) -> Dict[str, List[str]]:
        """Extract expected entities based on question category and patient data"""
        category = gold_question.get("category", "")
        hadm_id = self._clean_hadm_id(gold_question.get("hadm_id"))

        if not hadm_id:
            return {}

        patient_facts = self._get_patient_facts(hadm_id)

        entity_mapping = {
            "diagnoses": {"diagnoses": patient_facts.get("diagnoses", [])},
            "prescriptions": {"medications": patient_facts.get("medications", [])},
            "labs": {"lab_values": patient_facts.get("lab_tests", [])},
            "procedures": {"procedures": patient_facts.get("procedures", [])},
            "microbiology": {"diagnoses": patient_facts.get("diagnoses", [])},
            "comprehensive": {
                "diagnoses": patient_facts.get("diagnoses", []),
                "medications": patient_facts.get("medications", []),
                "procedures": patient_facts.get("procedures", [])
            }
        }

        return entity_mapping.get(category, {})

    def _get_medical_keywords_for_category(self, category: str) -> List[str]:
        """Get relevant medical keywords for semantic similarity by category"""
        keyword_mapping = {
            "header": ["admission", "discharge", "hospital", "patient", "type", "emergency"],
            "diagnoses": ["diagnosis", "condition", "disease", "disorder", "syndrome", "icd"],
            "labs": ["laboratory", "test", "value", "result", "normal", "abnormal", "level"],
            "prescriptions": ["medication", "drug", "prescription", "dose", "mg", "administration"],
            "procedures": ["procedure", "surgery", "operation", "intervention", "treatment"],
            "microbiology": ["culture", "organism", "bacteria", "infection", "sensitivity", "specimen"],
            "comprehensive": ["patient", "medical", "clinical", "treatment", "diagnosis", "care"]
        }

        return keyword_mapping.get(category, [])

    def get_pass_threshold(self, category: str) -> float:
        """Get pass threshold based on question category from config"""
        return EVALUATION_PASS_THRESHOLDS.get(category, EVALUATION_PASS_THRESHOLDS["default"])

    def _process_questions(self, questions: List[Dict], show_detailed: bool = False) -> Dict:
        """Core method to process questions with optional detailed output"""
        # Initialize retrieval metrics evaluator with current questions
        if self.retrieval_metrics is None:
            self.retrieval_metrics = RetrievalMetricsEvaluator()
            gold_standard = self.retrieval_metrics.build_gold_standard_from_questions(
                questions)
            self.retrieval_metrics.gold_standard = gold_standard

        detailed_results = []
        category_results = {}
        passed_count = 0
        retrieval_results = {}  # For batch retrieval evaluation

        for i, question in enumerate(questions):
            question_id = f"q_{i}"

            if show_detailed:
                print(f"\nQUESTION {i+1}: {question['category']}")
                print(
                    f"Q: {question['question'][:EVALUATION_OUTPUT_CONFIG['question_preview_length']]}...")
            else:
                print(
                    f"[{i+1}/{len(questions)}] {question['category']}: {question['question'][:60]}...")

            # Evaluate question with question_id for retrieval metrics
            result = self.evaluate_question(question, question_id)
            detailed_results.append(result)

            # Store for batch retrieval evaluation
            if result.get("source_documents"):
                retrieval_results[question_id] = {
                    "source_documents": result["source_documents"]
                }

            if show_detailed:
                print(
                    f"A: {result['response'][:EVALUATION_OUTPUT_CONFIG['answer_preview_length']]}...")
                print(f"Scores: Factual={result.get('factual_accuracy_score', 0):.2f}, "
                      f"Behavior={result.get('behavior_score', 0):.2f}, "
                      f"Performance={result.get('performance_score', 0):.2f}")

                # Show retrieval metrics if available
                ret_metrics = result.get('retrieval_metrics', {})
                if ret_metrics and 'precision' in ret_metrics:
                    print(f"Retrieval: P={ret_metrics['precision']:.2f}, "
                          f"R={ret_metrics['recall']:.2f}, "
                          f"F1={ret_metrics['f1']:.2f}")

                print(
                    f"Overall: {result['overall_score']:.2f} ({'PASS' if result['passed'] else 'FAIL'})")
                print("-" * EVALUATION_OUTPUT_CONFIG["separator_length"])

            # Track statistics
            if result['passed']:
                passed_count += 1

            # Group by category
            category = result["category"]
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)

        # Calculate batch retrieval metrics
        batch_retrieval_metrics = {}
        if retrieval_results and self.retrieval_metrics:
            batch_retrieval_metrics = self.retrieval_metrics.evaluate_retrieval_batch(
                retrieval_results)

        return {
            "detailed_results": detailed_results,
            "category_results": category_results,
            "passed_count": passed_count,
            "total_questions": len(questions),
            "batch_retrieval_metrics": batch_retrieval_metrics
        }

    def _calculate_summary(self, processed_results: Dict) -> Dict:
        """Calculate summary statistics from processed results"""
        detailed_results = processed_results["detailed_results"]
        category_results = processed_results["category_results"]
        passed_count = processed_results["passed_count"]
        total_questions = processed_results["total_questions"]
        batch_retrieval_metrics = processed_results.get(
            "batch_retrieval_metrics", {})

        summary = {
            "total_questions": total_questions,
            "passed": passed_count,
            "pass_rate": passed_count / total_questions if total_questions else 0,
            "average_score": sum(r["overall_score"] for r in detailed_results) / len(detailed_results) if detailed_results else 0,
            "category_breakdown": {},
            "retrieval_metrics": batch_retrieval_metrics
        }

        for category, results in category_results.items():
            summary["category_breakdown"][category] = {
                "count": len(results),
                "pass_rate": sum(r["passed"] for r in results) / len(results),
                "average_score": sum(r["overall_score"] for r in results) / len(results)
            }

        return summary

    def _print_summary(self, summary: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*50}")
        print(
            f"SUMMARY: {summary['passed']}/{summary['total_questions']} passed ({summary['pass_rate']:.1%})")
        print(f"Average Score: {summary['average_score']:.2f}")

        for category, stats in summary["category_breakdown"].items():
            print(
                f"  {category}: {stats['pass_rate']:.1%} ({stats['count']} questions)")

        # Print retrieval metrics if available
        retrieval_metrics = summary.get("retrieval_metrics", {})
        if retrieval_metrics.get("questions_evaluated", 0) > 0:
            print(format_retrieval_metrics_summary(retrieval_metrics))

    def run_evaluation(self, gold_questions: List[Dict]) -> Dict:
        """Run evaluation on all questions"""
        print(f"\nEvaluating {len(gold_questions)} questions...")
        print("="*50)

        # Process questions without detailed output
        processed_results = self._process_questions(
            gold_questions, show_detailed=False)

        # Calculate and display summary
        summary = self._calculate_summary(processed_results)
        self._print_summary(summary)

        return {
            "summary": summary,
            "category_results": processed_results["category_results"],
            "detailed_results": processed_results["detailed_results"]
        }

    def _validate_header_answer(self, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Validate header/admission info questions using config keywords"""
        question = gold_question["question"].lower()

        # Check for admission type, dates, expire flag
        if "type" in question:
            return 1.0 if any(keyword in answer.lower() for keyword in EVALUATION_MEDICAL_KEYWORDS["header"]["admission_type"]) else 0.3
        elif "when" in question or "admitted" in question or "discharged" in question:
            # Look for date/time information using config patterns
            return 1.0 if re.search(EVALUATION_MEDICAL_KEYWORDS["header"]["date_pattern"], answer) else 0.4
        elif "expire" in question:
            return 1.0 if any(keyword in answer.lower() for keyword in EVALUATION_MEDICAL_KEYWORDS["header"]["expire"]) else 0.3
        else:
            return self._basic_medical_score(answer, facts)

    def _validate_diagnoses_answer(self, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Validate diagnosis-related questions using config keywords"""
        return self._validate_structured_data(
            answer,
            EVALUATION_MEDICAL_KEYWORDS["diagnoses"]["primary"],
            EVALUATION_MEDICAL_KEYWORDS["diagnoses"]["code_pattern"]
        )

    def _validate_procedures_answer(self, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Validate procedure-related questions using config keywords"""
        return self._validate_structured_data(
            answer,
            EVALUATION_MEDICAL_KEYWORDS["procedures"]["primary"],
            EVALUATION_MEDICAL_KEYWORDS["procedures"]["code_pattern"]
        )

    def _validate_labs_answer(self, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Validate laboratory-related questions using config keywords"""
        lab_keywords = EVALUATION_MEDICAL_KEYWORDS["labs"]["primary"]
        unit_keywords = EVALUATION_MEDICAL_KEYWORDS["labs"]["units"]
        value_pattern = EVALUATION_MEDICAL_KEYWORDS["labs"]["value_pattern"]

        lab_terms_found = sum(
            1 for keyword in lab_keywords if keyword in answer.lower())
        values_found = len(re.findall(value_pattern, answer))
        units_found = sum(
            1 for unit in unit_keywords if unit in answer.lower())

        score = 0.0
        if lab_terms_found > 0:
            score += EVALUATION_CATEGORY_SCORING_WEIGHTS["labs"]["lab_terms"]
        if values_found > 0:
            score += EVALUATION_CATEGORY_SCORING_WEIGHTS["labs"]["values"]
        if units_found > 0:
            score += EVALUATION_CATEGORY_SCORING_WEIGHTS["labs"]["units"]
        return min(1.0, score)

    def _validate_microbiology_answer(self, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Validate microbiology-related questions using config keywords"""
        return self._validate_structured_data(
            answer,
            EVALUATION_MEDICAL_KEYWORDS["microbiology"]["primary"],
            None,
            EVALUATION_MEDICAL_KEYWORDS["microbiology"]["specimen_types"]
        )

    def _validate_prescriptions_answer(self, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Validate prescription/medication questions using config keywords"""
        med_keywords = EVALUATION_MEDICAL_KEYWORDS["prescriptions"]["primary"]
        timing_keywords = EVALUATION_MEDICAL_KEYWORDS["prescriptions"]["timing"]
        dosage_pattern = EVALUATION_MEDICAL_KEYWORDS["prescriptions"]["dosage_pattern"]

        med_terms_found = sum(
            1 for keyword in med_keywords if keyword in answer.lower())
        dosages_found = len(re.findall(dosage_pattern, answer.lower()))
        timing_found = sum(
            1 for timing in timing_keywords if timing in answer.lower())

        score = 0.0
        if med_terms_found > 0:
            score += EVALUATION_CATEGORY_SCORING_WEIGHTS["prescriptions"]["med_terms"]
        if dosages_found > 0:
            score += EVALUATION_CATEGORY_SCORING_WEIGHTS["prescriptions"]["dosages"]
        if timing_found > 0:
            score += EVALUATION_CATEGORY_SCORING_WEIGHTS["prescriptions"]["timing"]
        return min(1.0, score)

    def _validate_comprehensive_answer(self, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Validate comprehensive/summary questions using config keywords"""
        section_keywords = EVALUATION_MEDICAL_KEYWORDS["comprehensive"]["sections"]

        sections_covered = sum(1 for keywords in section_keywords.values()
                               if any(keyword in answer.lower() for keyword in keywords))

        score = sections_covered / len(section_keywords)
        if len(answer) > EVALUATION_RESPONSE_LENGTH["comprehensive_bonus_threshold"]:
            score += EVALUATION_RESPONSE_LENGTH["comprehensive_bonus"]
        return min(1.0, score)

    def _validate_structured_data(self, answer: str, keywords: list, code_pattern: str = None, additional_keywords: list = None) -> float:
        """Helper method to validate structured medical data using config weights"""
        score = 0.0

        # Check for domain keywords
        keywords_found = sum(
            1 for keyword in keywords if keyword in answer.lower())
        if keywords_found > 0:
            score += EVALUATION_STRUCTURED_DATA_SCORING["keywords_found_weight"]

        # Check for codes if pattern provided
        if code_pattern:
            codes_found = len(re.findall(code_pattern, answer))
            if codes_found > 0:
                score += EVALUATION_STRUCTURED_DATA_SCORING["codes_found_weight"]

        # Check for additional keywords if provided
        if additional_keywords:
            additional_found = sum(
                1 for keyword in additional_keywords if keyword in answer.lower())
            if additional_found > 0:
                score += EVALUATION_STRUCTURED_DATA_SCORING["additional_keywords_weight"]

        # Check for "no data" responses using config patterns
        if any(phrase in answer.lower() for phrase in EVALUATION_NO_DATA_PATTERNS):
            score = max(
                score, EVALUATION_STRUCTURED_DATA_SCORING["no_data_response_score"])

        return min(1.0, score)

    def _basic_medical_score(self, answer: str, facts: Dict) -> float:
        """Basic scoring based on medical term overlap using config parameters"""
        if not facts.get("diagnoses") and not facts.get("medications"):
            return 0.5

        all_medical_terms = " ".join(
            facts.get("diagnoses", []) + facts.get("medications", [])).lower()
        answer_words = set(answer.lower().split())
        medical_words = set(all_medical_terms.split())
        overlap = len(answer_words.intersection(medical_words))

        return min(
            EVALUATION_BASIC_MEDICAL_SCORING["max_score"],
            overlap / EVALUATION_BASIC_MEDICAL_SCORING["overlap_scale_factor"]
        )

    def run_short_evaluation(self, limit: int = None) -> Dict:
        """Run evaluation on a subset with detailed output"""
        if limit is None:
            limit = EVALUATION_DEFAULT_PARAMS["short_evaluation_limit"]

        all_questions = generate_gold_questions_from_data()
        if not all_questions:
            return {"error": "No questions generated"}

        test_questions = all_questions[:limit]
        print(
            f"\nSHORT EVALUATION: {len(test_questions)} questions with detailed output")
        print("="*EVALUATION_OUTPUT_CONFIG["long_separator_length"])

        # Process questions with detailed output
        processed_results = self._process_questions(
            test_questions, show_detailed=True)

        # Calculate and display summary
        summary = self._calculate_summary(processed_results)
        self._print_summary(summary)

        return {
            "summary": summary,
            "category_results": processed_results["category_results"],
            "detailed_results": processed_results["detailed_results"]
        }

    def run_quick_test(self, num_questions: int = None) -> Dict:
        """Quick test with just a few questions for debugging"""
        if num_questions is None:
            num_questions = EVALUATION_DEFAULT_PARAMS["quick_test_limit"]

        all_questions = generate_gold_questions_from_data()
        if not all_questions:
            return {"error": "No questions generated"}

        test_questions = all_questions[:num_questions]
        print(f"\nQUICK TEST: {len(test_questions)} questions")
        print("="*EVALUATION_OUTPUT_CONFIG["separator_length"])

        results = []
        for i, question in enumerate(test_questions):
            print(
                f"\n{i+1}. {question['question'][:EVALUATION_OUTPUT_CONFIG['question_preview_length']]}...")
            # Pass question_id for retrieval metrics
            result = self.evaluate_question(question, f"q_{i}")
            results.append(result)
            print(
                f"   -> {result['overall_score']:.2f} ({'PASS' if result['passed'] else 'FAIL'})")

        return {"summary": {"tested": len(test_questions)}, "results": results}

    def generate_comprehensive_report(self, results: Dict, output_dir: str = "evaluation_reports") -> Dict[str, str]:
        """Generate comprehensive evaluation report using modular components"""
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize modular components
        reporter = EvaluationReporter(output_path)
        visualizer = EvaluationVisualizer(output_path)
        dashboard_generator = EvaluationDashboard(output_path)

        # Generate reports and visualizations
        report_files = reporter.export_comprehensive_csv(results, timestamp)
        viz_files = visualizer.generate_all_visualizations(results, timestamp)

        # Create dashboard
        dashboard_file = dashboard_generator.create_performance_dashboard(
            results, report_files, viz_files, timestamp)

        all_files = {**report_files, **viz_files, "dashboard": dashboard_file}

        print(f"\n📊 Comprehensive report generated in: {output_path}")
        print("Generated files:")
        for file_type, file_path in all_files.items():
            if file_path:
                print(f"  - {file_type}: {Path(file_path).name}")

        return all_files

    def compare_evaluations(self, evaluation_results: List[Dict], labels: List[str],
                            output_dir: str = "evaluation_reports") -> str:
        """Compare multiple evaluation results side by side"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = output_path / \
            f"evaluation_comparison_{timestamp}.csv"

        comparison_data = []

        for i, (results, label) in enumerate(zip(evaluation_results, labels)):
            summary = results.get("summary", {})

            comparison_data.append({
                "Evaluation": label,
                "Pass_Rate": f"{summary.get('pass_rate', 0):.2%}",
                "Average_Score": f"{summary.get('average_score', 0):.3f}",
                "Total_Questions": summary.get('total_questions', 0),
                "Excellent_Count": len([r for r in results.get("detailed_results", []) if r.get("pass_grade") == "excellent"]),
                "Pass_Count": len([r for r in results.get("detailed_results", []) if r.get("pass_grade") == "pass"]),
                "Borderline_Count": len([r for r in results.get("detailed_results", []) if r.get("pass_grade") == "borderline"]),
                "Fail_Count": len([r for r in results.get("detailed_results", []) if r.get("pass_grade") == "fail"]),
                "Avg_Search_Time": f"{np.mean([r.get('search_time', 0) for r in results.get('detailed_results', [])]):.2f}s",
                "Avg_Documents_Found": f"{np.mean([r.get('documents_found', 0) for r in results.get('detailed_results', [])]):.1f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comparison_file, index=False)

        # Generate comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Pass rates comparison
        pass_rates = [float(row["Pass_Rate"].strip('%')) /
                      100 for row in comparison_data]
        ax1.bar(labels, pass_rates, color='lightblue', alpha=0.7)
        ax1.set_ylabel('Pass Rate')
        ax1.set_title('Pass Rate Comparison')
        ax1.set_ylim(0, 1)

        # Grade distribution stacked bar
        grades_data = np.array([[row["Excellent_Count"], row["Pass_Count"],
                               row["Borderline_Count"], row["Fail_Count"]]
                                for row in comparison_data])

        bottom = np.zeros(len(labels))
        colors = ['green', 'lightgreen', 'orange', 'red']
        grade_labels = ['Excellent', 'Pass', 'Borderline', 'Fail']

        for i, (color, label) in enumerate(zip(colors, grade_labels)):
            ax2.bar(labels, grades_data[:, i], bottom=bottom,
                    color=color, alpha=0.7, label=label)
            bottom += grades_data[:, i]

        ax2.set_ylabel('Count')
        ax2.set_title('Grade Distribution Comparison')
        ax2.legend()

        # Average scores comparison
        avg_scores = [float(row["Average_Score"]) for row in comparison_data]
        ax3.bar(labels, avg_scores, color='lightcoral', alpha=0.7)
        ax3.set_ylabel('Average Score')
        ax3.set_title('Average Score Comparison')
        ax3.set_ylim(0, 1)

        # Search time comparison
        search_times = [float(row["Avg_Search_Time"].rstrip('s'))
                        for row in comparison_data]
        ax4.bar(labels, search_times, color='lightsteelblue', alpha=0.7)
        ax4.set_ylabel('Search Time (seconds)')
        ax4.set_title('Average Search Time Comparison')

        plt.tight_layout()

        comparison_viz_file = output_path / \
            f"evaluation_comparison_{timestamp}.png"
        plt.savefig(comparison_viz_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Evaluation comparison saved: {comparison_file}")
        print(f"📈 Comparison visualization saved: {comparison_viz_file}")

        return str(comparison_file)


# Duplicate methods removed - functionality moved to modular components

def main():
    """Main function for running evaluations"""
    if len(sys.argv) < 2:
        print(
            "Usage: python rag_evaluator.py [full|short|quick|comprehensive]")
        print(f"  full          - Run complete evaluation")
        print(
            f"  short         - Run evaluation on {EVALUATION_DEFAULT_PARAMS['short_evaluation_limit']} questions with details")
        print(
            f"  quick         - Run quick test with {EVALUATION_DEFAULT_PARAMS['quick_test_limit']} questions")
        print(f"  comprehensive - Run full evaluation with comprehensive reporting and visualizations")
        return

    from RAG_chat_pipeline.core.main import initialize_clinical_rag

    print("Initializing RAG system...")
    chatbot = initialize_clinical_rag()
    evaluator = ClinicalRAGEvaluator(chatbot)
    mode = sys.argv[1].lower()

    if mode == "full":
        questions = generate_gold_questions_from_data()
        results = evaluator.run_evaluation(questions)

        # Save results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to evaluation_results.json")

    elif mode == "comprehensive":
        questions = generate_gold_questions_from_data()
        results = evaluator.run_evaluation_with_comprehensive_reporting(
            questions)

        # Save comprehensive results
        with open('comprehensive_evaluation_results.json', 'w') as f:
            json.dump({k: v for k, v in results.items()
                      if k != "report_files"}, f, indent=2)
        print(f"\nComprehensive results saved to comprehensive_evaluation_results.json")

    elif mode == "short":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = evaluator.run_short_evaluation(limit)

    elif mode == "quick":
        num = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = evaluator.run_quick_test(num)

    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
