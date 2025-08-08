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

from RAG_chat_pipeline.helper.data_loader import get_sample_data
from RAG_chat_pipeline.core.clinical_rag import ClinicalRAGBot
from .gold_questions import generate_gold_questions_from_data
from .retrieval_metrics import RetrievalMetricsEvaluator, format_retrieval_metrics_summary
from RAG_chat_pipeline.config.config import *


class ClinicalRAGEvaluator:
    """Configuration-driven RAG evaluator using config.py parameters"""

    def __init__(self, chatbot: ClinicalRAGBot):
        self.chatbot = chatbot
        self.patient_data = get_sample_data()
        self.results = []

        # Initialize retrieval metrics evaluator (will be populated when questions are available)
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

            # Add retrieval metrics if available
            retrieval_metrics = {}
            if self.retrieval_metrics and question_id:
                retrieval_metrics = self.retrieval_metrics.evaluate_retrieval_for_question(
                    question_id, source_docs
                )

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
        """Generate comprehensive evaluation report with visualizations and CSV exports"""

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}

        # 1. Export detailed results to CSV
        csv_file = output_path / f"detailed_results_{timestamp}.csv"
        df_results = pd.DataFrame(self.detailed_results)
        df_results.to_csv(csv_file, index=False)
        report_files["detailed_csv"] = str(csv_file)

        # 2. Create summary statistics CSV
        summary_file = output_path / f"summary_stats_{timestamp}.csv"
        summary_df = self._create_summary_dataframe(results)
        summary_df.to_csv(summary_file, index=False)
        report_files["summary_csv"] = str(summary_file)

        # 3. Generate visualizations
        viz_files = self._generate_visualizations(
            results, output_path, timestamp)
        report_files.update(viz_files)

        # 4. Create performance comparison table
        perf_file = output_path / f"performance_comparison_{timestamp}.csv"
        perf_df = self._create_performance_comparison(results)
        perf_df.to_csv(perf_file, index=False)
        report_files["performance_csv"] = str(perf_file)

        # 5. Generate category-wise analysis
        category_file = output_path / f"category_analysis_{timestamp}.csv"
        category_df = self._create_category_analysis(results)
        category_df.to_csv(category_file, index=False)
        report_files["category_csv"] = str(category_file)

        print(f"\n📊 Comprehensive report generated in: {output_path}")
        print("Generated files:")
        for file_type, file_path in report_files.items():
            print(f"  - {file_type}: {file_path}")

        return report_files

    def _create_summary_dataframe(self, results: Dict) -> pd.DataFrame:
        """Create summary statistics DataFrame"""
        summary = results.get("summary", {})
        category_breakdown = summary.get("category_breakdown", {})

        summary_data = []

        # Overall statistics
        summary_data.append({
            "Metric": "Overall Pass Rate",
            "Value": f"{summary.get('pass_rate', 0):.2%}",
            "Score": f"{summary.get('average_score', 0):.3f}"
        })

        summary_data.append({
            "Metric": "Total Questions",
            "Value": summary.get('total_questions', 0),
            "Score": "-"
        })

        # Category-wise statistics
        for category, stats in category_breakdown.items():
            summary_data.append({
                "Metric": f"{category.title()} Pass Rate",
                "Value": f"{stats.get('pass_rate', 0):.2%}",
                "Score": f"{stats.get('average_score', 0):.3f}"
            })

        return pd.DataFrame(summary_data)

    def _create_performance_comparison(self, results: Dict) -> pd.DataFrame:
        """Create performance comparison DataFrame"""
        detailed_results = results.get("detailed_results", [])

        performance_data = []
        for result in detailed_results:
            performance_data.append({
                "Question_ID": result.get("question_id", ""),
                "Category": result.get("category", ""),
                "Overall_Score": result.get("overall_score", 0),
                "Factual_Score": result.get("factual_accuracy_score", 0),
                "Behavior_Score": result.get("behavior_score", 0),
                "Performance_Score": result.get("performance_score", 0),
                "Context_Relevance": result.get("context_relevance_score", 0),
                "Completeness": result.get("completeness_score", 0),
                "Semantic_Similarity": result.get("semantic_similarity_score", 0),
                "Pass_Grade": result.get("pass_grade", "fail"),
                "Search_Time": result.get("search_time", 0),
                "Documents_Found": result.get("documents_found", 0),
                "Documents_Filtered": result.get("documents_filtered", False)
            })

        return pd.DataFrame(performance_data)

    def _create_category_analysis(self, results: Dict) -> pd.DataFrame:
        """Create category-wise analysis DataFrame"""
        category_results = results.get("category_results", {})

        analysis_data = []
        for category, cat_results in category_results.items():
            scores = [r.get("overall_score", 0) for r in cat_results]
            search_times = [r.get("search_time", 0) for r in cat_results]
            doc_counts = [r.get("documents_found", 0) for r in cat_results]

            analysis_data.append({
                "Category": category,
                "Question_Count": len(cat_results),
                "Pass_Rate": f"{sum(1 for r in cat_results if r.get('passed', False)) / len(cat_results):.2%}",
                "Avg_Score": f"{np.mean(scores):.3f}",
                "Score_StdDev": f"{np.std(scores):.3f}",
                "Min_Score": f"{np.min(scores):.3f}",
                "Max_Score": f"{np.max(scores):.3f}",
                "Avg_Search_Time": f"{np.mean(search_times):.2f}s",
                "Avg_Documents": f"{np.mean(doc_counts):.1f}",
                "Grade_Distribution": self._get_grade_distribution(cat_results)
            })

        return pd.DataFrame(analysis_data)

    def _get_grade_distribution(self, results: List[Dict]) -> str:
        """Get grade distribution for a category"""
        grades = [r.get("pass_grade", "fail") for r in results]
        grade_counts = pd.Series(grades).value_counts().to_dict()

        distribution = []
        for grade in ["excellent", "pass", "borderline", "fail"]:
            count = grade_counts.get(grade, 0)
            if count > 0:
                distribution.append(f"{grade}: {count}")

        return ", ".join(distribution)

    def _generate_visualizations(self, results: Dict, output_path: Path, timestamp: str) -> Dict[str, str]:
        """Generate various visualizations"""
        viz_files = {}

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Score Distribution Heatmap (Enhanced)
        fig, ax = plt.subplots(figsize=(12, 8))
        category_results = results.get("category_results", {})

        # Create score matrix for heatmap
        score_matrix = []
        categories = []
        metrics = ["Overall", "Factual", "Behavior",
                   "Performance", "Context", "Completeness", "Semantic"]

        for category, cat_results in category_results.items():
            categories.append(category)
            row = [
                np.mean([r.get("overall_score", 0) for r in cat_results]),
                np.mean([r.get("factual_accuracy_score", 0)
                        for r in cat_results]),
                np.mean([r.get("behavior_score", 0) for r in cat_results]),
                np.mean([r.get("performance_score", 0) for r in cat_results]),
                np.mean([r.get("context_relevance_score", 0)
                        for r in cat_results]),
                np.mean([r.get("completeness_score", 0) for r in cat_results]),
                np.mean([r.get("semantic_similarity_score", 0)
                        for r in cat_results])
            ]
            score_matrix.append(row)

        sns.heatmap(score_matrix,
                    xticklabels=metrics,
                    yticklabels=categories,
                    annot=True,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1,
                    fmt='.3f')

        plt.title("RAG Evaluation Score Heatmap by Category and Metric")
        plt.tight_layout()

        heatmap_file = output_path / f"score_heatmap_{timestamp}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files["heatmap"] = str(heatmap_file)

        # 2. Grade Distribution Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        all_grades = [r.get("pass_grade", "fail")
                      for r in results.get("detailed_results", [])]
        grade_counts = pd.Series(all_grades).value_counts()

        colors = {"excellent": "green", "pass": "lightgreen",
                  "borderline": "orange", "fail": "red"}
        grade_counts.plot(kind='bar',
                          color=[colors.get(x, 'gray')
                                 for x in grade_counts.index],
                          ax=ax)

        plt.title("Distribution of Pass Grades")
        plt.ylabel("Count")
        plt.xlabel("Grade")
        plt.xticks(rotation=45)
        plt.tight_layout()

        grades_file = output_path / f"grade_distribution_{timestamp}.png"
        plt.savefig(grades_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files["grades"] = str(grades_file)

        # 3. Search Time vs Score Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        detailed_results = results.get("detailed_results", [])

        scores = [r.get("overall_score", 0) for r in detailed_results]
        search_times = [r.get("search_time", 0) for r in detailed_results]
        categories = [r.get("category", "") for r in detailed_results]

        scatter = ax.scatter(search_times, scores, c=range(len(categories)),
                             alpha=0.6, cmap='tab10')

        ax.set_xlabel("Search Time (seconds)")
        ax.set_ylabel("Overall Score")
        ax.set_title("Search Time vs Overall Score")

        # Add trend line
        z = np.polyfit(search_times, scores, 1)
        p = np.poly1d(z)
        ax.plot(search_times, p(search_times), "r--", alpha=0.8)

        plt.tight_layout()

        scatter_file = output_path / f"time_vs_score_{timestamp}.png"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files["scatter"] = str(scatter_file)

        # 4. Document Count Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Documents found distribution
        doc_counts = [r.get("documents_found", 0) for r in detailed_results]
        ax1.hist(doc_counts, bins=20, alpha=0.7,
                 color='skyblue', edgecolor='black')
        ax1.set_xlabel("Documents Found")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Documents Found")

        # Documents found vs Score
        ax2.scatter(doc_counts, scores, alpha=0.6, color='coral')
        ax2.set_xlabel("Documents Found")
        ax2.set_ylabel("Overall Score")
        ax2.set_title("Documents Found vs Overall Score")

        # Add trend line
        if len(doc_counts) > 1:
            z = np.polyfit(doc_counts, scores, 1)
            p = np.poly1d(z)
            ax2.plot(doc_counts, p(doc_counts), "r--", alpha=0.8)

        plt.tight_layout()

        docs_file = output_path / f"document_analysis_{timestamp}.png"
        plt.savefig(docs_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files["documents"] = str(docs_file)

        return viz_files

    def create_performance_dashboard(self, results: Dict, output_dir: str = "evaluation_reports") -> str:
        """Create an HTML dashboard with all evaluation metrics"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = output_path / \
            f"performance_dashboard_{timestamp}.html"

        # Generate comprehensive report files
        report_files = self.generate_comprehensive_report(results, output_dir)

        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG Evaluation Dashboard - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metric-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px; border-radius: 8px; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .metric-label {{ font-size: 0.9em; color: #666; }}
                .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
                .chart-container {{ text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 8px; }}
                .chart-container img {{ max-width: 100%; height: auto; border-radius: 5px; }}
                .files-section {{ margin-top: 30px; }}
                .file-link {{ display: inline-block; margin: 5px; padding: 8px 15px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                .file-link:hover {{ background-color: #0056b3; }}
                .grade-colors {{ display: flex; justify-content: center; gap: 20px; margin: 20px 0; }}
                .grade-item {{ text-align: center; }}
                .grade-color {{ width: 20px; height: 20px; border-radius: 50%; margin: 0 auto 5px; }}
                .excellent {{ background-color: green; }}
                .pass {{ background-color: lightgreen; }}
                .borderline {{ background-color: orange; }}
                .fail {{ background-color: red; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Clinical RAG Evaluation Dashboard</h1>
                    <p>Generated on: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                    <div class="metric-card">
                        <div class="metric-value">{results.get("summary", {}).get("pass_rate", 0):.1%}</div>
                        <div class="metric-label">Overall Pass Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results.get("summary", {}).get("average_score", 0):.3f}</div>
                        <div class="metric-label">Average Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results.get("summary", {}).get("total_questions", 0)}</div>
                        <div class="metric-label">Total Questions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(results.get("category_results", {}))}</div>
                        <div class="metric-label">Categories Tested</div>
                    </div>
                </div>
                
                <div class="grade-colors">
                    <div class="grade-item">
                        <div class="grade-color excellent"></div>
                        <div>Excellent (≥0.8)</div>
                    </div>
                    <div class="grade-item">
                        <div class="grade-color pass"></div>
                        <div>Pass (≥0.6)</div>
                    </div>
                    <div class="grade-item">
                        <div class="grade-color borderline"></div>
                        <div>Borderline (≥0.4)</div>
                    </div>
                    <div class="grade-item">
                        <div class="grade-color fail"></div>
                        <div>Fail (<0.4)</div>
                    </div>
                </div>
                
                <div class="charts-grid">"""

        # Add chart images if they exist
        chart_types = ["heatmap", "grades", "scatter", "documents"]
        chart_titles = [
            "Score Heatmap by Category and Metric",
            "Grade Distribution",
            "Search Time vs Overall Score",
            "Document Count Analysis"
        ]

        for chart_type, title in zip(chart_types, chart_titles):
            if chart_type in report_files:
                chart_path = Path(report_files[chart_type]).name
                html_content += f"""
                    <div class="chart-container">
                        <h3>{title}</h3>
                        <img src="{chart_path}" alt="{title}">
                    </div>"""

        html_content += """
                </div>
                
                <div class="files-section">
                    <h3>📋 Generated Report Files</h3>
                    <p>Click the links below to download detailed analysis files:</p>"""

        # Add download links
        file_descriptions = {
            "detailed_csv": "📊 Detailed Results (CSV)",
            "summary_csv": "📈 Summary Statistics (CSV)",
            "performance_csv": "⚡ Performance Comparison (CSV)",
            "category_csv": "📂 Category Analysis (CSV)"
        }

        for file_type, description in file_descriptions.items():
            if file_type in report_files:
                file_name = Path(report_files[file_type]).name
                html_content += f"""
                    <a href="{file_name}" class="file-link" download>{description}</a>"""

        html_content += """
                </div>
                
                <div style="margin-top: 30px; text-align: center; color: #666; font-size: 0.9em;">
                    <p>Generated by Clinical RAG Evaluation System</p>
                </div>
            </div>
        </body>
        </html>"""

        # Write HTML file
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\n🌐 Performance dashboard created: {dashboard_file}")
        return str(dashboard_file)

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

    def generate_comprehensive_report(self, results: Dict, output_dir: str = "evaluation_reports") -> Dict[str, str]:
        """Generate comprehensive evaluation report with visualizations and CSV exports"""

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}

        # 1. Export detailed results to CSV
        csv_file = output_path / f"detailed_results_{timestamp}.csv"
        df_results = pd.DataFrame(self.detailed_results)
        df_results.to_csv(csv_file, index=False)
        report_files["detailed_csv"] = str(csv_file)

        # 2. Create summary statistics CSV
        summary_file = output_path / f"summary_stats_{timestamp}.csv"
        summary_df = self._create_summary_dataframe(results)
        summary_df.to_csv(summary_file, index=False)
        report_files["summary_csv"] = str(summary_file)

        # 3. Generate visualizations
        viz_files = self._generate_visualizations(
            results, output_path, timestamp)
        report_files.update(viz_files)

        # 4. Create performance comparison table
        perf_file = output_path / f"performance_comparison_{timestamp}.csv"
        perf_df = self._create_performance_comparison(results)
        perf_df.to_csv(perf_file, index=False)
        report_files["performance_csv"] = str(perf_file)

        # 5. Generate category-wise analysis
        category_file = output_path / f"category_analysis_{timestamp}.csv"
        category_df = self._create_category_analysis(results)
        category_df.to_csv(category_file, index=False)
        report_files["category_csv"] = str(category_file)

        print(f"\n📊 Comprehensive report generated in: {output_path}")
        print("Generated files:")
        for file_type, file_path in report_files.items():
            print(f"  - {file_type}: {file_path}")

        return report_files

    def _create_summary_dataframe(self, results: Dict) -> pd.DataFrame:
        """Create summary statistics DataFrame"""
        summary = results.get("summary", {})
        category_breakdown = summary.get("category_breakdown", {})

        summary_data = []

        # Overall statistics
        summary_data.append({
            "Metric": "Overall Pass Rate",
            "Value": f"{summary.get('pass_rate', 0):.2%}",
            "Score": f"{summary.get('average_score', 0):.3f}"
        })

        summary_data.append({
            "Metric": "Total Questions",
            "Value": summary.get('total_questions', 0),
            "Score": "-"
        })

        # Category-wise statistics
        for category, stats in category_breakdown.items():
            summary_data.append({
                "Metric": f"{category.title()} Pass Rate",
                "Value": f"{stats.get('pass_rate', 0):.2%}",
                "Score": f"{stats.get('average_score', 0):.3f}"
            })

        return pd.DataFrame(summary_data)

    def _create_performance_comparison(self, results: Dict) -> pd.DataFrame:
        """Create performance comparison DataFrame"""
        detailed_results = results.get("detailed_results", [])

        performance_data = []
        for result in detailed_results:
            performance_data.append({
                "Question_ID": result.get("question_id", ""),
                "Category": result.get("category", ""),
                "Overall_Score": result.get("overall_score", 0),
                "Factual_Score": result.get("factual_accuracy_score", 0),
                "Behavior_Score": result.get("behavior_score", 0),
                "Performance_Score": result.get("performance_score", 0),
                "Context_Relevance": result.get("context_relevance_score", 0),
                "Completeness": result.get("completeness_score", 0),
                "Semantic_Similarity": result.get("semantic_similarity_score", 0),
                "Pass_Grade": result.get("pass_grade", "fail"),
                "Search_Time": result.get("search_time", 0),
                "Documents_Found": result.get("documents_found", 0),
                "Documents_Filtered": result.get("documents_filtered", False)
            })

        return pd.DataFrame(performance_data)

    def _create_category_analysis(self, results: Dict) -> pd.DataFrame:
        """Create category-wise analysis DataFrame"""
        category_results = results.get("category_results", {})

        analysis_data = []
        for category, cat_results in category_results.items():
            scores = [r.get("overall_score", 0) for r in cat_results]
            search_times = [r.get("search_time", 0) for r in cat_results]
            doc_counts = [r.get("documents_found", 0) for r in cat_results]

            analysis_data.append({
                "Category": category,
                "Question_Count": len(cat_results),
                "Pass_Rate": f"{sum(1 for r in cat_results if r.get('passed', False)) / len(cat_results):.2%}",
                "Avg_Score": f"{np.mean(scores):.3f}",
                "Score_StdDev": f"{np.std(scores):.3f}",
                "Min_Score": f"{np.min(scores):.3f}",
                "Max_Score": f"{np.max(scores):.3f}",
                "Avg_Search_Time": f"{np.mean(search_times):.2f}s",
                "Avg_Documents": f"{np.mean(doc_counts):.1f}",
                "Grade_Distribution": self._get_grade_distribution(cat_results)
            })

        return pd.DataFrame(analysis_data)

    def _get_grade_distribution(self, results: List[Dict]) -> str:
        """Get grade distribution for a category"""
        grades = [r.get("pass_grade", "fail") for r in results]
        grade_counts = pd.Series(grades).value_counts().to_dict()

        distribution = []
        for grade in ["excellent", "pass", "borderline", "fail"]:
            count = grade_counts.get(grade, 0)
            if count > 0:
                distribution.append(f"{grade}: {count}")

        return ", ".join(distribution)

    def _generate_visualizations(self, results: Dict, output_path: Path, timestamp: str) -> Dict[str, str]:
        """Generate various visualizations"""
        viz_files = {}

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Score Distribution Heatmap (Enhanced)
        fig, ax = plt.subplots(figsize=(12, 8))
        category_results = results.get("category_results", {})

        # Create score matrix for heatmap
        score_matrix = []
        categories = []
        metrics = ["Overall", "Factual", "Behavior",
                   "Performance", "Context", "Completeness", "Semantic"]

        for category, cat_results in category_results.items():
            categories.append(category)
            row = [
                np.mean([r.get("overall_score", 0) for r in cat_results]),
                np.mean([r.get("factual_accuracy_score", 0)
                        for r in cat_results]),
                np.mean([r.get("behavior_score", 0) for r in cat_results]),
                np.mean([r.get("performance_score", 0) for r in cat_results]),
                np.mean([r.get("context_relevance_score", 0)
                        for r in cat_results]),
                np.mean([r.get("completeness_score", 0) for r in cat_results]),
                np.mean([r.get("semantic_similarity_score", 0)
                        for r in cat_results])
            ]
            score_matrix.append(row)

        sns.heatmap(score_matrix,
                    xticklabels=metrics,
                    yticklabels=categories,
                    annot=True,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1,
                    fmt='.3f')

        plt.title("RAG Evaluation Score Heatmap by Category and Metric")
        plt.tight_layout()

        heatmap_file = output_path / f"score_heatmap_{timestamp}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files["heatmap"] = str(heatmap_file)

        # 2. Grade Distribution Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        all_grades = [r.get("pass_grade", "fail")
                      for r in results.get("detailed_results", [])]
        grade_counts = pd.Series(all_grades).value_counts()

        colors = {"excellent": "green", "pass": "lightgreen",
                  "borderline": "orange", "fail": "red"}
        grade_counts.plot(kind='bar',
                          color=[colors.get(x, 'gray')
                                 for x in grade_counts.index],
                          ax=ax)

        plt.title("Distribution of Pass Grades")
        plt.ylabel("Count")
        plt.xlabel("Grade")
        plt.xticks(rotation=45)
        plt.tight_layout()

        grades_file = output_path / f"grade_distribution_{timestamp}.png"
        plt.savefig(grades_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files["grades"] = str(grades_file)

        # 3. Search Time vs Score Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        detailed_results = results.get("detailed_results", [])

        scores = [r.get("overall_score", 0) for r in detailed_results]
        search_times = [r.get("search_time", 0) for r in detailed_results]
        categories = [r.get("category", "") for r in detailed_results]

        scatter = ax.scatter(search_times, scores, c=range(len(categories)),
                             alpha=0.6, cmap='tab10')

        ax.set_xlabel("Search Time (seconds)")
        ax.set_ylabel("Overall Score")
        ax.set_title("Search Time vs Overall Score")

        # Add trend line
        if len(search_times) > 1:
            z = np.polyfit(search_times, scores, 1)
            p = np.poly1d(z)
            ax.plot(search_times, p(search_times), "r--", alpha=0.8)

        plt.tight_layout()

        scatter_file = output_path / f"time_vs_score_{timestamp}.png"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files["scatter"] = str(scatter_file)

        return viz_files

    def create_performance_dashboard(self, results: Dict, output_dir: str = "evaluation_reports") -> str:
        """Create an HTML dashboard with all evaluation metrics"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = output_path / \
            f"performance_dashboard_{timestamp}.html"

        # Generate comprehensive report files
        report_files = self.generate_comprehensive_report(results, output_dir)

        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Dashboard - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric-card {{ display: inline-block; margin: 10px; padding: 15px; 
                               border: 1px solid #ddd; border-radius: 5px; min-width: 200px; }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
                .chart-container img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RAG System Evaluation Dashboard</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Summary Metrics</h2>
            <div class="metric-card">
                <h3>Overall Performance</h3>
                <p>Pass Rate: {results.get('summary', {}).get('pass_rate', 0):.1%}</p>
                <p>Average Score: {results.get('summary', {}).get('average_score', 0):.3f}</p>
            </div>
            
            <h2>Visualizations</h2>
            <div class="chart-container">
                <h3>Score Heatmap</h3>
                <img src="{Path(report_files.get('heatmap', '')).name}" alt="Score Heatmap">
            </div>
            
            <div class="chart-container">
                <h3>Grade Distribution</h3>
                <img src="{Path(report_files.get('grades', '')).name}" alt="Grade Distribution">
            </div>
            
            <div class="chart-container">
                <h3>Search Time vs Score Analysis</h3>
                <img src="{Path(report_files.get('scatter', '')).name}" alt="Time vs Score">
            </div>
            
            <h2>Detailed Reports</h2>
            <ul>
                <li><a href="{Path(report_files.get('detailed_csv', '')).name}">Detailed Results (CSV)</a></li>
                <li><a href="{Path(report_files.get('summary_csv', '')).name}">Summary Statistics (CSV)</a></li>
                <li><a href="{Path(report_files.get('performance_csv', '')).name}">Performance Comparison (CSV)</a></li>
                <li><a href="{Path(report_files.get('category_csv', '')).name}">Category Analysis (CSV)</a></li>
            </ul>
        </body>
        </html>
        """

        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📊 Performance dashboard created: {dashboard_file}")
        return str(dashboard_file)


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
