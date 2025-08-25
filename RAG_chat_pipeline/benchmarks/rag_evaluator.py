"""RAG evaluation framework focused on practical assessment"""
import re
import json
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from RAG_chat_pipeline.utils.data_provider import get_sample_data
from RAG_chat_pipeline.core.clinical_rag import ClinicalRAGBot, ClinicalLogger
from .gold_questions import generate_gold_questions_from_data
from .reporting import EvaluationReporter
from RAG_chat_pipeline.config.config import *
from RAG_chat_pipeline.benchmarks.visualization import EvaluationVisualizer


class ClinicalRAGEvaluator:
    """Configuration-driven RAG evaluator using config.py parameters"""

    def __init__(self, chatbot: ClinicalRAGBot):
        self.chatbot = chatbot
        self.patient_data = get_sample_data()
        self.results = []

        # Embedding cache to avoid redundant calculations
        self.embedding_cache = {}

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
            original_docs_len = len(source_docs)
            docs_were_filtered = False
            if original_docs_len > 10:
                filtered_docs = self.filter_and_rank_documents(
                    source_docs, gold_question["question"], top_k=10
                )
                response["source_documents"] = filtered_docs
                response["documents_found"] = len(filtered_docs)
                source_docs = response.get("source_documents", [])
                docs_were_filtered = True
            # Medical QA focused scoring - factual accuracy dominates
            factual_score = self.evaluate_factual_accuracy(
                gold_question, response)

            # Context relevance and semantic similarity (supporting metrics)
            context_relevance = self.evaluate_context_relevance(
                gold_question["question"], source_docs
            )

            # Semantic similarity with medical keywords
            medical_keywords = self._get_medical_keywords_for_category(
                gold_question["category"])
            semantic_score = self.evaluate_semantic_similarity(
                response.get("answer", ""), medical_keywords
            )

            # Performance evaluation
            performance_score = self.evaluate_performance(response)

            # Medical QA weighted overall score (factual accuracy dominates)
            overall_score = (
                factual_score * EVALUATION_SCORING_WEIGHTS["factual_accuracy"] +
                context_relevance * EVALUATION_SCORING_WEIGHTS["context_relevance"] +
                semantic_score * EVALUATION_SCORING_WEIGHTS["semantic_similarity"] +
                performance_score * EVALUATION_SCORING_WEIGHTS["performance"]
            )

            # Score is already normalized (weights sum to 1.0)

            # Enhanced pass/fail determination with grades
            pass_grade, grade_value = self.get_pass_grade(
                overall_score, gold_question["category"])
            passed = grade_value >= 0

            result.update({
                "response": response.get("answer", ""),
                "enhanced_question": enhanced_question,
                "factual_accuracy_score": factual_score,
                "context_relevance_score": context_relevance,
                "semantic_similarity_score": semantic_score,
                "performance_score": performance_score,
                "overall_score": overall_score,
                "pass_threshold": self.get_pass_threshold(gold_question["category"]),
                "pass_grade": pass_grade,
                "grade_value": grade_value,
                "passed": passed,
                "search_time": response.get("search_time", 0),
                "documents_found": response.get("documents_found", 0),
                "documents_filtered": docs_were_filtered,
                "source_documents": response.get("source_documents", []),
                "error": None
            })

        except Exception as e:
            result.update({
                "error": str(e),
                "passed": False,
                "overall_score": 0.0,
                "response": "",
                "pass_grade": "fail",
                "grade_value": -1
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
        if category in ["labs", "prescriptions"]:
            # Keep complex scoring for labs and prescriptions
            if category == "labs":
                return self._validate_labs_answer(answer, patient_facts, gold_question)
            else:  # prescriptions
                return self._validate_prescriptions_answer(answer, patient_facts, gold_question)
        else:
            # Use simplified validation for all other categories
            return self._validate_category_answer(category, answer, patient_facts, gold_question)

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
            ClinicalLogger.warning(f"Error extracting patient facts: {e}")
            pass  # Return empty facts on error

        return facts

    def evaluate_semantic_similarity(self, generated_answer: str, expected_keywords: List[str]) -> float:
        """Evaluate semantic similarity using clinical embeddings with caching"""
        if not expected_keywords or not generated_answer.strip():
            return 0.0

        try:
            # Get cached or compute embedding for generated answer
            gen_embedding = self._get_cached_embedding(generated_answer)

            # Get cached or compute embedding for expected keywords (combined)
            expected_text = " ".join(expected_keywords)
            exp_embedding = self._get_cached_embedding(expected_text)

            # Calculate cosine similarity and normalize to [0,1]
            similarity = cosine_similarity(
                [gen_embedding], [exp_embedding])[0][0]
            normalized = float((similarity + 1.0) / 2.0)
            return max(0.0, min(1.0, normalized))

        except Exception as e:
            ClinicalLogger.warning(
                f"Semantic similarity calculation failed: {e}")
            return 0.0

    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get cached embedding or compute and cache if not exists"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.chatbot.clinical_emb.embed_query(
                text)
        return self.embedding_cache[text]

    def evaluate_performance(self, response: Dict) -> float:
        """Evaluate system performance metrics"""
        score = 1.0

        # Search time penalty
        search_time = response.get("search_time", 0)
        if search_time > EVALUATION_PERFORMANCE_THRESHOLDS["slow_search_time"]:
            score *= (1 - EVALUATION_PERFORMANCE_THRESHOLDS["slow_penalty"])
        elif search_time > EVALUATION_PERFORMANCE_THRESHOLDS["moderate_search_time"]:
            score *= (1 -
                      EVALUATION_PERFORMANCE_THRESHOLDS["moderate_penalty"])

        # Document retrieval penalty
        docs_found = response.get("documents_found", 0)
        if docs_found == 0:
            score *= (1 - EVALUATION_PERFORMANCE_THRESHOLDS["no_docs_penalty"])

        return max(0.0, min(1.0, score))

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
            ClinicalLogger.warning(
                f"Context relevance calculation failed: {e}")
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
            ClinicalLogger.warning(f"Document filtering failed: {e}")
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
            You are a clinical expert analyzing patient medical records. Based on all available medical data for this patient, provide a comprehensive clinical answer to this question with enhanced query routing.

            QUERY ROUTING STRATEGY:
            1. First, identify which medical record sections are most relevant to answer this question
            2. Cross-reference data across multiple sources to validate findings
            3. Prioritize factual medical accuracy over narrative completeness
            
            Your comprehensive analysis should integrate information from:
            - Admission records and patient demographics
            - All documented diagnoses (ICD codes with clinical descriptions)
            - Procedures performed (chronological with clinical context)
            - Laboratory results (with normal ranges and clinical significance)
            - Microbiology findings (cultures, sensitivities, treatment implications)
            - Complete medication profile (dosages, timing, interactions)
            - Clinical notes and provider assessments
            
            Question: {question}
            
            MEDICAL QA FRAMEWORK - Provide detailed analysis that:
            1. FACTUAL ACCURACY: Answer directly with verified medical data from records
            2. CLINICAL EVIDENCE: Support all conclusions with specific documented findings
            3. PATTERN RECOGNITION: Identify clinically significant trends across record types
            4. COMPLETENESS ASSESSMENT: Note missing data that impacts clinical understanding
            5. MEDICAL REASONING: Apply clinical logic and appropriate medical terminology
            6. QUERY ROUTING: Explain which record sections provided key information
            
            Focus on medical accuracy as the primary evaluation criterion. Use clinical reasoning to synthesize information across multiple data sources.
            """
        }

        return category_prompts.get(category, question)

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

        detailed_results = []
        category_results = {}
        passed_count = 0

        for i, question in enumerate(questions):
            question_id = f"q_{i}"

            if show_detailed:
                ClinicalLogger.info(
                    f"\nQUESTION {i+1}: {question['category']}")
                ClinicalLogger.info(
                    f"Q: {question['question'][:EVALUATION_OUTPUT_CONFIG['question_preview_length']]}...")
            else:
                ClinicalLogger.info(
                    f"[{i+1}/{len(questions)}] {question['category']}: {question['question'][:60]}...")

            # Evaluate question
            result = self.evaluate_question(question, question_id)
            detailed_results.append(result)

            if show_detailed:
                ClinicalLogger.info(
                    f"A: {result['response'][:EVALUATION_OUTPUT_CONFIG['answer_preview_length']]}...")
                ClinicalLogger.info(f"Medical Accuracy: {result.get('factual_accuracy_score', 0):.2f}, "
                                    f"Context Relevance: {result.get('context_relevance_score', 0):.2f}, "
                                    f"Semantic Match: {result.get('semantic_similarity_score', 0):.2f}")

                ClinicalLogger.info(
                    f"Overall: {result['overall_score']:.2f} ({'PASS' if result['passed'] else 'FAIL'})")
                ClinicalLogger.info(
                    "-" * EVALUATION_OUTPUT_CONFIG["separator_length"])

            # Track statistics
            if result['passed']:
                passed_count += 1

            # Group by category
            category = result["category"]
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)

        # Calculate batch retrieval metrics
        # Removed broken batch retrieval metrics functionality

        return {
            "detailed_results": detailed_results,
            "category_results": category_results,
            "passed_count": passed_count,
            "total_questions": len(questions),
        }

    def _calculate_summary(self, processed_results: Dict) -> Dict:
        """Calculate summary statistics from processed results"""
        detailed_results = processed_results["detailed_results"]
        category_results = processed_results["category_results"]
        passed_count = processed_results["passed_count"]
        total_questions = processed_results["total_questions"]

        summary = {
            "total_questions": total_questions,
            "passed": passed_count,
            "pass_rate": passed_count / total_questions if total_questions else 0,
            "average_score": sum(r["overall_score"] for r in detailed_results) / len(detailed_results) if detailed_results else 0,
            "category_breakdown": {}
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
        ClinicalLogger.info(f"\n{'='*50}")
        ClinicalLogger.info(
            f"SUMMARY: {summary['passed']}/{summary['total_questions']} passed ({summary['pass_rate']:.1%})")
        ClinicalLogger.info(f"Average Score: {summary['average_score']:.2f}")

        for category, stats in summary["category_breakdown"].items():
            ClinicalLogger.info(
                f"  {category}: {stats['pass_rate']:.1%} ({stats['count']} questions)")

        # Removed broken retrieval metrics printing functionality

    def run_evaluation(self, gold_questions: List[Dict]) -> Dict:
        """Run evaluation on all questions"""
        ClinicalLogger.info(f"\nEvaluating {len(gold_questions)} questions...")
        ClinicalLogger.info("="*50)

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

    def _validate_category_answer(self, category: str, answer: str, facts: Dict, gold_question: Dict) -> float:
        """Unified validation method for simplified categories"""
        if category == "header":
            question = gold_question["question"].lower()
            if "type" in question:
                return 1.0 if any(keyword in answer.lower() for keyword in EVALUATION_MEDICAL_KEYWORDS["header"]["admission_type"]) else 0.3
            elif "when" in question or "admitted" in question or "discharged" in question:
                return 1.0 if re.search(EVALUATION_MEDICAL_KEYWORDS["header"]["date_pattern"], answer) else 0.4
            elif "expire" in question:
                return 1.0 if any(keyword in answer.lower() for keyword in EVALUATION_MEDICAL_KEYWORDS["header"]["expire"]) else 0.3
            else:
                return self._basic_medical_score(answer, facts)

        elif category in ["diagnoses", "procedures", "microbiology"]:
            # Use structured data validation for these categories
            keywords = EVALUATION_MEDICAL_KEYWORDS[category]["primary"]
            code_pattern = EVALUATION_MEDICAL_KEYWORDS[category].get(
                "code_pattern")
            additional_keywords = EVALUATION_MEDICAL_KEYWORDS[category].get(
                "specimen_types") if category == "microbiology" else None
            return self._validate_structured_data(answer, keywords, code_pattern, additional_keywords)

        elif category == "comprehensive":
            section_keywords = EVALUATION_MEDICAL_KEYWORDS["comprehensive"]["sections"]
            sections_covered = sum(1 for keywords in section_keywords.values()
                                   if any(keyword in answer.lower() for keyword in keywords))
            return min(1.0, sections_covered / len(section_keywords))

        else:
            return self._basic_medical_score(answer, facts)

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
        ClinicalLogger.info(
            f"\nSHORT EVALUATION: {len(test_questions)} questions with detailed output")
        ClinicalLogger.info(
            "="*EVALUATION_OUTPUT_CONFIG["long_separator_length"])

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
        ClinicalLogger.info(f"\nQUICK TEST: {len(test_questions)} questions")
        ClinicalLogger.info("="*EVALUATION_OUTPUT_CONFIG["separator_length"])

        results = []
        for i, question in enumerate(test_questions):
            ClinicalLogger.info(
                f"\n{i+1}. {question['question'][:EVALUATION_OUTPUT_CONFIG['question_preview_length']]}...")
            # Pass question_id for retrieval metrics
            result = self.evaluate_question(question, f"q_{i}")
            results.append(result)
            ClinicalLogger.info(
                f"   -> {result['overall_score']:.2f} ({'PASS' if result['passed'] else 'FAIL'})")

        return {"summary": {"tested": len(test_questions)}, "results": results}

    def generate_comprehensive_report(self, results: Dict, output_dir: str = "evaluation_reports") -> Dict[str, str]:
        """Generate comprehensive evaluation report using modular components"""
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize reporter
        reporter = EvaluationReporter(output_path)

        # Generate reports only (no visualizations or dashboard)
        report_files = reporter.export_comprehensive_csv(results, timestamp)

        ClinicalLogger.info(
            f"\n Evaluation report generated in: {output_path}")
        ClinicalLogger.info("Generated files:")
        for file_type, file_path in report_files.items():
            if file_path:
                ClinicalLogger.info(f"  - {file_type}: {Path(file_path).name}")

        return report_files

    def compare_evaluations(self, evaluation_results: List[Dict], labels: List[str],
                            output_dir: str = "evaluation_reports", generate_plots: bool = False,
                            quiet: bool = False) -> str:
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

        if generate_plots:
            try:
                visualizer = EvaluationVisualizer(output_dir=output_path)
                img_path = visualizer.create_evaluation_comparison_plot(
                    labels, comparison_data, timestamp)
                if img_path and not quiet:
                    ClinicalLogger.info(
                        f" Comparison visualization saved: {img_path}")
            except Exception as e:
                if not quiet:
                    ClinicalLogger.error(
                        f" Error generating comparison visualization: {e}")

        if not quiet:
            ClinicalLogger.info(
                f" Evaluation comparison saved: {comparison_file}")
        return str(comparison_file)


# Duplicate methods removed - functionality moved to modular components

def main():
    """Main function for running evaluations"""
    if len(sys.argv) < 2:
        ClinicalLogger.info(
            "Usage: python rag_evaluator.py [full|short|quick|comprehensive]")
        ClinicalLogger.info(f"  full          - Run complete evaluation")
        ClinicalLogger.info(
            f"  short         - Run evaluation on {EVALUATION_DEFAULT_PARAMS['short_evaluation_limit']} questions with details")
        ClinicalLogger.info(
            f"  quick         - Run quick test with {EVALUATION_DEFAULT_PARAMS['quick_test_limit']} questions")
        ClinicalLogger.info(
            f"  comprehensive - Run full evaluation with comprehensive reporting and visualizations")
        return

    from RAG_chat_pipeline.core.main import initialize_clinical_rag

    ClinicalLogger.info("Initializing RAG system...")
    chatbot = initialize_clinical_rag()
    evaluator = ClinicalRAGEvaluator(chatbot)
    mode = sys.argv[1].lower()

    if mode == "full":
        questions = generate_gold_questions_from_data()
        results = evaluator.run_evaluation(questions)

        # Save results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        ClinicalLogger.info("\nResults saved to evaluation_results.json")

    elif mode == "comprehensive":
        questions = generate_gold_questions_from_data()
        results = evaluator.run_evaluation(questions)

        # Generate comprehensive report
        report_files = evaluator.generate_comprehensive_report(results)
        results["report_files"] = report_files

        # Save comprehensive results
        with open('comprehensive_evaluation_results.json', 'w') as f:
            json.dump({k: v for k, v in results.items()
                      if k != "report_files"}, f, indent=2)
        ClinicalLogger.info(
            f"\nComprehensive results saved to comprehensive_evaluation_results.json")

    elif mode == "short":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = evaluator.run_short_evaluation(limit)

    elif mode == "quick":
        num = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = evaluator.run_quick_test(num)

    else:
        ClinicalLogger.warning(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
