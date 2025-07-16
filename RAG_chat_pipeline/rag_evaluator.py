"""Streamlined RAG evaluation framework focused on practical assessment"""
import re
import json
import sys
from typing import Dict, List
from datetime import datetime

from data_loader import get_sample_data
from clinical_rag import ClinicalRAGBot
from gold_questions import generate_gold_questions_from_data
from config import *


class ClinicalRAGEvaluator:
    """Configuration-driven RAG evaluator using config.py parameters"""

    def __init__(self, chatbot: ClinicalRAGBot):
        self.chatbot = chatbot
        self.patient_data = get_sample_data()
        self.results = []

    def evaluate_question(self, gold_question: Dict) -> Dict:
        """Evaluate a single question with weighted scoring"""
        result = {
            "question": gold_question["question"],
            "category": gold_question["category"],
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Get chatbot response
            hadm_id = self._clean_hadm_id(
                gold_question.get("hadm_id"))
            response = self.chatbot.ask_single_question(
                question=gold_question["question"],
                hadm_id=hadm_id,
                k=5,
                search_strategy="auto"
            )

            # Score the response
            factual_score = self.evaluate_factual_accuracy(
                gold_question, response)
            behavior_score = self.evaluate_behavior(gold_question, response)
            performance_score = self.evaluate_performance(response)

            # Weighted overall score using config
            overall_score = (
                factual_score * EVALUATION_SCORING_WEIGHTS["factual_accuracy"] +
                behavior_score * EVALUATION_SCORING_WEIGHTS["behavior"] +
                performance_score * EVALUATION_SCORING_WEIGHTS["performance"]
            )

            # Pass/fail determination
            pass_threshold = self.get_pass_threshold(gold_question["category"])
            passed = overall_score >= pass_threshold

            result.update({
                "response": response.get("answer", ""),
                "factual_accuracy_score": factual_score,
                "behavior_score": behavior_score,
                "performance_score": performance_score,
                "overall_score": overall_score,
                "pass_threshold": pass_threshold,
                "passed": passed,
                "search_time": response.get("search_time", 0),
                "documents_found": response.get("documents_found", 0),
                "error": None
            })

        except Exception as e:
            result.update({
                "error": str(e),
                "passed": False,
                "overall_score": 0.0,
                "response": ""
            })

        self.results.append(result)
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

    def get_pass_threshold(self, category: str) -> float:
        """Get pass threshold based on question category from config"""
        return EVALUATION_PASS_THRESHOLDS.get(category, EVALUATION_PASS_THRESHOLDS["default"])

    def _process_questions(self, questions: List[Dict], show_detailed: bool = False) -> Dict:
        """Core method to process questions with optional detailed output"""
        detailed_results = []
        category_results = {}
        passed_count = 0

        for i, question in enumerate(questions):
            if show_detailed:
                print(f"\nQUESTION {i+1}: {question['category']}")
                print(
                    f"Q: {question['question'][:EVALUATION_OUTPUT_CONFIG['question_preview_length']]}...")
            else:
                print(
                    f"[{i+1}/{len(questions)}] {question['category']}: {question['question'][:60]}...")

            # Evaluate question
            result = self.evaluate_question(question)
            detailed_results.append(result)

            if show_detailed:
                print(
                    f"A: {result['response'][:EVALUATION_OUTPUT_CONFIG['answer_preview_length']]}...")
                print(f"Scores: Factual={result.get('factual_accuracy_score', 0):.2f}, "
                      f"Behavior={result.get('behavior_score', 0):.2f}, "
                      f"Performance={result.get('performance_score', 0):.2f}")
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

        return {
            "detailed_results": detailed_results,
            "category_results": category_results,
            "passed_count": passed_count,
            "total_questions": len(questions)
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
        print(f"\n{'='*50}")
        print(
            f"SUMMARY: {summary['passed']}/{summary['total_questions']} passed ({summary['pass_rate']:.1%})")
        print(f"Average Score: {summary['average_score']:.2f}")

        for category, stats in summary["category_breakdown"].items():
            print(
                f"  {category}: {stats['pass_rate']:.1%} ({stats['count']} questions)")

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
            result = self.evaluate_question(question)
            results.append(result)
            print(
                f"   -> {result['overall_score']:.2f} ({'PASS' if result['passed'] else 'FAIL'})")

        return {"summary": {"tested": len(test_questions)}, "results": results}


def main():
    """Main function for running evaluations"""
    if len(sys.argv) < 2:
        print("Usage: python rag_evaluator.py [full|short|quick]")
        print(f"  full  - Run complete evaluation")
        print(
            f"  short - Run evaluation on {EVALUATION_DEFAULT_PARAMS['short_evaluation_limit']} questions with details")
        print(
            f"  quick - Run quick test with {EVALUATION_DEFAULT_PARAMS['quick_test_limit']} questions")
        return
    from main import initialize_clinical_rag

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
