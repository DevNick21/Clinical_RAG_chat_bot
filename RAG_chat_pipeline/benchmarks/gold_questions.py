#!/usr/bin/env python3
"""
Gold Question Generator for Clinical RAG Evaluation

Generates practical test questions from real patient data with minimal complexity.
Focus: Simple, effective questions that test RAG system capabilities.
"""

import json
import random
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path


try:
    from RAG_chat_pipeline.utils.data_provider import get_sample_data
except ImportError:
    print("Warning: Could not import data_provider")
    get_sample_data = None


def generate_gold_questions_from_data(num_questions: int = 20,
                                      save_to_file: bool = False) -> List[Dict]:
    """Generate gold questions from clinical data based on actual document structure

    Questions are generated to test the RAG system's ability to retrieve and answer
    questions about the available clinical data sections:
    - header: admission info, admit/discharge times, expire flag
    - diagnoses: ICD diagnosis codes with descriptions
    - procedures: ICD procedure codes with descriptions
    - labs: laboratory tests with values, times, categories, flags
    - microbiology: culture tests with specimen types and comments
    - prescriptions: medications with dosages, times, order status

    Args:
        num_questions: Number of questions to generate
        save_to_file: Whether to save questions to file

    Returns:
        List of generated questions with expected patterns
    """

    if get_sample_data is None:
        print("Error: Data loader not available")
        return []

    try:
        data = get_sample_data()
        if not data or 'hadm_ids' not in data:
            print("Error: No admission data available")
            return []
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

    questions = []
    hadm_ids = data['hadm_ids']

    if len(hadm_ids) == 0:
        print("Error: No admission records found")
        return []

    # Question templates based on actual document structure
    # All templates use hadm_id and reflect available data sections
    templates = [
        # Header/admission info (20% of questions) - from header section
        ("What type of admission was {hadm_id}?",
         "header", "admission type from header"),
        ("When was admission {hadm_id} admitted and discharged?",
         "header", "admission and discharge times"),
        ("What is the expire flag status for admission {hadm_id}?",
         "header", "hospital expire flag from header"),
        ("Show me the basic information for admission {hadm_id}",
         "header", "header with admit/discharge times and type"),

        # Diagnoses (25% of questions) - from diagnoses_icd section
        ("What diagnoses were recorded for admission {hadm_id}?",
         "diagnoses", "ICD diagnosis codes with descriptions"),
        ("List all ICD diagnosis codes for admission {hadm_id}",
         "diagnoses", "ICD codes and long titles"),
        ("What conditions does admission {hadm_id} have?",
         "diagnoses", "diagnosis descriptions from ICD codes"),
        ("Show me the diagnostic information for admission {hadm_id}",
         "diagnoses", "complete diagnosis list with ICD codes"),

        # Procedures (15% of questions) - from procedures_icd section
        ("What procedures were performed during admission {hadm_id}?",
         "procedures", "ICD procedure codes with descriptions"),
        ("List all ICD procedure codes for admission {hadm_id}",
         "procedures", "procedure codes and long titles"),
        ("What surgical interventions were done for admission {hadm_id}?",
         "procedures", "procedures from ICD procedure codes"),

        # Laboratory tests (25% of questions) - from labs section
        ("What lab tests were performed for admission {hadm_id}?",
         "labs", "lab item IDs, labels, and categories"),
        ("Show me the laboratory results for admission {hadm_id}",
         "labs", "lab values with chart/store times"),
        ("What were the lab values and flags for admission {hadm_id}?",
         "labs", "lab values, units, and abnormal flags"),
        ("List all laboratory categories tested for admission {hadm_id}?",
         "labs", "lab categories and fluid types"),
        ("What priority lab tests were done for admission {hadm_id}?",
         "labs", "lab priorities and timing information"),

        # Microbiology (15% of questions) - from microbiology section
        ("What microbiology tests were performed for admission {hadm_id}?",
         "microbiology", "microbiology test names and specimen types"),
        ("Were there any cultures taken during admission {hadm_id}?",
         "microbiology", "culture tests with chart/store times"),
        ("Show me the microbiology results for admission {hadm_id}",
         "microbiology", "test results with comments and dates"),

        # Prescriptions/medications (25% of questions) - from prescriptions section
        ("What medications were prescribed for admission {hadm_id}?",
         "prescriptions", "drug names and formulary codes"),
        ("Show me the drug dosages for admission {hadm_id}",
         "prescriptions", "dose values, units, and frequencies"),
        ("What were the medication administration times for admission {hadm_id}?",
         "prescriptions", "order times and chart times"),
        ("List all drug types prescribed for admission {hadm_id}",
         "prescriptions", "drug types and product strengths"),
        ("What is the medication order status for admission {hadm_id}?",
         "prescriptions", "order types and status information"),

        # Cross-section temporal queries (10% of questions)
        ("When were the first lab tests performed for admission {hadm_id}?",
         "labs", "earliest lab chart times"),
        ("What was the timeline of medication administration for admission {hadm_id}?",
         "prescriptions", "chronological medication events"),
        ("Show me the dates of microbiology tests for admission {hadm_id}",
         "microbiology", "microbiology test dates"),

        # Comprehensive queries (5% of questions) - cross-section
        ("Give me a complete summary of admission {hadm_id}",
         "comprehensive", "header, diagnoses, procedures, labs, meds"),
        ("What happened during the entire stay for admission {hadm_id}?",
         "comprehensive", "chronological summary across all sections"),
    ]

    # Generate questions
    for i in range(num_questions):
        generation_time = datetime.now()
        # Select random admission ID
        hadm_id = random.choice(hadm_ids)

        # Select question template
        template, category, expected_pattern = random.choice(templates)

        # Format question using hadm_id
        question_text = template.format(hadm_id=hadm_id)

        # Create question object
        question = {
            "id": f"q_{i+1:03d}_{generation_time.strftime('%Y%m%d_%H%M%S')}",
            "question": question_text,
            "category": category,
            "hadm_id": str(hadm_id),  # Use hadm_id instead of patient_id
            "expected_answer_pattern": expected_pattern,
        }

        questions.append(question)

    # Save if requested
    if save_to_file and questions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gold_questions_{timestamp}.json"
        output_path = filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        print(f"✅ Generated {len(questions)} questions, saved to: {filename}")

    # Print statistics
    categories = {}
    for q in questions:
        cat = q['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n📊 Question Statistics:")
    print(f"   Total: {len(questions)} questions")
    print(f"   Categories: {dict(categories)}")
    print(
        f"   Unique admissions: {len(set(q['hadm_id'] for q in questions))}")

    return questions


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate gold questions for RAG evaluation")
    parser.add_argument("-n", "--num-questions", type=int, default=20,
                        help="Number of questions (default: 20)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to file")

    args = parser.parse_args()

    print("🎯 Generating Gold Questions for RAG Evaluation")
    print("=" * 50)

    questions = generate_gold_questions_from_data(
        num_questions=args.num_questions,
        save_to_file=not args.no_save
    )

    if questions:
        print(f"\n📋 Sample Questions:")
        for i, q in enumerate(questions[:3], 1):
            print(f"   {i}. [{q['category']}] {q['question']}")
            print(f"      Expected: {q['expected_answer_pattern']}")

        if len(questions) > 3:
            print(f"   ... and {len(questions) - 3} more questions")
    else:
        print("❌ No questions generated")


if __name__ == "__main__":
    main()
