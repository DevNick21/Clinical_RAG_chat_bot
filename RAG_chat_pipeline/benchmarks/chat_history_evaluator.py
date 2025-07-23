"""
Chat History Evaluation Framework
Evaluates RAG performance with conversational context and follow-up questions
"""

import json
import time
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from RAG_chat_pipeline.core.clinical_rag import ClinicalRAGBot
from RAG_chat_pipeline.benchmarks.rag_evaluator import ClinicalRAGEvaluator
from RAG_chat_pipeline.benchmarks.gold_questions import generate_gold_questions_from_data
from RAG_chat_pipeline.config.config import *


class ChatHistoryEvaluator(ClinicalRAGEvaluator):
    """Evaluates RAG system with chat history context"""

    def __init__(self, chatbot: ClinicalRAGBot):
        super().__init__(chatbot)
        self.conversation_scenarios = []
        self.results = []

    def generate_conversation_scenarios(self, num_scenarios: int = 10) -> List[Dict]:
        """Generate conversation scenarios with follow-up questions"""
        base_questions = generate_gold_questions_from_data(num_scenarios * 2)
        scenarios = []

        for i in range(0, len(base_questions), 2):
            if i + 1 < len(base_questions):
                initial_q = base_questions[i]
                follow_up_q = base_questions[i + 1]

                # Create conversation scenario
                scenario = {
                    "id": f"conv_{i//2 + 1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "hadm_id": initial_q["hadm_id"],
                    "category": "conversational",
                    "conversation": [
                        {
                            "turn": 1,
                            "question": initial_q["question"],
                            "expected_category": initial_q["category"],
                            "expected_pattern": initial_q["expected_answer_pattern"]
                        },
                        {
                            "turn": 2,
                            "question": self._create_follow_up_question(initial_q, follow_up_q),
                            "expected_category": follow_up_q["category"],
                            "expected_pattern": follow_up_q["expected_answer_pattern"],
                            "requires_context": True
                        }
                    ]
                }
                scenarios.append(scenario)

        return scenarios

    def _create_follow_up_question(self, initial_q: Dict, follow_up_q: Dict) -> str:
        """Create contextual follow-up questions"""
        follow_up_templates = {
            "header": [
                "Can you tell me more about that?",
                "What else happened during this admission?",
                "When exactly did this occur?"
            ],
            "diagnoses": [
                "How serious are these conditions?",
                "Are there any related complications?",
                "What's the primary diagnosis?"
            ],
            "procedures": [
                "Were these procedures successful?",
                "How long did they take?",
                "What was the recovery like?"
            ],
            "labs": [
                "Are these values normal?",
                "How do they compare to previous results?",
                "What do these numbers indicate?"
            ],
            "microbiology": [
                "What do these cultures show?",
                "Were any antibiotics prescribed?",
                "Is this a serious infection?"
            ],
            "prescriptions": [
                "How often should these be taken?",
                "Are there any side effects?",
                "What are these medications for?"
            ]
        }

        category = initial_q.get("category", "header")
        templates = follow_up_templates.get(
            category, follow_up_templates["header"])

        import random
        return random.choice(templates)

    def evaluate_conversation(self, scenario: Dict) -> Dict:
        """Evaluate a complete conversation scenario"""
        result = {
            "scenario_id": scenario["id"],
            "hadm_id": scenario["hadm_id"],
            "conversation_results": [],
            "overall_score": 0.0,
            "context_preservation_score": 0.0,
            "timestamp": datetime.now().isoformat()
        }

        chat_history = []

        for turn in scenario["conversation"]:
            turn_result = {
                "turn": turn["turn"],
                "question": turn["question"],
                "requires_context": turn.get("requires_context", False)
            }

            try:
                # Get response with chat history
                response = self.chatbot.chat(turn["question"], chat_history)

                # Create evaluation data structure
                gold_question = {
                    "question": turn["question"],
                    "category": turn["expected_category"],
                    "hadm_id": scenario["hadm_id"],
                    "expected_answer_pattern": turn["expected_pattern"]
                }

                # Evaluate the response
                eval_result = self.evaluate_question(gold_question)
                eval_result["response"] = response

                # Additional scoring for context preservation in follow-up questions
                if turn.get("requires_context"):
                    context_score = self._evaluate_context_preservation(
                        turn["question"], response, chat_history
                    )
                    eval_result["context_preservation_score"] = context_score
                else:
                    eval_result["context_preservation_score"] = 1.0

                turn_result.update(eval_result)

                # Add to chat history for next turn
                chat_history.extend([
                    ("user", turn["question"]),
                    ("assistant", response)
                ])

            except Exception as e:
                turn_result.update({
                    "error": str(e),
                    "passed": False,
                    "overall_score": 0.0,
                    "context_preservation_score": 0.0,
                    "response": ""
                })

            result["conversation_results"].append(turn_result)

        # Calculate overall scores
        if result["conversation_results"]:
            scores = [r.get("overall_score", 0)
                      for r in result["conversation_results"]]
            context_scores = [r.get("context_preservation_score", 0)
                              for r in result["conversation_results"]]

            result["overall_score"] = sum(scores) / len(scores)
            result["context_preservation_score"] = sum(
                context_scores) / len(context_scores)

        return result

    def _evaluate_context_preservation(self, question: str, response: str, chat_history: List) -> float:
        """Evaluate how well the system preserves conversation context"""
        if not chat_history:
            return 1.0  # No previous context to preserve

        score = 0.0

        # Check if response acknowledges previous context
        context_indicators = [
            "these", "them", "those", "that", "this", "mentioned", "discussed",
            "previously", "earlier", "above", "the admission", "the patient"
        ]

        response_lower = response.lower()
        context_found = sum(
            1 for indicator in context_indicators if indicator in response_lower)

        if context_found > 0:
            score += 0.5

        # Check if admission ID is preserved when needed
        if any("admission" in msg[1].lower() or "hadm_id" in msg[1].lower() for msg in chat_history):
            if "admission" in response_lower or any(str(i) in response for i in range(10000000, 99999999)):
                score += 0.3

        # Check for logical consistency with previous responses
        if len(chat_history) >= 2:
            prev_response = chat_history[-1][1].lower()
            if any(word in response_lower and word in prev_response for word in
                   ["diagnosis", "procedure", "medication", "lab", "test"]):
                score += 0.2

        return min(1.0, score)

    def run_chat_history_evaluation(self, num_scenarios: int = 10) -> Dict:
        """Run comprehensive chat history evaluation"""
        print(
            f"\n🗣️ Running Chat History Evaluation with {num_scenarios} scenarios")
        print("=" * 60)

        # Generate conversation scenarios
        scenarios = self.generate_conversation_scenarios(num_scenarios)

        detailed_results = []
        total_scenarios = len(scenarios)
        passed_scenarios = 0

        for i, scenario in enumerate(scenarios):
            print(
                f"\n[{i+1}/{total_scenarios}] Evaluating conversation scenario...")
            print(f"HADM ID: {scenario['hadm_id']}")

            result = self.evaluate_conversation(scenario)
            detailed_results.append(result)

            if result["overall_score"] >= 0.7:  # Threshold for passing
                passed_scenarios += 1

            print(f"Overall Score: {result['overall_score']:.3f}")
            print(
                f"Context Preservation: {result['context_preservation_score']:.3f}")

        # Calculate summary statistics
        overall_scores = [r["overall_score"] for r in detailed_results]
        context_scores = [r["context_preservation_score"]
                          for r in detailed_results]

        summary = {
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "pass_rate": passed_scenarios / total_scenarios if total_scenarios else 0,
            "average_overall_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            "average_context_score": sum(context_scores) / len(context_scores) if context_scores else 0,
            "evaluation_type": "chat_history",
            "evaluation_date": datetime.now().isoformat()
        }

        print(f"\n{'='*60}")
        print(f"CHAT HISTORY EVALUATION SUMMARY:")
        print(
            f"  Scenarios: {summary['passed_scenarios']}/{summary['total_scenarios']} passed ({summary['pass_rate']:.1%})")
        print(
            f"  Average Overall Score: {summary['average_overall_score']:.3f}")
        print(
            f"  Average Context Score: {summary['average_context_score']:.3f}")

        return {
            "summary": summary,
            "detailed_results": detailed_results,
            "scenarios": scenarios
        }

    def save_chat_evaluation_results(self, results: Dict, filename: str = None) -> str:
        """Save chat history evaluation results"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_evaluation_{timestamp}.json"

        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Chat history evaluation results saved to: {filepath}")
        return str(filepath)


def main():
    """Main function for running chat history evaluations"""
    import sys

    if len(sys.argv) < 2:
        print("""
Clinical RAG Chat History Evaluator

Usage:
    python chat_history_evaluator.py <num_scenarios>

Examples:
    python chat_history_evaluator.py 5   # Run 5 conversation scenarios
    python chat_history_evaluator.py 10  # Run 10 conversation scenarios
        """)
        return

    try:
        num_scenarios = int(sys.argv[1])
    except ValueError:
        print("Error: Number of scenarios must be an integer")
        return

    # Initialize RAG system
    from RAG_chat_pipeline.core.main import main as initialize_clinical_rag

    print("🚀 Initializing Clinical RAG System for chat history evaluation...")
    chatbot = initialize_clinical_rag()
    evaluator = ChatHistoryEvaluator(chatbot)

    # Run evaluation
    results = evaluator.run_chat_history_evaluation(num_scenarios)

    # Save results
    evaluator.save_chat_evaluation_results(results)


if __name__ == "__main__":
    main()
