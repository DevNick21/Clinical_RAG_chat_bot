"""
Automated Model Evaluation Runner
Runs evaluations across all model combinations and stores results
"""

import sys
import time
from pathlib import Path
from config import model_names, llms
from evaluation_results_manager import EvaluationResultsManager
from rag_evaluator import ClinicalRAGEvaluator
from main import initialize_clinical_rag
from gold_questions import generate_gold_questions_from_data


class ModelEvaluationRunner:
    """Runs systematic evaluations across model combinations"""

    def __init__(self):
        self.results_manager = EvaluationResultsManager()
        self.embedding_models = list(model_names.keys())
        self.llm_models = list(llms.keys())

    def run_single_evaluation(self,
                              embedding_model: str,
                              llm_model: str,
                              evaluation_type: str = "short",
                              notes: str = "") -> str:
        """
        Run evaluation for a specific model combination

        Args:
            embedding_model: Embedding model nickname (e.g., "ms-marco")
            llm_model: LLM model nickname (e.g., "deepseek")
            evaluation_type: "quick", "short", or "full"
            notes: Optional notes for this evaluation

        Returns:
            experiment_id: Unique identifier for this experiment
        """

        print(f"\n🚀 Running evaluation: {embedding_model} + {llm_model}")
        print(f"📊 Evaluation type: {evaluation_type}")
        print("=" * 60)

        try:
            # Update config to use the specified models
            self._update_config(embedding_model, llm_model)

            # Initialize the RAG system with new config
            print("Initializing RAG system...")
            chatbot = initialize_clinical_rag()
            evaluator = ClinicalRAGEvaluator(chatbot)

            # Run the appropriate evaluation
            start_time = time.time()

            if evaluation_type == "quick":
                results = evaluator.run_quick_test()
            elif evaluation_type == "short":
                results = evaluator.run_short_evaluation()
            elif evaluation_type == "full":
                questions = generate_gold_questions_from_data()
                results = evaluator.run_evaluation(questions)
            else:
                raise ValueError(f"Unknown evaluation type: {evaluation_type}")

            evaluation_time = time.time() - start_time

            # Add timing information to notes
            timing_note = f"Evaluation completed in {evaluation_time:.1f}s"
            full_notes = f"{notes}. {timing_note}" if notes else timing_note

            # Store results
            experiment_id = self.results_manager.add_evaluation_result(
                embedding_model, llm_model, results, full_notes
            )

            print(f"✅ Evaluation completed: {experiment_id}")
            print(f"⏱️  Total time: {evaluation_time:.1f}s")

            return experiment_id

        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            print(f"   Models: {embedding_model} + {llm_model}")
            return None

    def _update_config(self, embedding_model: str, llm_model: str):
        """Update config.py to use specified models"""
        config_file = Path("config.py")

        # Read current config
        with open(config_file, 'r') as f:
            content = f.read()

        # Update the model_in_use and LLM_MODEL variables
        lines = content.split('\\n')
        updated_lines = []

        for line in lines:
            if line.startswith('model_in_use = '):
                updated_lines.append(f'model_in_use = "{embedding_model}"')
            elif line.startswith('LLM_MODEL = '):
                updated_lines.append(f'LLM_MODEL = llms["{llm_model}"]')
            else:
                updated_lines.append(line)

        # Write updated config
        with open(config_file, 'w') as f:
            f.write('\\n'.join(updated_lines))

        print(f"📝 Updated config: {embedding_model} + {llm_model}")

    def run_all_combinations(self, evaluation_type: str = "short", skip_existing: bool = True):
        """
        Run evaluations for all model combinations

        Args:
            evaluation_type: "quick", "short", or "full"
            skip_existing: Skip combinations that already have results
        """

        total_combinations = len(self.embedding_models) * len(self.llm_models)
        print(
            f"🔄 Running evaluations for {total_combinations} model combinations")
        print(f"📊 Evaluation type: {evaluation_type}")
        print(f"⏭️  Skip existing: {skip_existing}")
        print("=" * 60)

        completed = 0
        failed = 0
        skipped = 0

        for i, embedding_model in enumerate(self.embedding_models):
            for j, llm_model in enumerate(self.llm_models):
                combination_num = i * len(self.llm_models) + j + 1

                print(
                    f"\\n[{combination_num}/{total_combinations}] {embedding_model} + {llm_model}")

                # Check if this combination already exists
                if skip_existing and self._combination_exists(embedding_model, llm_model):
                    print("⏭️  Skipping - results already exist")
                    skipped += 1
                    continue

                # Run evaluation
                experiment_id = self.run_single_evaluation(
                    embedding_model,
                    llm_model,
                    evaluation_type,
                    f"Batch evaluation {combination_num}/{total_combinations}"
                )

                if experiment_id:
                    completed += 1
                else:
                    failed += 1

                print(
                    f"✅ Progress: {completed} completed, {failed} failed, {skipped} skipped")

        print(f"\\n🎉 Batch evaluation complete!")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")

        # Generate final report
        if completed > 0:
            print("\\n📊 Generating comparison report...")
            self.results_manager.create_comparison_table()
            self.results_manager.generate_summary_report()
            print("\\n📈 Generating performance heatmap...")
            self.results_manager.create_heatmap("pass_rate")

    def _combination_exists(self, embedding_model: str, llm_model: str) -> bool:
        """Check if a model combination already has results"""
        for exp_id, exp_data in self.results_manager.results_data["experiments"].items():
            config = exp_data["config"]
            if (config["embedding_model_nickname"] == embedding_model and
                    config["llm_nickname"] == llm_model):
                return True
        return False

    def list_existing_results(self):
        """List all existing evaluation results"""
        experiments = self.results_manager.results_data["experiments"]

        if not experiments:
            print("No evaluation results found.")
            return

        print(f"📊 Found {len(experiments)} evaluation results:")
        print("=" * 60)

        for exp_id, exp_data in experiments.items():
            config = exp_data["config"]
            metrics = exp_data["metrics"]

            print(f"🔬 {exp_id}")
            print(
                f"   Models: {config['embedding_model_nickname']} + {config['llm_nickname']}")
            print(f"   Pass Rate: {metrics['pass_rate']:.1%}")
            print(f"   Avg Score: {metrics['average_score']:.3f}")
            print(f"   Date: {exp_data['experiment_date'][:10]}")
            print()

    def clear_results(self, confirm: bool = False):
        """Clear all evaluation results (use with caution!)"""
        if not confirm:
            print("⚠️  This will delete ALL evaluation results!")
            response = input("Are you sure? Type 'yes' to confirm: ")
            if response.lower() != 'yes':
                print("❌ Operation cancelled")
                return

        # Reset results
        self.results_manager.results_data = {
            "metadata": {
                "created_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "last_updated": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_experiments": 0
            },
            "experiments": {}
        }

        self.results_manager._save_results()
        print("🗑️  All evaluation results cleared")


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("""
Clinical RAG Model Evaluation Runner

Usage:
    python model_evaluation_runner.py <command> [options]

Commands:
    single <emb_model> <llm_model> [eval_type]  - Run single evaluation
    all [eval_type]                             - Run all combinations
    list                                        - List existing results
    report                                      - Generate comparison report
    clear                                       - Clear all results

Examples:
    python model_evaluation_runner.py single ms-marco deepseek short
    python model_evaluation_runner.py all quick
    python model_evaluation_runner.py list
    python model_evaluation_runner.py report

Evaluation types: quick, short, full
Embedding models: ms-marco, multi-qa, mini-lm, static-retr
LLM models: deepseek, qwen, llama
        """)
        return

    runner = ModelEvaluationRunner()
    command = sys.argv[1].lower()

    if command == "single":
        if len(sys.argv) < 4:
            print("Usage: single <emb_model> <llm_model> [eval_type]")
            return

        emb_model = sys.argv[2]
        llm_model = sys.argv[3]
        eval_type = sys.argv[4] if len(sys.argv) > 4 else "short"

        if emb_model not in model_names:
            print(f"Invalid embedding model: {emb_model}")
            print(f"Available: {list(model_names.keys())}")
            return

        if llm_model not in llms:
            print(f"Invalid LLM model: {llm_model}")
            print(f"Available: {list(llms.keys())}")
            return

        runner.run_single_evaluation(emb_model, llm_model, eval_type)

    elif command == "all":
        eval_type = sys.argv[2] if len(sys.argv) > 2 else "short"
        runner.run_all_combinations(eval_type)

    elif command == "list":
        runner.list_existing_results()

    elif command == "report":
        print("📊 Generating comparison report...")
        runner.results_manager.create_comparison_table()
        report = runner.results_manager.generate_summary_report()
        print(report)

    elif command == "clear":
        runner.clear_results()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
