"""
Automated Model Evaluation Runner
Runs evaluations across all model combinations and stores results
"""

import sys
import time
from pathlib import Path
from typing import Dict
from RAG_chat_pipeline.config.config import model_names, llms, QUIET
from RAG_chat_pipeline.benchmarks.evaluation_results_manager import EvaluationResultsManager
from RAG_chat_pipeline.benchmarks.rag_evaluator import ClinicalRAGEvaluator
# from RAG_chat_pipeline.benchmarks.chat_history_evaluator import ChatHistoryEvaluator  # TODO: Create this module if needed
from RAG_chat_pipeline.core.main import main as initialize_clinical_rag
from RAG_chat_pipeline.benchmarks.gold_questions import generate_gold_questions_from_data


class ModelEvaluationRunner:
    """Runs systematic evaluations across model combinations"""

    def __init__(self):
        # Use the correct results directory where existing results are stored
        # Path to RAG_chat_pipeline/results instead of RAG_chat_pipeline/benchmarks/results
        results_dir = Path(__file__).parent.parent / "results"
        # Create results directory if it doesn't exist
        results_dir.mkdir(exist_ok=True)
        self.results_manager = EvaluationResultsManager(
            results_dir, quiet=QUIET)
        self.embedding_models = list(model_names.keys())
        self.llm_models = list(llms.keys())

    def run_single_evaluation(self,
                              embedding_model: str,
                              llm_model: str,
                              evaluation_type: str = "short",
                              notes: str = "",
                              use_chat_history: bool = False) -> str:
        """
        Run evaluation for a specific model combination

        Args:
            embedding_model: Embedding model nickname (e.g., "ms-marco")
            llm_model: LLM model nickname (e.g., "deepseek")
            evaluation_type: "quick", "short", or "full"
            notes: Optional notes for this evaluation
            use_chat_history: Whether to evaluate with chat history context

        Returns:
            experiment_id: Unique identifier for this experiment
        """

        eval_mode = "chat_history" if use_chat_history else "single_turn"
        if not QUIET:
            print(f"\n🚀 Running evaluation: {embedding_model} + {llm_model}")
            print(f"📊 Evaluation type: {evaluation_type}")
            print(f"🗣️ Evaluation mode: {eval_mode}")
            print("=" * 60)

        try:
            # Update config to use the specified models
            self._update_config(embedding_model, llm_model)

            # Initialize the RAG system with new config
            if not QUIET:
                print("Initializing RAG system...")
            chatbot = initialize_clinical_rag()

            # Choose evaluator based on mode
            if use_chat_history:
                # TODO: Implement ChatHistoryEvaluator when conversational evaluation is needed
                if not QUIET:
                    print(
                        "Warning: ChatHistoryEvaluator not implemented, falling back to ClinicalRAGEvaluator")
                evaluator = ClinicalRAGEvaluator(chatbot)
                if not QUIET:
                    print(
                        "Using ClinicalRAGEvaluator for single-turn evaluation (fallback)")
            else:
                evaluator = ClinicalRAGEvaluator(chatbot)
                if not QUIET:
                    print("Using ClinicalRAGEvaluator for single-turn evaluation")

            # Run the appropriate evaluation
            start_time = time.time()

            if use_chat_history:
                # Chat history evaluation with conversation scenarios
                num_scenarios = {"quick": 3, "short": 5,
                                 "full": 10}.get(evaluation_type, 5)
                results = evaluator.run_chat_history_evaluation(num_scenarios)
            else:
                # Standard single-turn evaluation
                if evaluation_type == "quick":
                    results = evaluator.run_quick_test()
                elif evaluation_type == "short":
                    results = evaluator.run_short_evaluation()
                elif evaluation_type == "full":
                    questions = generate_gold_questions_from_data()
                    results = evaluator.run_evaluation(questions)
                else:
                    raise ValueError(
                        f"Unknown evaluation type: {evaluation_type}")

            evaluation_time = time.time() - start_time

            # Add timing information to notes
            timing_note = f"Evaluation completed in {evaluation_time:.1f}s"
            full_notes = f"{notes}. {timing_note}" if notes else timing_note

            # Store results with additional config information
            # First, let's create a custom add_evaluation_result call that includes eval mode
            experiment_id = self._add_evaluation_result_with_mode(
                embedding_model, llm_model, results, full_notes, evaluation_type, use_chat_history
            )

            if not QUIET:
                print(f"✅ Evaluation completed: {experiment_id}")
                print(f"⏱️  Total time: {evaluation_time:.1f}s")

            return experiment_id

        except Exception as e:
            if not QUIET:
                print(f"❌ Error during evaluation: {e}")
            raise

    def _update_config(self, embedding_model: str, llm_model: str):
        """Update config files"""
        from RAG_chat_pipeline.config.config import set_models, get_config_summary

        try:
            # Set models dynamically
            set_models(embedding_model, llm_model)

            # Print configuration summary
            if not QUIET:
                print(f"📝 Updated config: {embedding_model} + {llm_model}")

        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            raise

    def _add_evaluation_result_with_mode(self, embedding_model: str, llm_model: str,
                                         results: Dict, notes: str, evaluation_type: str,
                                         use_chat_history: bool) -> str:
        """Add evaluation result with evaluation mode information"""
        # Call the original add_evaluation_result method
        experiment_id = self.results_manager.add_evaluation_result(
            embedding_model, llm_model, results, notes
        )

        # Update the config to include evaluation type and mode
        if experiment_id:
            experiments = self.results_manager.results_data["experiments"]
            if experiment_id in experiments:
                experiments[experiment_id]["config"]["eval_type"] = evaluation_type
                experiments[experiment_id]["config"]["evaluation_mode"] = "chat_history" if use_chat_history else "single_turn"
                self.results_manager._save_results()

        return experiment_id

    def run_all_combinations(self, evaluation_type: str = "short", skip_existing: bool = True, use_chat_history: bool = False):
        """
        Run evaluations for all model combinations

        Args:
            evaluation_type: "quick", "short", or "full"
            skip_existing: Skip combinations that already have results
            use_chat_history: Whether to evaluate with chat history context
        """

        total_combinations = len(self.embedding_models) * len(self.llm_models)
        eval_mode = "chat_history" if use_chat_history else "single_turn"
        print(
            f"🔄 Running evaluations for {total_combinations} model combinations")
        print(f"📊 Evaluation type: {evaluation_type}")
        print(f"🗣️ Evaluation mode: {eval_mode}")
        print(f"⏭️  Skip existing: {skip_existing}")
        print("=" * 60)

        completed = 0
        failed = 0
        skipped = 0

        for i, embedding_model in enumerate(self.embedding_models):
            for j, llm_model in enumerate(self.llm_models):
                combination_num = i * len(self.llm_models) + j + 1

                print(
                    f"\n[{combination_num}/{total_combinations}] {embedding_model} + {llm_model}")

                # Check if this combination already exists
                exists = self._combination_exists(
                    embedding_model, llm_model, evaluation_type, use_chat_history)
                if skip_existing and exists:
                    print("⏭️  Skipping - results already exist")
                    skipped += 1
                    continue
                elif exists:
                    print(
                        "⚠️  Found existing results but --no-skip flag was used. Running evaluation anyway.")

                # Run evaluation
                experiment_id = self.run_single_evaluation(
                    embedding_model,
                    llm_model,
                    evaluation_type,
                    f"Batch evaluation {combination_num}/{total_combinations}",
                    use_chat_history
                )

                if experiment_id:
                    completed += 1
                else:
                    failed += 1

                print(
                    f"✅ Progress: {completed} completed, {failed} failed, {skipped} skipped")

        print(f"\n🎉 Batch evaluation complete!")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")

        # Generate final report
        if completed > 0:
            print("\n📊 Generating comparison report...")
            self.results_manager.create_comparison_table()
            self.results_manager.generate_summary_report()
            print("\n📈 Generating performance heatmap...")
            self.results_manager.create_heatmap("average_score")

    def _combination_exists(self, embedding_model: str, llm_model: str, evaluation_type: str = None, use_chat_history: bool = False) -> bool:
        """Check if a model combination already has results"""
        print(
            f"Checking if combination exists: {embedding_model} + {llm_model}, eval_type={evaluation_type}, use_chat_history={use_chat_history}")
        print(
            f"Results data contains {len(self.results_manager.results_data['experiments'])} experiments")

        for exp_id, exp_data in self.results_manager.results_data["experiments"].items():
            config = exp_data["config"]

            # Check models match
            models_match = (config["embedding_model_nickname"] == embedding_model and
                            config["llm_nickname"] == llm_model)

            if not models_match:
                continue

            print(
                f"Found matching models in {exp_id}: {embedding_model} + {llm_model}")

            # Check evaluation type if specified
            if evaluation_type is not None:
                exp_eval_type = config.get("eval_type", "short")
                if exp_eval_type != evaluation_type:
                    print(
                        f"  Eval type mismatch: found {exp_eval_type}, looking for {evaluation_type}")
                    continue
                else:
                    print(f"  Eval type match: {evaluation_type}")

            # Check evaluation mode (chat history vs single turn)
            exp_mode = config.get("evaluation_mode", "single_turn")
            current_mode = "chat_history" if use_chat_history else "single_turn"
            if exp_mode != current_mode:
                print(
                    f"  Mode mismatch: found {exp_mode}, looking for {current_mode}")
                continue
            else:
                print(f"  Mode match: {current_mode}")

            # If we've reached here, we have a match
            print(
                f"✓ Found existing results for {embedding_model} + {llm_model}, eval_type={evaluation_type}, mode={current_mode}")
            return True

        print(
            f"✗ No existing results found for {embedding_model} + {llm_model}, eval_type={evaluation_type}, mode={'chat_history' if use_chat_history else 'single_turn'}")
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

    def debug_results_file(self):
        """Print debug information about the results file"""
        print(f"Results file location: {self.results_manager.results_file}")

        if not self.results_manager.results_file.exists():
            print("Results file does not exist!")
            return

        print(
            f"File size: {self.results_manager.results_file.stat().st_size} bytes")
        print(
            f"Last modified: {time.ctime(self.results_manager.results_file.stat().st_mtime)}")

        experiments = self.results_manager.results_data.get("experiments", {})
        print(f"Total experiments: {len(experiments)}")

        if experiments:
            print("\nExperiment summary:")
            for exp_id, exp_data in experiments.items():
                config = exp_data.get("config", {})
                emb_model = config.get("embedding_model_nickname", "unknown")
                llm_model = config.get("llm_nickname", "unknown")
                eval_type = config.get("eval_type", "unknown")
                mode = config.get("evaluation_mode", "unknown")
                print(
                    f"  {exp_id}: {emb_model} + {llm_model}, type={eval_type}, mode={mode}")


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("""
Clinical RAG Model Evaluation Runner

Usage:
    python model_evaluation_runner.py <command> [options]

Commands:
    single <emb_model> <llm_model> [eval_type] [--chat-history]  - Run single evaluation
    all [eval_type] [--chat-history] [--no-skip]               - Run all combinations
    list                                                        - List existing results
    report                                                      - Generate comparison report
    debug                                                      - Print debug information about results file
    clear                                                       - Clear all results

Examples:
    python model_evaluation_runner.py single ms-marco deepseek short
    python model_evaluation_runner.py single ms-marco deepseek short --chat-history
    python model_evaluation_runner.py all quick
    python model_evaluation_runner.py all full --no-skip  # Run all and don't skip existing
    python model_evaluation_runner.py all short --chat-history
    python model_evaluation_runner.py list
    python model_evaluation_runner.py report

Evaluation types: quick, short, full
Evaluation modes: single-turn (default), chat-history (with --chat-history flag)
Embedding models: ms-marco, multi-qa, mini-lm, static-retr
LLM models: deepseek, qwen, llama
        """)
        return

    runner = ModelEvaluationRunner()
    command = sys.argv[1].lower()

    # Check for chat history flag
    use_chat_history = "--chat-history" in sys.argv

    if command == "single":
        if len(sys.argv) < 4:
            print(
                "Usage: single <emb_model> <llm_model> [eval_type] [--chat-history]")
            return

        emb_model = sys.argv[2]
        llm_model = sys.argv[3]
        eval_type = "short"

        # Find eval_type in arguments (skip flags)
        for arg in sys.argv[4:]:
            if not arg.startswith("--"):
                eval_type = arg
                break

        if emb_model not in model_names:
            print(f"Invalid embedding model: {emb_model}")
            print(f"Available: {list(model_names.keys())}")
            return

        if llm_model not in llms:
            print(f"Invalid LLM model: {llm_model}")
            print(f"Available: {list(llms.keys())}")
            return

        runner.run_single_evaluation(
            emb_model, llm_model, eval_type, use_chat_history=use_chat_history)

    elif command == "all":
        eval_type = "short"

        # Find eval_type in arguments (skip flags)
        for arg in sys.argv[2:]:
            if not arg.startswith("--"):
                eval_type = arg
                break

        # Check if --no-skip flag is present
        skip_existing = "--no-skip" not in sys.argv

        runner.run_all_combinations(
            eval_type, skip_existing=skip_existing, use_chat_history=use_chat_history)

    elif command == "list":
        runner.list_existing_results()

    elif command == "report":
        print("📊 Generating comparison report...")
        runner.results_manager.create_comparison_table()
        report = runner.results_manager.generate_summary_report()
        print(report)

    elif command == "clear":
        runner.clear_results()

    elif command == "debug":
        runner.debug_results_file()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
