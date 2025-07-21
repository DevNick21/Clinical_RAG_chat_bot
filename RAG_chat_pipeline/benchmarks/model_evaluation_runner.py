"""
Automated Model Evaluation Runner
Runs evaluations across all model combinations and stores results
"""

# Fix imports to avoid using RAG_chat_pipeline prefix when inside the package
from ..config.config import model_names, llms
from .evaluation_results_manager import EvaluationResultsManager
from .rag_evaluator import ClinicalRAGEvaluator
from ..core.main import initialize_clinical_rag
from .gold_questions import generate_gold_questions_from_data
import sys
import time
import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict, Optional, Set


class ModelEvaluationRunner:
    """Runs systematic evaluations across model combinations"""

    def __init__(self):
        # Initialize results manager with the correct path - using absolute paths
        current_dir = Path(__file__).resolve().parent
        results_dir = current_dir.parent / "results"
        # Ensure results directory exists
        results_dir.mkdir(exist_ok=True)
        self.results_manager = EvaluationResultsManager(results_dir)
        self.embedding_models = list(model_names.keys())
        self.llm_models = list(llms.keys())

    def run_single_evaluation(self,
                              embedding_model: str,
                              llm_model: str,
                              evaluation_type: str = "short",
                              notes: str = "",
                              verbose: bool = True) -> str:
        """
        Run evaluation for a specific model combination

        Args:
            embedding_model: Embedding model nickname (e.g., "ms-marco")
            llm_model: LLM model nickname (e.g., "deepseek")
            evaluation_type: "quick", "short", or "full"
            notes: Optional notes for this evaluation
            verbose: Whether to print detailed output (defaults to True for single runs)

        Returns:
            experiment_id: Unique identifier for this experiment
        """
        # Detect if we're in a batch run (progress bar) or individual run
        in_progress_bar = hasattr(
            tqdm, '_instances') and len(tqdm._instances) > 0

        # If we're in a batch run with a progress bar, reduce verbosity by default
        if in_progress_bar and verbose is True:
            verbose = False

        if verbose:
            print(f"\n🚀 Running evaluation: {embedding_model} + {llm_model}")
            print(f"📊 Evaluation type: {evaluation_type}")
            print("=" * 60)

        try:
            # Update config to use the specified models
            self._update_config(embedding_model, llm_model, verbose=verbose)

            # Initialize the RAG system with new config
            if verbose:
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

            if verbose:
                print(f"✅ Evaluation completed: {experiment_id}")
                print(f"⏱️  Total time: {evaluation_time:.1f}s")

                # Generate visualizations
                print("\n📊 Generating visualizations...")
                try:
                    # Only create comparison table and heatmaps if we have enough data
                    if len(self.results_manager.results_data["experiments"]) >= 1:
                        self.results_manager.create_comparison_table()

                    if len(self.results_manager.results_data["experiments"]) >= 2:
                        # Only create heatmaps if we have multiple experiments for comparison
                        self.results_manager.create_heatmap("pass_rate")
                        self.results_manager.create_heatmap("average_score")
                except Exception as e:
                    print(
                        f"⚠️ Warning: Could not generate some visualizations: {e}")

            return experiment_id

        except ValueError as e:
            # Handle specific validation errors
            print(f"❌ Evaluation failed due to invalid value: {str(e)}")
            print(f"   Models: {embedding_model} + {llm_model}")
            return None
        except ImportError as e:
            # Handle model loading errors
            print(f"❌ Evaluation failed to import required model: {str(e)}")
            print(f"   Models: {embedding_model} + {llm_model}")
            print(f"   Please check that the models are properly installed/configured")
            return None
        except Exception as e:
            # Handle general errors with more diagnostic information
            print(f"❌ Evaluation failed with error: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Models: {embedding_model} + {llm_model}")
            import traceback
            print("   Detailed traceback:")
            traceback.print_exc()
            return None

    def _update_config(self, embedding_model: str, llm_model: str, verbose: bool = True):
        """Update config files"""
        from RAG_chat_pipeline.config.config import set_models, get_config_summary

        try:
            # Set models dynamically
            set_models(embedding_model, llm_model)

            # Print configuration summary if verbose
            if verbose:
                print(f"📝 Updated config: {embedding_model} + {llm_model}")

        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            raise

    def run_all_combinations(self, evaluation_type: str = "short", skip_existing: bool = True,
                             selected_embeddings: List[str] = None, selected_llms: List[str] = None):
        """
        Run evaluations for all model combinations

        Args:
            evaluation_type: "quick", "short", or "full"
            skip_existing: Skip combinations that already have results
            selected_embeddings: List of specific embedding models to evaluate (default: all)
            selected_llms: List of specific LLM models to evaluate (default: all)
        """
        # Filter models if selections are provided
        embedding_models = selected_embeddings if selected_embeddings else self.embedding_models
        llm_models = selected_llms if selected_llms else self.llm_models

        # Validate model selections
        invalid_embeddings = [
            m for m in embedding_models if m not in self.embedding_models]
        invalid_llms = [m for m in llm_models if m not in self.llm_models]

        if invalid_embeddings:
            print(f"⚠️ Invalid embedding models: {invalid_embeddings}")
            print(f"Available embedding models: {self.embedding_models}")
            return

        if invalid_llms:
            print(f"⚠️ Invalid LLM models: {invalid_llms}")
            print(f"Available LLM models: {self.llm_models}")
            return

        # Calculate combinations after filtering
        combinations = [(e, l) for e in embedding_models for l in llm_models]
        total_combinations = len(combinations)

        print(
            f"🔄 Running evaluations for {total_combinations} model combinations")
        print(f"📊 Evaluation type: {evaluation_type}")
        print(f"⏭️  Skip existing: {skip_existing}")
        if selected_embeddings:
            print(f"📋 Selected embedding models: {selected_embeddings}")
        if selected_llms:
            print(f"📋 Selected LLM models: {selected_llms}")
        print("=" * 60)

        # Pre-check which combinations exist to provide better estimates
        if skip_existing:
            existing_combinations = []
            for emb, llm in combinations:
                if self._combination_exists(emb, llm):
                    existing_combinations.append((emb, llm))

            if existing_combinations:
                print(
                    f"⏭️ Found {len(existing_combinations)} existing results that will be skipped:")
                for emb, llm in existing_combinations:
                    print(f"  • {emb} + {llm}")

        completed = 0
        failed = 0
        skipped = 0

        # Track timing for better estimates
        start_time = time.time()
        avg_evaluation_time = 0

        # Setup progress bar
        pbar = tqdm(combinations, desc="Evaluating model combinations",
                    unit="combo", ncols=100, leave=True)

        for embedding_model, llm_model in pbar:
            combination_num = completed + failed + skipped + 1

            # Update progress bar description
            pbar.set_description(f"Evaluating {embedding_model} + {llm_model}")

            # Check if this combination already exists
            if skip_existing and self._combination_exists(embedding_model, llm_model):
                skipped += 1
                pbar.set_postfix(completed=completed,
                                 failed=failed, skipped=skipped)
                continue

            # Run evaluation
            combo_start_time = time.time()
            experiment_id = self.run_single_evaluation(
                embedding_model,
                llm_model,
                evaluation_type,
                f"Batch evaluation {combination_num}/{total_combinations}"
            )
            combo_time = time.time() - combo_start_time

            if experiment_id:
                completed += 1
                # Update average time for remaining estimates
                avg_evaluation_time = (
                    (avg_evaluation_time * (completed - 1)) + combo_time) / completed

                # Calculate estimated time remaining
                remaining_combos = total_combinations - \
                    (completed + failed + skipped)
                est_remaining_time = remaining_combos * avg_evaluation_time
                est_completion_time = datetime.datetime.now(
                ) + datetime.timedelta(seconds=est_remaining_time)

                # Update progress bar
                pbar.set_postfix(
                    completed=completed,
                    failed=failed,
                    skipped=skipped,
                    avg_time=f"{avg_evaluation_time:.1f}s",
                    eta=est_completion_time.strftime("%H:%M:%S")
                )
            else:
                failed += 1
                pbar.set_postfix(completed=completed,
                                 failed=failed, skipped=skipped)

        # Calculate runtime statistics
        total_runtime = time.time() - start_time
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        print(f"\\n🎉 Batch evaluation complete!")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   ⏱️  Total runtime: {runtime_str}")

        # Generate final report
        if completed > 0:
            try:
                print("\\n📊 Generating comparison report...")
                self.results_manager.create_comparison_table()
                report = self.results_manager.generate_summary_report()
                print(report)

                print("\\n📈 Generating performance heatmaps...")
                # Generate heatmaps for multiple metrics
                for metric in ["pass_rate", "average_score", "avg_search_time"]:
                    try:
                        self.results_manager.create_heatmap(metric)
                    except Exception as e:
                        print(
                            f"⚠️ Failed to create heatmap for {metric}: {str(e)}")

                # Find the best performing model combination
                df = self.results_manager.get_results_dataframe()
                if not df.empty:
                    best_overall = df.loc[df["average_score"].idxmax()]
                    fastest = df.loc[df["avg_search_time"].idxmin()]

                    print("\\n🏆 Recommended Model Combinations:")
                    print(
                        f"   🥇 Best overall performance: {best_overall['embedding_model']} + {best_overall['llm_model']}")
                    print(
                        f"      Score: {best_overall['average_score']:.3f}, Pass Rate: {best_overall['pass_rate']:.1%}")
                    print(
                        f"   ⚡ Fastest performance: {fastest['embedding_model']} + {fastest['llm_model']}")
                    print(
                        f"      Time: {fastest['avg_search_time']:.2f}s, Score: {fastest['average_score']:.3f}")

                    # Suggest command to use the best model
                    print("\\n💡 To use the best performing model combination:")
                    print(f"   from RAG_chat_pipeline.config.config import set_models")
                    print(
                        f"   set_models('{best_overall['embedding_model']}', '{best_overall['llm_model']}')")
            except Exception as e:
                print(f"⚠️ Error generating final reports: {str(e)}")
                print("   Individual evaluation results were still saved successfully.")

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
    """Main CLI interface using argparse for better command handling"""
    parser = argparse.ArgumentParser(
        description="Clinical RAG Model Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single evaluation command
    single_parser = subparsers.add_parser(
        "single", help="Run a single model evaluation")
    single_parser.add_argument("emb_model", help="Embedding model nickname")
    single_parser.add_argument("llm_model", help="LLM model nickname")
    single_parser.add_argument("--type", "-t", default="short", choices=["quick", "short", "full"],
                               help="Evaluation type (default: short)")
    single_parser.add_argument("--verbose", "-v", action="store_true",
                               help="Show detailed output during evaluation")

    # All combinations command
    all_parser = subparsers.add_parser(
        "all", help="Run evaluations for all model combinations")
    all_parser.add_argument("--type", "-t", default="short", choices=["quick", "short", "full"],
                            help="Evaluation type (default: short)")
    all_parser.add_argument("--no-skip", action="store_true",
                            help="Don't skip combinations that already have results")
    all_parser.add_argument("--embeddings", "-e", nargs="+",
                            help="Specific embedding models to evaluate")
    all_parser.add_argument("--llms", "-l", nargs="+",
                            help="Specific LLM models to evaluate")
    all_parser.add_argument("--quiet", "-q", action="store_true",
                            help="Reduce output verbosity")

    # List results command
    list_parser = subparsers.add_parser(
        "list", help="List existing evaluation results")

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate comparison report")
    report_parser.add_argument("--metrics", "-m", nargs="+",
                               default=["pass_rate", "average_score",
                                        "avg_search_time"],
                               help="Metrics to include in the report")

    # Clear results command
    clear_parser = subparsers.add_parser(
        "clear", help="Clear all evaluation results")
    clear_parser.add_argument("--confirm", "-y", action="store_true",
                              help="Skip confirmation prompt")

    # Parse arguments
    if len(sys.argv) == 1:
        # Print help if no arguments provided
        parser.print_help()
        return

    args = parser.parse_args()
    runner = ModelEvaluationRunner()

    # Process commands
    if args.command == "single":
        # Validate models
        if args.emb_model not in model_names:
            print(f"❌ Invalid embedding model: {args.emb_model}")
            print(f"📋 Available models: {list(model_names.keys())}")
            return

        if args.llm_model not in llms:
            print(f"❌ Invalid LLM model: {args.llm_model}")
            print(f"📋 Available models: {list(llms.keys())}")
            return

        # Run single evaluation
        runner.run_single_evaluation(args.emb_model, args.llm_model, args.type)

    elif args.command == "all":
        # Run all combinations with options
        runner.run_all_combinations(
            evaluation_type=args.type,
            skip_existing=not args.no_skip,
            selected_embeddings=args.embeddings,
            selected_llms=args.llms
        )

    elif args.command == "list":
        runner.list_existing_results()

    elif args.command == "report":
        print("📊 Generating comparison report...")
        runner.results_manager.create_comparison_table()
        report = runner.results_manager.generate_summary_report()
        print(report)

        print("📈 Generating performance heatmaps...")
        for metric in args.metrics:
            try:
                runner.results_manager.create_heatmap(metric)
            except Exception as e:
                print(f"⚠️ Failed to create heatmap for {metric}: {e}")

    elif args.command == "clear":
        runner.clear_results(confirm=args.confirm)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
