"""
Evaluation Results Manager for Clinical RAG System
Handles storage, analysis, and visualization of evaluation results across different model combinations
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from RAG_chat_pipeline.config.config import model_names, llms
from langchain.schema import Document
from RAG_chat_pipeline.benchmarks.visualization import EvaluationVisualizer


@dataclass
class ModelConfiguration:
    """Configuration for a specific model combination"""
    embedding_model: str
    llm_model: str
    vectorstore_name: str
    model_nickname: str  # e.g., "ms-marco", "multi-qa"
    llm_nickname: str    # e.g., "deepseek", "qwen"


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a single run"""
    # Basic metrics
    total_questions: int
    passed: int
    pass_rate: float
    average_score: float

    # Detailed scoring breakdown
    avg_factual_accuracy: float
    avg_behavior_score: float
    avg_performance_score: float

    # Category-specific metrics
    header_pass_rate: float = 0.0
    header_avg_score: float = 0.0
    diagnoses_pass_rate: float = 0.0
    diagnoses_avg_score: float = 0.0
    procedures_pass_rate: float = 0.0
    procedures_avg_score: float = 0.0
    labs_pass_rate: float = 0.0
    labs_avg_score: float = 0.0
    microbiology_pass_rate: float = 0.0
    microbiology_avg_score: float = 0.0
    prescriptions_pass_rate: float = 0.0
    prescriptions_avg_score: float = 0.0
    comprehensive_pass_rate: float = 0.0
    comprehensive_avg_score: float = 0.0

    # Performance metrics
    avg_search_time: float = 0.0
    avg_documents_found: float = 0.0

    # Additional metadata
    evaluation_date: str = ""
    notes: str = ""


class EvaluationResultsManager:
    """Manages evaluation results across different model combinations"""

    def __init__(self, results_dir: Path = None, quiet: bool = False):
        self.quiet = quiet
        # Use the results folder in the parent directory with absolute paths
        if results_dir is None:
            current_dir = Path(__file__).resolve().parent
            self.results_dir = current_dir.parent / "results"
        else:
            # Ensure provided path is absolute
            self.results_dir = results_dir.resolve(
            ) if not results_dir.is_absolute() else results_dir

        # Ensure the results directory exists
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Main results file
        self.results_file = self.results_dir / "model_comparison_results.json"
        self.dataframe_file = self.results_dir / "results_dataframe.csv"

        # Load existing results or initialize empty
        self.results_data = self._load_existing_results()

        # Generate model configurations
        self.model_configs = self._generate_model_configurations()

    def _generate_model_configurations(self) -> List[ModelConfiguration]:
        """Generate all possible model combinations from config"""
        configurations = []

        for emb_nickname, emb_config in model_names.items():
            for llm_nickname, llm_model in llms.items():
                config = ModelConfiguration(
                    embedding_model=emb_config[1],  # Full model name
                    llm_model=llm_model,
                    vectorstore_name=emb_config[2],  # Vectorstore name
                    model_nickname=emb_nickname,
                    llm_nickname=llm_nickname
                )
                configurations.append(config)

        return configurations

    def _load_existing_results(self) -> Dict:
        """Load existing results from file"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if not self.quiet:
                    print(f"Warning: Could not load existing results: {e}")

        return {
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_experiments": 0
            },
            "experiments": {}
        }

    def _save_results(self):
        """Save results to file"""
        self.results_data["metadata"]["last_updated"] = datetime.now(
        ).isoformat()

        with open(self.results_file, 'w') as f:
            json.dump(self.results_data, f, indent=2,
                      default=self._serialize_documents)

    def _serialize_documents(self, obj):
        """Custom JSON serializer for Document objects and other non-serializable types"""
        if isinstance(obj, Document):
            return {
                "page_content": obj.page_content,
                "metadata": obj.metadata,
                "_type": "Document"  # Mark as Document for potential deserialization
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            raise TypeError(
                f"Object of type {type(obj)} is not JSON serializable")

    def _serialize_evaluation_result(self, result):
        """Serialize evaluation result to handle Document objects and other non-serializable types"""
        if isinstance(result, dict):
            serialized = {}
            for key, value in result.items():
                serialized[key] = self._serialize_evaluation_result(value)
            return serialized
        elif isinstance(result, list):
            return [self._serialize_evaluation_result(item) for item in result]
        elif isinstance(result, Document):
            return {
                "page_content": result.page_content,
                "metadata": result.metadata,
                "_type": "Document"
            }
        elif isinstance(result, (np.ndarray, np.integer, np.floating)):
            # Handle numpy types
            if isinstance(result, np.ndarray):
                return result.tolist()
            elif isinstance(result, np.integer):
                return int(result)
            elif isinstance(result, np.floating):
                return float(result)
        else:
            # Return as-is for basic types (str, int, float, bool, None)
            return result

    def add_evaluation_result(self,
                              embedding_model_nickname: str,
                              llm_nickname: str,
                              evaluation_result: Dict,
                              notes: str = "") -> str:
        """
        Add an evaluation result for a specific model combination

        Args:
            embedding_model_nickname: e.g., "ms-marco", "multi-qa"
            llm_nickname: e.g., "deepseek", "qwen"
            evaluation_result: The result dict from RAG evaluator
            notes: Optional notes about this experiment

        Returns:
            experiment_id: Unique identifier for this experiment
        """

        # Generate experiment ID
        experiment_id = f"{embedding_model_nickname}_{llm_nickname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract metrics from evaluation result
        metrics = self._extract_metrics_from_result(evaluation_result, notes)

        # Serialize evaluation result to handle Document objects
        serialized_result = self._serialize_evaluation_result(
            evaluation_result)

        # Store the experiment
        self.results_data["experiments"][experiment_id] = {
            "config": {
                "embedding_model_nickname": embedding_model_nickname,
                "llm_nickname": llm_nickname,
                "embedding_model_full": model_names[embedding_model_nickname][1],
                "llm_model_full": llms[llm_nickname],
                "vectorstore": model_names[embedding_model_nickname][2]
            },
            "metrics": asdict(metrics),
            "raw_results": serialized_result,
            "experiment_date": datetime.now().isoformat()
        }

        self.results_data["metadata"]["total_experiments"] += 1
        self._save_results()

        if not self.quiet:
            print(f" Added evaluation result: {experiment_id}")
            print(f"    Pass Rate: {metrics.pass_rate:.1%}")
            print(f"    Average Score: {metrics.average_score:.3f}")

        return experiment_id

    def _extract_metrics_from_result(self, result: Dict, notes: str = "") -> EvaluationMetrics:
        """Extract structured metrics from evaluation result"""

        summary = result.get("summary", {})
        category_breakdown = summary.get("category_breakdown", {})

        # Check if using quick test format or full/short evaluation format
        if "results" in result:  # Quick test format
            detailed_results = result.get("results", [])
        else:  # Full/short evaluation format
            detailed_results = result.get("detailed_results", [])

        # Calculate average component scores
        if detailed_results:
            factual_scores = [r.get("factual_accuracy_score", 0)
                              for r in detailed_results]
            behavior_scores = [r.get("behavior_score", 0)
                               for r in detailed_results]
            performance_scores = [r.get("performance_score", 0)
                                  for r in detailed_results]
            search_times = [r.get("search_time", 0) for r in detailed_results]
            docs_found = [r.get("documents_found", 0)
                          for r in detailed_results]

            # Count passed questions
            passed_count = sum(
                1 for r in detailed_results if r.get("passed", False))
            total_questions = len(detailed_results)
            pass_rate = passed_count / total_questions if total_questions else 0
            avg_score = sum(r.get("overall_score", 0)
                            for r in detailed_results) / total_questions if total_questions else 0

            avg_factual = np.mean(factual_scores) if factual_scores else 0
            avg_behavior = np.mean(behavior_scores) if behavior_scores else 0
            avg_performance = np.mean(
                performance_scores) if performance_scores else 0
            avg_search_time = np.mean(search_times) if search_times else 0
            avg_docs_found = np.mean(docs_found) if docs_found else 0
        else:
            avg_factual = avg_behavior = avg_performance = 0
            avg_search_time = avg_docs_found = 0
            total_questions = 0
            passed_count = 0
            pass_rate = 0
            avg_score = 0

        # Extract category-specific metrics
        def get_category_metrics(category_name: str) -> Tuple[float, float]:
            # For quick test, we need to calculate this from the results
            if "results" in result:
                category_results = [r for r in detailed_results if r.get(
                    "category") == category_name]
                if category_results:
                    passed = sum(
                        1 for r in category_results if r.get("passed", False))
                    pass_rate = passed / len(category_results)
                    avg_score = sum(r.get("overall_score", 0)
                                    for r in category_results) / len(category_results)
                    return pass_rate, avg_score

            # For full/short evaluation format
            cat_data = category_breakdown.get(category_name, {})
            return cat_data.get("pass_rate", 0.0), cat_data.get("average_score", 0.0)

        # Use the calculated metrics for quick tests or summary metrics for full evaluations
        metrics = EvaluationMetrics(
            total_questions=total_questions if detailed_results else summary.get(
                "total_questions", 0),
            passed=passed_count if detailed_results else summary.get(
                "passed", 0),
            pass_rate=pass_rate if detailed_results else summary.get(
                "pass_rate", 0.0),
            average_score=avg_score if detailed_results else summary.get(
                "average_score", 0.0),
            avg_factual_accuracy=avg_factual,
            avg_behavior_score=avg_behavior,
            avg_performance_score=avg_performance,
            avg_search_time=avg_search_time,
            avg_documents_found=avg_docs_found,
            evaluation_date=datetime.now().isoformat(),
            notes=notes
        )

        # Add category-specific metrics
        metrics.header_pass_rate, metrics.header_avg_score = get_category_metrics(
            "header")
        metrics.diagnoses_pass_rate, metrics.diagnoses_avg_score = get_category_metrics(
            "diagnoses")
        metrics.procedures_pass_rate, metrics.procedures_avg_score = get_category_metrics(
            "procedures")
        metrics.labs_pass_rate, metrics.labs_avg_score = get_category_metrics(
            "labs")
        metrics.microbiology_pass_rate, metrics.microbiology_avg_score = get_category_metrics(
            "microbiology")
        metrics.prescriptions_pass_rate, metrics.prescriptions_avg_score = get_category_metrics(
            "prescriptions")
        metrics.comprehensive_pass_rate, metrics.comprehensive_avg_score = get_category_metrics(
            "comprehensive")

        return metrics

    def get_results_dataframe(self) -> pd.DataFrame:
        """Generate a pandas DataFrame with all results"""

        if not self.results_data["experiments"]:
            if not self.quiet:
                print("No experiments found. Run some evaluations first!")
            return pd.DataFrame()

        # Convert experiments to DataFrame rows
        rows = []
        for exp_id, exp_data in self.results_data["experiments"].items():
            config = exp_data["config"]
            metrics = exp_data["metrics"]

            row = {
                "experiment_id": exp_id,
                "embedding_model": config["embedding_model_nickname"],
                "llm_model": config["llm_nickname"],
                "embedding_full_name": config["embedding_model_full"],
                "llm_full_name": config["llm_model_full"],
                "experiment_date": exp_data["experiment_date"],
                **metrics  # Unpack all metrics
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure expected metric columns exist for backward compatibility
        expected_columns = [
            "pass_rate", "average_score", "avg_search_time", "avg_documents_found",
            "header_pass_rate", "header_avg_score",
            "diagnoses_pass_rate", "diagnoses_avg_score",
            "procedures_pass_rate", "procedures_avg_score",
            "labs_pass_rate", "labs_avg_score",
            "microbiology_pass_rate", "microbiology_avg_score",
            "prescriptions_pass_rate", "prescriptions_avg_score",
            "comprehensive_pass_rate", "comprehensive_avg_score",
        ]
        for col in expected_columns:
            if col not in df.columns:
                # Default numeric metrics to 0.0
                df[col] = 0.0

        # Save to CSV
        df.to_csv(self.dataframe_file, index=False)
        if not self.quiet:
            print(f" Results dataframe saved to: {self.dataframe_file}")

        return df

    def create_comparison_table(self, save_html: bool = True) -> pd.DataFrame:
        """Create a formatted comparison table"""

        df = self.get_results_dataframe()
        if df.empty:
            return df

        # Create pivot table for easy comparison
        comparison_cols = [
            "embedding_model", "llm_model", "pass_rate", "average_score",
            "avg_factual_accuracy", "avg_behavior_score", "avg_performance_score",
            "avg_search_time"
        ]

        comparison_df = df[comparison_cols].copy()

        # Format percentages and scores
        comparison_df["pass_rate"] = comparison_df["pass_rate"].apply(
            lambda x: f"{x:.1%}")
        comparison_df["average_score"] = comparison_df["average_score"].apply(
            lambda x: f"{x:.3f}")
        comparison_df["avg_factual_accuracy"] = comparison_df["avg_factual_accuracy"].apply(
            lambda x: f"{x:.3f}")
        comparison_df["avg_behavior_score"] = comparison_df["avg_behavior_score"].apply(
            lambda x: f"{x:.3f}")
        comparison_df["avg_performance_score"] = comparison_df["avg_performance_score"].apply(
            lambda x: f"{x:.3f}")
        comparison_df["avg_search_time"] = comparison_df["avg_search_time"].apply(
            lambda x: f"{x:.2f}s")

        # Rename columns for display
        comparison_df.columns = [
            "Embedding Model", "LLM Model", "Pass Rate", "Avg Score",
            "Factual Acc.", "Behavior", "Performance", "Search Time"
        ]

        if save_html:
            html_file = self.results_dir / "comparison_table.html"
            comparison_df.to_html(html_file, index=False, table_id="comparison_table",
                                  classes="table table-striped table-hover")
            if not self.quiet:
                print(f" Comparison table saved to: {html_file}")

        return comparison_df

    def create_heatmap(self, metric: str = "average_score", save_fig: bool = True) -> None:
        """Create a heatmap showing performance across model combinations"""

        df = self.get_results_dataframe()
        if df.empty:
            if not self.quiet:
                print("No data available for heatmap")
            return

        if not save_fig:
            return

        # Delegate to visualization module for plotting and saving
        try:
            visualizer = EvaluationVisualizer(output_dir=self.results_dir)
            html_path = visualizer.create_model_comparison_heatmap(
                df, metric=metric)
            if html_path and not self.quiet:
                print(f" Heatmap HTML saved to: {html_path}")
        except Exception as e:
            if not self.quiet:
                print(f" Error creating heatmap: {str(e)}")

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""

        df = self.get_results_dataframe()
        if df.empty:
            return "No evaluation results available."

        # Find best performing combinations
        best_overall = df.loc[df["average_score"].idxmax()]
        best_pass_rate = df.loc[df["pass_rate"].idxmax()]
        fastest = df.loc[df["avg_search_time"].idxmin()]

        report = f"""
# Clinical RAG Model Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Experiments**: {len(df)}
- **Embedding Models Tested**: {df['embedding_model'].nunique()}
- **LLM Models Tested**: {df['llm_model'].nunique()}

## Best Performing Combinations

### Highest Overall Score
- **Models**: {best_overall['embedding_model']} + {best_overall['llm_model']}
- **Score**: {best_overall['average_score']:.3f}
- **Pass Rate**: {best_overall['pass_rate']:.1%}

### Highest Pass Rate
- **Models**: {best_pass_rate['embedding_model']} + {best_pass_rate['llm_model']}
- **Pass Rate**: {best_pass_rate['pass_rate']:.1%}
- **Score**: {best_pass_rate['average_score']:.3f}

### Fastest Performance
- **Models**: {fastest['embedding_model']} + {fastest['llm_model']}
- **Search Time**: {fastest['avg_search_time']:.2f}s
- **Score**: {fastest['average_score']:.3f}

## Performance Statistics
- **Average Pass Rate**: {df['pass_rate'].mean():.1%}
- **Average Score**: {df['average_score'].mean():.3f}
- **Average Search Time**: {df['avg_search_time'].mean():.2f}s

## Category Performance Breakdown
- **Header Questions**: {df['header_pass_rate'].mean():.1%} pass rate
- **Diagnoses**: {df['diagnoses_pass_rate'].mean():.1%} pass rate
- **Procedures**: {df['procedures_pass_rate'].mean():.1%} pass rate
- **Labs**: {df['labs_pass_rate'].mean():.1%} pass rate
- **Microbiology**: {df['microbiology_pass_rate'].mean():.1%} pass rate
- **Prescriptions**: {df['prescriptions_pass_rate'].mean():.1%} pass rate
- **Comprehensive**: {df['comprehensive_pass_rate'].mean():.1%} pass rate

"""

        # Save report
        report_file = self.results_dir / "summary_report.md"
        with open(report_file, 'w') as f:
            f.write(report)

        if not self.quiet:
            print(f"📄 Summary report saved to: {report_file}")
        return report

    def list_model_combinations(self) -> None:
        """List all possible model combinations"""
        if self.quiet:
            return
        print("🔄 Available Model Combinations:")
        print("=" * 50)
        for i, config in enumerate(self.model_configs, 1):
            print(f"{i:2d}. {config.model_nickname} + {config.llm_nickname}")
            print(f"    Embedding: {config.embedding_model}")
            print(f"    LLM: {config.llm_model}")
            print(f"    Vectorstore: {config.vectorstore_name}")
            print()

    def quick_add_sample_results(self):
        """Add some sample results for testing"""
        # This is for testing the system
        sample_result = {
            "summary": {
                "total_questions": 10,
                "passed": 8,
                "pass_rate": 0.8,
                "average_score": 0.75,
                "category_breakdown": {
                    "header": {"count": 2, "pass_rate": 0.9, "average_score": 0.82},
                    "diagnoses": {"count": 3, "pass_rate": 0.67, "average_score": 0.72},
                    "labs": {"count": 2, "pass_rate": 1.0, "average_score": 0.85},
                    "prescriptions": {"count": 3, "pass_rate": 0.67, "average_score": 0.68}
                }
            },
            "detailed_results": [
                {"factual_accuracy_score": 0.8, "behavior_score": 0.9, "performance_score": 0.95,
                 "search_time": 1.2, "documents_found": 5, "overall_score": 0.83, "passed": True},
                {"factual_accuracy_score": 0.7, "behavior_score": 0.8, "performance_score": 0.9,
                 "search_time": 2.1, "documents_found": 4, "overall_score": 0.75, "passed": True}
            ]
        }

        # Add sample results for a few combinations
        self.add_evaluation_result(
            "ms-marco", "deepseek", sample_result, "Sample test run")
        if not self.quiet:
            print(" Sample results added for testing")


# Convenience functions for easy usage

def load_results_manager(quiet: bool = False) -> EvaluationResultsManager:
    """Load the results manager"""
    return EvaluationResultsManager(quiet=quiet)


def add_result(embedding_model: str, llm_model: str, eval_result: Dict, notes: str = "", quiet: bool = False) -> str:
    """Quick function to add a result"""
    manager = load_results_manager(quiet=quiet)
    return manager.add_evaluation_result(embedding_model, llm_model, eval_result, notes)


def get_comparison_table(quiet: bool = False) -> pd.DataFrame:
    """Quick function to get comparison table"""
    manager = load_results_manager(quiet=quiet)
    return manager.create_comparison_table()


def generate_report(quiet: bool = False) -> str:
    """Quick function to generate report"""
    manager = load_results_manager(quiet=quiet)
    return manager.generate_summary_report()


if __name__ == "__main__":
    # Example usage
    manager = EvaluationResultsManager()

    print("Clinical RAG Evaluation Results Manager")
    print("=" * 40)

    manager.list_model_combinations()

    # Add sample data for demonstration
    print("Adding sample results for demonstration...")
    manager.quick_add_sample_results()

    # Generate reports
    df = manager.get_results_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print("\nComparison table:")
    comparison = manager.create_comparison_table()
    print(comparison)

    # Generate summary report
    report = manager.generate_summary_report()
    print(report)
