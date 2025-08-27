"""
Centralized Evaluation Results Manager for Clinical RAG System
Handles evaluation execution, storage, analysis, visualization, and report generation.
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import argparse
import sys
import logging

from RAG_chat_pipeline.config.config import (
    model_names, llms
)
# Optional import: handle environments without langchain
try:
    from langchain.schema import Document  # type: ignore
except Exception:
    Document = None  # sentinel, check before isinstance

# Set up visualization style
plt.style.use('default')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Inlined metrics normalization to avoid external dependency on utils.metrics ---
REQUIRED_NUMERIC_COLS = [
    "precision",
    "recall",
    "f1_score",
    "search_time",
]

FALLBACK_MAP = {
    # canonical: [alternatives...]
    "search_time": ["avg_search_time"],
    "precision": ["factual_accuracy_score"],  # Legacy compatibility
    "recall": ["context_relevance_score"],    # Legacy compatibility
    "f1_score": ["semantic_similarity_score"],  # Legacy compatibility
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a results DataFrame to have consistent schema.

    - Fills canonical columns from known alternatives
    - Adds missing numeric columns with defaults
    - Ensures presence of expected categorical fields
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # 1) Fill canonical columns from alternatives
    for canonical, alts in FALLBACK_MAP.items():
        if canonical not in df.columns:
            for alt in alts:
                if alt in df.columns:
                    df[canonical] = df[alt]
                    break
        if canonical not in df.columns:
            # Default numeric 0.0 for missing metrics
            df[canonical] = 0.0

    # 2) Ensure required numeric columns exist
    for col in REQUIRED_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # 3) Ensure expected identifiers exist
    if 'category' not in df.columns:
        df['category'] = 'default'

    if 'embedding_model' not in df.columns:
        df['embedding_model'] = df.get('embedding', 'unknown')

    if 'llm_model' not in df.columns:
        df['llm_model'] = df.get('llm', 'unknown')

    return df


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
    """Simplified evaluation metrics focused on core semantic scores"""
    # Basic metrics
    total_questions: int

    # Core semantic scoring
    avg_precision: float
    avg_recall: float
    avg_f1_score: float

    # Performance metrics
    avg_search_time: float = 0.0
    avg_documents_found: float = 0.0

    # Additional metadata
    evaluation_date: str = ""
    notes: str = ""


class EvaluationResultsManager:
    """Centralized manager for RAG evaluation, results processing, and report generation."""

    def __init__(self, results_dir: Path = None, quiet: bool = False):
        self.quiet = quiet
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Directory structure
        if results_dir is None:
            current_dir = Path(__file__).resolve().parent
            # Prefer repository root (which contains top-level 'report') when available
            repo_root = current_dir.parent.parent.parent
            candidate_report_dir = repo_root / "report"
            if candidate_report_dir.exists():
                self.base_dir = repo_root
            else:
                # Fallback to package root
                self.base_dir = current_dir.parent.parent
        else:
            self.base_dir = results_dir.resolve() if not results_dir.is_absolute() else results_dir

        # Results directory (for JSON/CSV data)
        self.results_dir = self.base_dir / "results"

        # Report directories (for images/tables used in LaTeX)
        self.report_dir = self.base_dir / "report" / "chap4_results"
        self.figures_dir = self.report_dir / "images"
        self.tables_dir = self.report_dir / "tables"

        # Create all directories
        for directory in [self.results_dir, self.figures_dir, self.tables_dir]:
            directory.mkdir(exist_ok=True, parents=True)

        # Main results files
        self.results_file = self.results_dir / "model_comparison_results.json"
        # Separate file for summary stats to avoid schema conflicts with experiments JSON
        self.summary_file = self.results_dir / "summary_stats.json"
        self.dataframe_file = self.results_dir / "results_dataframe.csv"
        self.per_question_file = self.results_dir / "per_question_results.csv"
        self.efficiency_file = self.results_dir / "run_efficiency_and_safety.csv"
        self.heatmap_csv_file = self.results_dir / "heatmap_average_score_matrix.csv"

        # Load existing results or initialize empty
        self.results_data = self._load_existing_results()

        # Results storage for current session
        self.current_results = []
        self.processed_df = None
        self.summary_stats = {}

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
        if (Document is not None) and isinstance(obj, Document):
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
        elif (Document is not None) and isinstance(result, Document):
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
            print(f"    F1-Score: {metrics.avg_f1_score:.3f}")
            print(
                f"    Precision: {metrics.avg_precision:.3f}, Recall: {metrics.avg_recall:.3f}")

        return experiment_id

    def _extract_metrics_from_result(self, result: Dict, notes: str = "") -> EvaluationMetrics:
        """Extract structured metrics from evaluation result"""

        summary = result.get("summary", {})

        # Check if using quick test format or full/short evaluation format
        if "results" in result:  # Quick test format
            detailed_results = result.get("results", [])
        else:  # Full/short evaluation format
            detailed_results = result.get("detailed_results", [])

        # Calculate average component scores
        if detailed_results:
            precision_scores = [r.get("precision", 0)
                                for r in detailed_results]
            recall_scores = [r.get("recall", 0)
                             for r in detailed_results]
            f1_scores = [r.get("f1_score", 0)
                         for r in detailed_results]
            search_times = [r.get("search_time", 0) for r in detailed_results]
            docs_found = [r.get("documents_found", 0)
                          for r in detailed_results]

            total_questions = len(detailed_results)
            avg_precision = np.mean(
                precision_scores) if precision_scores else 0
            avg_recall = np.mean(recall_scores) if recall_scores else 0
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            avg_search_time = np.mean(search_times) if search_times else 0
            avg_docs_found = np.mean(docs_found) if docs_found else 0
        else:
            avg_precision = avg_recall = avg_f1 = 0
            avg_search_time = avg_docs_found = 0
            total_questions = 0

        # Create simplified metrics
        metrics = EvaluationMetrics(
            total_questions=total_questions,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1_score=avg_f1,
            avg_search_time=avg_search_time,
            avg_documents_found=avg_docs_found,
            evaluation_date=datetime.now().isoformat(),
            notes=notes
        )

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
            "avg_search_time", "avg_documents_found",
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
            "embedding_model", "llm_model",
            "avg_precision", "avg_recall", "avg_f1_score",
            "avg_search_time"
        ]

        comparison_df = df[comparison_cols].copy()

        # Format scores
        comparison_df["avg_precision"] = comparison_df["avg_precision"].apply(
            lambda x: f"{x:.3f}")
        comparison_df["avg_recall"] = comparison_df["avg_recall"].apply(
            lambda x: f"{x:.3f}")
        comparison_df["avg_f1_score"] = comparison_df["avg_f1_score"].apply(
            lambda x: f"{x:.3f}")
        comparison_df["avg_search_time"] = comparison_df["avg_search_time"].apply(
            lambda x: f"{x:.2f}s")

        # Rename columns for display
        comparison_df.columns = [
            "Embedding Model", "LLM Model",
            "Precision", "Recall", "F1-Score", "Search Time"
        ]

        if save_html:
            html_file = self.results_dir / "comparison_table.html"
            comparison_df.to_html(html_file, index=False, table_id="comparison_table",
                                  classes="table table-striped table-hover")
            if not self.quiet:
                print(f" Comparison table saved to: {html_file}")

        return comparison_df

    def create_heatmap(self, metric: str = "avg_f1_score", save_fig: bool = True) -> None:
        """Create a heatmap showing performance across model combinations"""

        df = self.get_results_dataframe()
        if df.empty:
            if not self.quiet:
                print("No data available for heatmap")
            return

        if not save_fig:
            return

        # Use dedicated visualization module
        try:
            from .visualization import EvaluationVisualizer
            visualizer = EvaluationVisualizer(output_dir=self.results_dir)
            html_path = visualizer.create_model_comparison_heatmap(
                df, metric=metric)
            if html_path and not self.quiet:
                print(f" Heatmap HTML saved to: {html_path}")
        except Exception as e:
            if not self.quiet:
                print(f" Error creating heatmap: {str(e)}")

    def run_complete_evaluation(self,
                                embedding_models: Optional[List[str]] = None,
                                llm_models: Optional[List[str]] = None,
                                quick_test: bool = False,
                                generate_reports: bool = True) -> Dict[str, Any]:
        """Run complete evaluation pipeline with all outputs."""

        logger.info(
            f"Starting centralized evaluation pipeline at {datetime.now()}")

        if embedding_models is None:
            embedding_models = list(model_names.keys())
        if llm_models is None:
            llm_models = list(llms.keys())

        # Step 1: Run evaluations
        logger.info("Step 1: Running model evaluations...")
        evaluation_results = self._run_model_evaluations(
            embedding_models, llm_models, quick_test
        )

        # Step 2: Process and enrich results
        logger.info("Step 2: Processing and enriching results...")
        self.processed_df = self._process_and_enrich_results(
            evaluation_results)

        # Step 3: Generate summary statistics
        logger.info("Step 3: Generating summary statistics...")
        self.summary_stats = self._generate_summary_statistics()

        # Step 4: Save all result formats
        logger.info("Step 4: Saving results in multiple formats...")
        file_paths = self._save_all_results()

        # Step 5: Generate visualizations and tables
        if generate_reports:
            logger.info(
                "Step 5: Generating visualizations and report tables...")
            report_paths = self._generate_all_reports()
            file_paths.update(report_paths)

        logger.info(f"Evaluation pipeline completed at {datetime.now()}")

        return {
            "summary_stats": self.summary_stats,
            "file_paths": file_paths,
            "timestamp": self.timestamp,
            "models_evaluated": {
                "embedding_models": embedding_models,
                "llm_models": llm_models
            }
        }

    def _run_model_evaluations(self, embedding_models: List[str],
                               llm_models: List[str], quick_test: bool) -> List[Dict]:
        """Run evaluations for all model combinations."""
        all_results: List[Dict] = []

        # Get gold questions from generator
        from .gold_questions import generate_gold_questions_from_data
        from pathlib import Path
        default_n = 5 if quick_test else 20

        # Ensure we use the correct dataset directory
        dataset_dir = Path.cwd() / "mimic_sample_1000"
        questions = generate_gold_questions_from_data(
            num_questions=default_n,
            save_to_file=False,
            dataset_dir=str(dataset_dir) if dataset_dir.exists() else None)

        logger.info(
            f"Evaluating {len(questions)} questions across {len(embedding_models)} embedding models and {len(llm_models)} LLM models")

        total_combinations = len(embedding_models) * len(llm_models)
        current_combination = 0

        for emb_model in embedding_models:
            for llm_model in llm_models:
                current_combination += 1
                logger.info(
                    f"Progress: {current_combination}/{total_combinations} - Testing {emb_model} + {llm_model}")

                try:
                    # Run evaluation for this model combination
                    from .rag_evaluator import ClinicalRAGEvaluator
                    from RAG_chat_pipeline.core.main import main as initialize_clinical_rag
                    from RAG_chat_pipeline.config.config import set_models

                    # Set models and initialize
                    set_models(emb_model, llm_model)
                    chatbot = initialize_clinical_rag()
                    evaluator = ClinicalRAGEvaluator(chatbot)

                    # Run evaluation batch
                    results: List[Dict] = []
                    for i, question in enumerate(questions):
                        try:
                            result = evaluator.evaluate_question(
                                question, f"{emb_model}_{llm_model}_{i}")
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Failed question {i}: {e}")
                            continue

                    # Add model metadata to each result
                    for result in results:
                        result.update({
                            "embedding_model": emb_model,
                            "llm_model": llm_model,
                            "evaluation_timestamp": self.timestamp
                        })

                    all_results.extend(results)

                    # Also add to the manager for compatibility
                    if results:
                        self.add_evaluation_result(emb_model, llm_model,
                                                   {"results": results}, "Centralized evaluation run")

                except Exception as e:
                    logger.error(
                        f"Failed evaluation for {emb_model} + {llm_model}: {e}")
                    continue

        logger.info(f"Completed {len(all_results)} individual evaluations")
        self.current_results = all_results
        return all_results

    def _process_and_enrich_results(self, raw_results: List[Dict]) -> pd.DataFrame:
        """Process raw results and add enriched metrics."""
        df = pd.DataFrame(raw_results)

        if df.empty:
            logger.warning("No results to process!")
            return df

        # Normalize schema/metrics to ensure consistent columns
        try:
            df = normalize_dataframe(df)
        except Exception as norm_err:
            logger.warning(f"Normalization skipped: {norm_err}")

        # Add derived metrics
        df['model_combination'] = df['embedding_model'] + ' + ' + df['llm_model']
        # All obsolete metric calculations removed - semantic evaluation uses only precision/recall/F1
        df['quality_tier'] = self._categorize_quality(df)

        # Add performance categories
        df['search_time_category'] = pd.cut(
            df.get('search_time', pd.Series([1.0] * len(df))),
            bins=[0, 2, 5, float('inf')],
            labels=['Fast', 'Moderate', 'Slow']
        )

        logger.info(f"Processed {len(df)} results with enriched metrics")
        return df

    def _determine_pass_status(self, row) -> bool:
        """Deprecated - pass/fail logic removed from semantic evaluation."""
        return True  # Always return True since we use raw F1-scores now

    # Obsolete efficiency calculation removed - semantic evaluation focuses on core metrics

    def _categorize_quality(self, df: pd.DataFrame) -> pd.Series:
        """Categorize results into quality tiers based on F1-score."""
        f1_scores = df.get('f1_score', pd.Series([0.5] * len(df)))
        conditions = [
            f1_scores >= 0.8,
            f1_scores >= 0.6,
            f1_scores >= 0.4
        ]
        choices = ['High', 'Medium', 'Low']
        return pd.Series(np.select(conditions, choices, default='Very Low'), index=df.index)

    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        if self.processed_df is None or self.processed_df.empty:
            return {}

        df = self.processed_df

        # Overall statistics
        overall_stats = {
            "total_evaluations": len(df),
            "unique_model_combinations": df['model_combination'].nunique(),
            "avg_precision": df.get('precision', pd.Series([0])).mean(),
            "avg_recall": df.get('recall', pd.Series([0])).mean(),
            "avg_f1_score": df.get('f1_score', pd.Series([0])).mean(),
            "average_search_time": df.get('search_time', pd.Series([1.0])).mean()
        }

        # Model performance rankings
        model_rankings = df.groupby('model_combination').agg({
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'search_time': 'mean'
        }).round(4).sort_values('f1_score', ascending=False)

        # Category performance
        if 'category' in df.columns:
            category_stats = df.groupby('category').agg({
                'f1_score': 'mean',
                'precision': 'mean',
                'recall': 'mean',
                'search_time': 'mean'
            }).round(4)
        else:
            category_stats = pd.DataFrame()

        # Top performers
        top_performers = model_rankings.head(5).to_dict('index')

        return {
            "overall": overall_stats,
            "model_rankings": model_rankings.to_dict('index'),
            "category_performance": category_stats.to_dict('index') if not category_stats.empty else {},
            "top_performers": top_performers,
            "evaluation_metadata": {
                "timestamp": self.timestamp,
                "total_questions": len(df),
                "categories_tested": df.get('category', pd.Series(['default'])).unique().tolist()
            }
        }

    def _save_all_results(self) -> Dict[str, str]:
        """Save results in all required formats."""
        file_paths = {}

        if self.processed_df is not None and not self.processed_df.empty:
            # 1. Complete results CSV
            self.processed_df.to_csv(self.dataframe_file, index=False)
            file_paths["results_csv"] = str(self.dataframe_file)

            # 2. Per-question results CSV
            question_cols = ['question', 'category', 'model_combination', 'search_time']
            # Add available columns
            for col in ['f1_score', 'precision', 'recall', 'answer']:
                if col in self.processed_df.columns:
                    question_cols.append(col)
            question_results = self.processed_df[question_cols].copy()
            question_results.to_csv(self.per_question_file, index=False)
            file_paths["per_question_csv"] = str(self.per_question_file)

            # 3. Summary statistics JSON (avoid overwriting experiments JSON)
            with open(self.summary_file, 'w') as f:
                json.dump(self.summary_stats, f, indent=2, default=str)
            file_paths["summary_json"] = str(self.summary_file)

            # 4. Efficiency and safety CSV
            efficiency_cols = {}
            # Only include columns that exist
            for col in ['f1_score', 'precision', 'recall', 'search_time']:
                if col in self.processed_df.columns:
                    efficiency_cols[col] = 'mean'
            
            if efficiency_cols:
                efficiency_data = self.processed_df.groupby('model_combination').agg(efficiency_cols).round(4)
                efficiency_data.to_csv(self.efficiency_file)
                file_paths["efficiency_csv"] = str(self.efficiency_file)

            logger.info(f"Saved results to {len(file_paths)} files")

        return file_paths

    def _generate_all_reports(self) -> Dict[str, str]:
        """Generate all visualization and report outputs using dedicated modules."""
        file_paths = {}

        if self.processed_df is None or self.processed_df.empty:
            logger.warning(
                "No processed results available for report generation")
            return file_paths

        try:
            # 1. Generate visualizations using dedicated visualization module
            try:
                from .visualization import EvaluationVisualizer
                visualizer = EvaluationVisualizer(output_dir=self.figures_dir)
                viz_files = visualizer.generate_all_visualizations(
                    self.processed_df)
                file_paths.update(viz_files)
            except Exception as viz_err:
                logger.warning(f"Visualization generation failed: {viz_err}")

            # 2. Generate reports using dedicated reporting module
            try:
                from .reporting import EvaluationReporter
                reporter = EvaluationReporter(output_dir=self.results_dir)
                report_file = reporter.save_summary_report(self.processed_df)
                file_paths["summary_report"] = str(report_file)
            except Exception as report_err:
                logger.warning(f"Report generation failed: {report_err}")

            # 2. Generate model comparison plots
            comparison_plots = self._generate_model_comparison_plots()
            file_paths.update(comparison_plots)

            # 3. Generate performance tables for LaTeX
            table_paths = self._generate_latex_tables()
            file_paths.update(table_paths)

            # 4. Generate quality vs throughput plot
            quality_plot = self._generate_quality_throughput_plot()
            if quality_plot:
                file_paths["quality_throughput_plot"] = quality_plot

            logger.info(f"Generated {len(file_paths)} report files")

        except Exception as e:
            logger.error(f"Error generating reports: {e}")

        return file_paths

    # Deprecated: internal seaborn heatmap generation removed in favor of visualization module

    def _generate_model_comparison_plots(self) -> Dict[str, str]:
        """Generate model comparison visualizations for LaTeX report."""
        plots = {}

        try:
            # Calculate model statistics
            df = self.processed_df.copy()
            # Map f1_score to overall_score for compatibility
            if 'f1_score' in df.columns:
                df['overall_score'] = df['f1_score']
            if 'search_time' not in df.columns and 'avg_search_time' in df.columns:
                df['search_time'] = df['avg_search_time']
            
            # Create pass_status based on f1_score threshold
            if 'f1_score' in df.columns:
                df['pass_status'] = (df['f1_score'] >= 0.6).astype(int)
            
            # Only aggregate columns that exist
            agg_cols = {}
            for col in ['overall_score', 'pass_status', 'search_time']:
                if col in df.columns:
                    agg_cols[col] = 'mean'
            
            model_stats = df.groupby('model_combination').agg(agg_cols)

            # Plot 1: Pass Rate vs Score
            plt.figure(figsize=(12, 8))
            plt.scatter(model_stats['overall_score'],
                        model_stats['pass_status'], s=100, alpha=0.7)
            for idx, row in model_stats.iterrows():
                plt.annotate(idx, (row['overall_score'], row['pass_status']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.xlabel('Average Score')
            plt.ylabel('Pass Rate')
            plt.title('Model Performance: Pass Rate vs Average Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            pass_rate_path = self.figures_dir / "pass_rate_vs_score.png"
            plt.savefig(pass_rate_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots["pass_rate_plot"] = str(pass_rate_path)

            # Plot 2: Time vs Score
            plt.figure(figsize=(12, 8))
            plt.scatter(model_stats['search_time'],
                        model_stats['overall_score'], s=100, alpha=0.7)

            for idx, row in model_stats.iterrows():
                plt.annotate(idx, (row['search_time'], row['overall_score']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.xlabel('Average Search Time (seconds)')
            plt.ylabel('Average Score')
            plt.title('Model Performance: Score vs Response Time')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            time_score_path = self.figures_dir / "time_vs_score.png"
            plt.savefig(time_score_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots["time_score_plot"] = str(time_score_path)

        except Exception as e:
            logger.error(f"Failed to generate model comparison plots: {e}")

        return plots

    def _generate_quality_throughput_plot(self) -> Optional[str]:
        """Generate quality vs throughput analysis for LaTeX report."""
        try:
            plt.figure(figsize=(12, 8))

            # Calculate throughput (inverse of search time)
            df = self.processed_df.copy()
            # Map f1_score to overall_score for compatibility
            if 'f1_score' in df.columns:
                df['overall_score'] = df['f1_score']
            if 'search_time' not in df.columns and 'avg_search_time' in df.columns:
                df['search_time'] = df['avg_search_time']
            
            # Only aggregate columns that exist
            agg_cols = {}
            for col in ['overall_score', 'search_time']:
                if col in df.columns:
                    agg_cols[col] = 'mean'
            
            model_stats = df.groupby('model_combination').agg(agg_cols)

            # Guard against division by zero
            safe_search_time = model_stats['search_time'].replace(0, np.nan)
            model_stats['throughput'] = 1 / \
                safe_search_time  # queries per second

            # Color by overall score (since efficiency_score doesn't exist)
            scatter = plt.scatter(model_stats['throughput'], model_stats['overall_score'],
                                  c=model_stats['overall_score'], s=100, alpha=0.7,
                                  cmap='viridis')

            for idx, row in model_stats.iterrows():
                plt.annotate(idx, (row['throughput'], row['overall_score']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.colorbar(scatter, label='Quality Score (F1)')
            plt.xlabel('Throughput (Queries/Second)')
            plt.ylabel('Average Quality Score')
            plt.title('Model Performance: Quality vs Throughput Trade-off')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            quality_plot_path = self.figures_dir / "quality_vs_throughput.png"
            plt.savefig(quality_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(quality_plot_path)

        except Exception as e:
            logger.error(f"Failed to generate quality vs throughput plot: {e}")
            return None

    def _generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for the report."""
        tables = {}

        try:
            # Table 1: Top performer rankings
            if 'model_rankings' in self.summary_stats:
                rankings_df = pd.DataFrame(
                    self.summary_stats['model_rankings']).T
                rankings_df = rankings_df.head(10).round(3)

                # Rename columns for better LaTeX output (only use existing columns)
                col_map = {
                    'f1_score': 'F1 Score',
                    'precision': 'Precision', 
                    'recall': 'Recall',
                    'search_time': 'Time (s)'
                }
                rankings_df = rankings_df.rename(columns={k: v for k, v in col_map.items() if k in rankings_df.columns})
                rankings_df.index.name = 'Model Combination'

                latex_table = rankings_df.to_latex(
                    float_format="{:.3f}".format,
                    caption="Top 10 Model Performance Rankings",
                    label="tab:model_rankings",
                    escape=False
                )

                rankings_path = self.tables_dir / "enhanced_top_performers.tex"
                with open(rankings_path, 'w') as f:
                    f.write(latex_table)
                tables["rankings_table"] = str(rankings_path)

            # Table 2: System statistics summary
            if 'overall' in self.summary_stats:
                overall_stats = self.summary_stats['overall']
                stats_data = {
                    'Metric': ['Total Evaluations', 'Model Combinations', 'Average Precision',
                               'Average Recall', 'Average F1-Score', 'Average Response Time'],
                    'Value': [
                        overall_stats['total_evaluations'],
                        overall_stats['unique_model_combinations'],
                        f"{overall_stats['avg_precision']:.3f}",
                        f"{overall_stats['avg_recall']:.3f}",
                        f"{overall_stats['avg_f1_score']:.3f}",
                        f"{overall_stats['average_search_time']:.2f}s"
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                stats_latex = stats_df.to_latex(
                    index=False,
                    caption="Evaluation System Statistics Summary",
                    label="tab:system_stats",
                    escape=False
                )

                stats_path = self.tables_dir / "system_statistics.tex"
                with open(stats_path, 'w') as f:
                    f.write(stats_latex)
                tables["stats_table"] = str(stats_path)

            # Table 3: Embedding model ranking (for compatibility)
            if not self.processed_df.empty:
                # Only use columns that exist
                agg_cols = {}
                col_names = []
                if 'f1_score' in self.processed_df.columns:
                    agg_cols['f1_score'] = 'mean'
                    col_names.append('F1 Score')
                if 'precision' in self.processed_df.columns:
                    agg_cols['precision'] = 'mean' 
                    col_names.append('Precision')
                if 'search_time' in self.processed_df.columns:
                    agg_cols['search_time'] = 'mean'
                    col_names.append('Search Time (s)')
                
                if agg_cols:
                    embedding_stats = self.processed_df.groupby('embedding_model').agg(agg_cols).round(3)
                    embedding_stats.columns = col_names
                    # Sort by first column (likely F1 Score or Precision)
                    embedding_stats = embedding_stats.sort_values(
                        embedding_stats.columns[0], ascending=False)
                    
                    embedding_latex = embedding_stats.to_latex(
                        caption="Embedding Model Performance Ranking",
                        label="tab:embedding_ranking",
                        escape=False
                    )

                    embedding_path = self.tables_dir / "embedding_ranking.tex"
                    with open(embedding_path, 'w') as f:
                        f.write(embedding_latex)
                    tables["embedding_table"] = str(embedding_path)

        except Exception as e:
            logger.error(f"Failed to generate performance tables: {e}")

        return tables

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""

        df = self.get_results_dataframe()
        if df.empty:
            return "No evaluation results available."

        # Find best performing combinations
        best_f1 = df.loc[df["avg_f1_score"].idxmax()]
        best_precision = df.loc[df["avg_precision"].idxmax()]
        best_recall = df.loc[df["avg_recall"].idxmax()]
        fastest = df.loc[df["avg_search_time"].idxmin()]

        report = f"""
# Clinical RAG Model Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Experiments**: {len(df)}
- **Embedding Models Tested**: {df['embedding_model'].nunique()}
- **LLM Models Tested**: {df['llm_model'].nunique()}

## Best Performing Combinations

### Highest F1-Score
- **Models**: {best_f1['embedding_model']} + {best_f1['llm_model']}
- **F1-Score**: {best_f1['avg_f1_score']:.3f}
- **Precision**: {best_f1['avg_precision']:.3f}
- **Recall**: {best_f1['avg_recall']:.3f}

### Highest Precision
- **Models**: {best_precision['embedding_model']} + {best_precision['llm_model']}
- **Precision**: {best_precision['avg_precision']:.3f}
- **F1-Score**: {best_precision['avg_f1_score']:.3f}

### Highest Recall
- **Models**: {best_recall['embedding_model']} + {best_recall['llm_model']}
- **Recall**: {best_recall['avg_recall']:.3f}
- **F1-Score**: {best_recall['avg_f1_score']:.3f}

### Fastest Performance
- **Models**: {fastest['embedding_model']} + {fastest['llm_model']}
- **Search Time**: {fastest['avg_search_time']:.2f}s
- **F1-Score**: {fastest['avg_f1_score']:.3f}

## Performance Statistics
- **Average Precision**: {df['avg_precision'].mean():.3f}
- **Average Recall**: {df['avg_recall'].mean():.3f}
- **Average F1-Score**: {df['avg_f1_score'].mean():.3f}
- **Average Search Time**: {df['avg_search_time'].mean():.2f}s

## Semantic Evaluation Metrics
This evaluation uses BioBERT-based semantic similarity for medical concept matching:
- **Precision**: Accuracy of retrieved medical information
- **Recall**: Completeness of expected medical concepts found  
- **F1-Score**: Harmonic mean balancing precision and recall

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
                "avg_precision": 0.78,
                "avg_recall": 0.82,
                "avg_f1_score": 0.80,
                "avg_search_time": 1.5
            },
            "detailed_results": [
                {"precision": 0.8, "recall": 0.9, "f1_score": 0.84,
                 "search_time": 1.2, "documents_found": 5, "category": "diagnoses"},
                {"precision": 0.7, "recall": 0.8, "f1_score": 0.74,
                 "search_time": 2.1, "documents_found": 4, "category": "labs"}
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
    """Quick function to generate report using dedicated reporting module"""
    try:
        from .reporting import EvaluationReporter

        manager = load_results_manager(quiet=quiet)
        df = manager.get_results_dataframe()

        reporter = EvaluationReporter(output_dir=manager.results_dir)
        report_file = reporter.save_summary_report(df)

        if not quiet:
            print(f"Report saved to: {report_file}")

        return reporter.generate_summary_report(df)

    except Exception as e:
        error_msg = f"Failed to generate report: {e}"
        if not quiet:
            print(error_msg)
        return error_msg


# ============================================================================
# CONVENIENCE FUNCTIONS AND CLI INTERFACE
# ============================================================================

def run_complete_evaluation(embedding_models: Optional[List[str]] = None,
                            llm_models: Optional[List[str]] = None,
                            quick_test: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run the complete evaluation pipeline.
    Replaces run_complete_evaluation.py functionality.

    Args:
        embedding_models: List of embedding models to test
        llm_models: List of LLM models to test
        quick_test: Whether to run a quick test with limited questions

    Returns:
        Complete results dictionary with all file paths and statistics
    """
    manager = EvaluationResultsManager()
    return manager.run_complete_evaluation(
        embedding_models=embedding_models,
        llm_models=llm_models,
        quick_test=quick_test,
        generate_reports=True
    )


def main():
    """CLI interface for the evaluation system."""
    parser = argparse.ArgumentParser(
        description="Clinical RAG Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with default models
  python -m RAG_chat_pipeline.benchmarks.evaluation_results_manager

  # Quick test with specific models
  python -m RAG_chat_pipeline.benchmarks.evaluation_results_manager --quick --embedding mini-lm,biomedbert --llm tinyllama,qwen

  # Generate reports only from existing data
  python -m RAG_chat_pipeline.benchmarks.evaluation_results_manager --reports-only
        """
    )

    parser.add_argument(
        '--embedding',
        type=str,
        help='Comma-separated list of embedding models to test'
    )

    parser.add_argument(
        '--llm',
        type=str,
        help='Comma-separated list of LLM models to test'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with limited questions (for development)'
    )

    parser.add_argument(
        '--reports-only',
        action='store_true',
        help='Generate reports from existing results (skip evaluation)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output messages'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Clinical RAG System - Centralized Evaluation Pipeline")
    print("=" * 70)
    print()

    # Handle list models
    if args.list_models:
        print("Available Embedding Models:")
        for i, (key, info) in enumerate(model_names.items(), 1):
            print(f"  {i:2d}. {key:12s} - {info[0]}")

        print("\nAvailable LLM Models:")
        for i, (key, model) in enumerate(llms.items(), 1):
            print(f"  {i:2d}. {key:12s} - {model}")
        return

    # Initialize manager
    manager = EvaluationResultsManager(quiet=args.quiet)

    # Handle reports-only mode
    if args.reports_only:
        print("Generating reports from existing results...")
        if manager.processed_df is None or manager.processed_df.empty:
            # Try to load existing data
            df = manager.get_results_dataframe()
            if df.empty:
                print("!! No existing results found. Run evaluation first.")
                return 1

            # Convert existing data to processed format
            try:
                manager.processed_df = normalize_dataframe(df)
            except Exception:
                manager.processed_df = df
            manager.summary_stats = manager._generate_summary_statistics()

        report_paths = manager._generate_all_reports()
        print(f"✅ Generated {len(report_paths)} report files")
        for file_type, path in report_paths.items():
            print(f"   {file_type:20s}: {path}")
        return 0

    # Parse model arguments
    embedding_models = None
    llm_models = None

    if args.embedding:
        embedding_models = [m.strip() for m in args.embedding.split(',')]
        # Validate models
        invalid = [m for m in embedding_models if m not in model_names]
        if invalid:
            print(f"❌ Invalid embedding models: {invalid}")
            return 1

    if args.llm:
        llm_models = [m.strip() for m in args.llm.split(',')]
        # Validate models
        invalid = [m for m in llm_models if m not in llms]
        if invalid:
            print(f"❌ Invalid LLM models: {invalid}")
            return 1

    # Use defaults if not specified: all keys from config
    if not embedding_models:
        embedding_models = list(model_names.keys())
        print(f"📋 Using all embedding models from config: {embedding_models}")

    if not llm_models:
        llm_models = list(llms.keys())
        print(f"📋 Using all LLM models from config: {llm_models}")

    # Show configuration
    print(f"\nEvaluation Configuration:")
    print(f"   Embedding Models: {', '.join(embedding_models)}")
    print(f"   LLM Models: {', '.join(llm_models)}")
    print(f"   Quick Test: {'Yes' if args.quick else 'No'}")
    print(f"   Total Combinations: {len(embedding_models) * len(llm_models)}")
    print()

    try:
        print("🚀 Starting evaluation pipeline...\n")

        # Run the complete evaluation
        results = manager.run_complete_evaluation(
            embedding_models=embedding_models,
            llm_models=llm_models,
            quick_test=args.quick
        )

        # Print summary
        print("\n" + "=" * 70)
        print("    ✅ EVALUATION COMPLETE")
        print("=" * 70)

        # Results summary
        if 'summary_stats' in results:
            stats = results['summary_stats']['overall']
            print(f"📊 Results Summary:")
            print(f"   Total Evaluations: {stats['total_evaluations']}")
            print(
                f"   Model Combinations: {stats['unique_model_combinations']}")
            print(f"   Average Precision: {stats['avg_precision']:.3f}")
            print(f"   Average Recall: {stats['avg_recall']:.3f}")
            print(f"   Average F1-Score: {stats['avg_f1_score']:.3f}")
            print(
                f"   Average Response Time: {stats['average_search_time']:.2f}s")

            # Top performer
            if 'model_rankings' in results['summary_stats']:
                top_model = list(results['summary_stats']
                                 ['model_rankings'].keys())[0]
                top_f1 = results['summary_stats']['model_rankings'][top_model]['f1_score']
                print(f"   🏆 Best Model: {top_model} (F1: {top_f1:.3f})")

        # Generated files
        if 'file_paths' in results:
            print(f"\n📁 Generated Files ({len(results['file_paths'])}):")
            print(f"   📊 Results Data:")
            for file_type, path in results['file_paths'].items():
                if file_type.endswith('_csv') or file_type.endswith('_json'):
                    print(f"     {file_type:20s}: {path}")
            print(f"   📈 Report Assets (LaTeX):")
            for file_type, path in results['file_paths'].items():
                if file_type.endswith('_plot') or file_type.endswith('_table') or 'heatmap' in file_type:
                    print(f"     {file_type:20s}: {path}")

        print(
            f"\n⏱️  Evaluation completed at: {results.get('timestamp', 'N/A')}")
        print("\n🎉 All results and reports generated successfully!")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
