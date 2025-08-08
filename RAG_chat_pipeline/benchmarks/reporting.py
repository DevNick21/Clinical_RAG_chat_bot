"""
Reporting module for Clinical RAG evaluation results
Handles report generation, data export, and summary statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class EvaluationReporter:
    """Handles all reporting and data export functionality"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("evaluation_reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def create_summary_dataframe(self, results: Dict) -> pd.DataFrame:
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

    def create_performance_comparison(self, results: Dict) -> pd.DataFrame:
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

    def create_category_analysis(self, results: Dict) -> pd.DataFrame:
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

    def export_comprehensive_csv(self, results: Dict, timestamp: str = None) -> Dict[str, str]:
        """Export comprehensive CSV reports"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_files = {}

        try:
            # 1. Export detailed results to CSV
            csv_file = self.output_dir / f"detailed_results_{timestamp}.csv"
            detailed_results = results.get("detailed_results", [])
            if detailed_results:
                df_results = pd.DataFrame(detailed_results)
                df_results.to_csv(csv_file, index=False)
                report_files["detailed_csv"] = str(csv_file)

            # 2. Create summary statistics CSV
            summary_file = self.output_dir / f"summary_stats_{timestamp}.csv"
            summary_df = self.create_summary_dataframe(results)
            summary_df.to_csv(summary_file, index=False)
            report_files["summary_csv"] = str(summary_file)

            # 3. Create performance comparison table
            perf_file = self.output_dir / \
                f"performance_comparison_{timestamp}.csv"
            perf_df = self.create_performance_comparison(results)
            perf_df.to_csv(perf_file, index=False)
            report_files["performance_csv"] = str(perf_file)

            # 4. Generate category-wise analysis
            category_file = self.output_dir / \
                f"category_analysis_{timestamp}.csv"
            category_df = self.create_category_analysis(results)
            category_df.to_csv(category_file, index=False)
            report_files["category_csv"] = str(category_file)

        except Exception as e:
            print(f"⚠️ Error exporting CSV files: {e}")

        return report_files

    def generate_text_summary(self, results: Dict) -> str:
        """Generate a comprehensive text summary report"""
        summary = results.get("summary", {})
        category_breakdown = summary.get("category_breakdown", {})

        report = f"""
# Clinical RAG Evaluation Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance
- **Total Questions**: {summary.get('total_questions', 0)}
- **Questions Passed**: {summary.get('passed', 0)}
- **Pass Rate**: {summary.get('pass_rate', 0):.1%}
- **Average Score**: {summary.get('average_score', 0):.3f}

## Category Breakdown
"""

        for category, stats in category_breakdown.items():
            report += f"""
### {category.title()}
- Question Count: {stats.get('count', 0)}
- Pass Rate: {stats.get('pass_rate', 0):.1%}
- Average Score: {stats.get('average_score', 0):.3f}
"""

        # Add retrieval metrics if available
        retrieval_metrics = summary.get("retrieval_metrics", {})
        if retrieval_metrics.get("questions_evaluated", 0) > 0:
            report += f"""
## Retrieval Performance
- Questions Evaluated: {retrieval_metrics.get('questions_evaluated', 0)}
- Average Precision: {retrieval_metrics.get('avg_precision', 0):.3f}
- Average Recall: {retrieval_metrics.get('avg_recall', 0):.3f}
- Average F1: {retrieval_metrics.get('avg_f1', 0):.3f}
"""

        return report

    def create_model_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a formatted comparison table for model results"""
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

        return comparison_df
