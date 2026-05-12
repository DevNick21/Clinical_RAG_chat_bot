"""
Reporting module for Clinical RAG evaluation results
Handles markdown report generation, data export, and summary statistics using semantic metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class EvaluationReporter:
    """Handles all reporting and data export functionality for semantic evaluation"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("evaluation_reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive markdown summary report using semantic metrics"""
        if df.empty:
            return "No evaluation results available."

        # Find best performing combinations
        best_f1 = df.loc[df["avg_f1_score"].idxmax()]
        best_precision = df.loc[df["avg_precision"].idxmax()]
        best_recall = df.loc[df["avg_recall"].idxmax()]
        fastest = df.loc[df["avg_search_time"].idxmin()]

        report = f"""# Clinical RAG Model Comparison Report
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
        return report
    
    def save_summary_report(self, df: pd.DataFrame, filename: str = None) -> Path:
        """Generate and save markdown summary report"""
        report_content = self.generate_summary_report(df)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_summary_{timestamp}.md"
        
        report_file = self.output_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_file

    def create_summary_dataframe(self, results: Dict) -> pd.DataFrame:
        """Create summary statistics DataFrame using semantic metrics"""
        summary = results.get("summary", {})
        summary_data = []

        # Core semantic metrics
        summary_data.append({
            "Metric": "Average Precision",
            "Value": f"{summary.get('avg_precision', 0):.3f}"
        })
        
        summary_data.append({
            "Metric": "Average Recall", 
            "Value": f"{summary.get('avg_recall', 0):.3f}"
        })
        
        summary_data.append({
            "Metric": "Average F1-Score",
            "Value": f"{summary.get('avg_f1_score', 0):.3f}"
        })
        
        summary_data.append({
            "Metric": "Average Search Time",
            "Value": f"{summary.get('average_search_time', 0):.2f}s"
        })

        summary_data.append({
            "Metric": "Total Questions",
            "Value": summary.get('total_questions', 0)
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
                "Precision": result.get("precision", 0),
                "Recall": result.get("recall", 0),
                "F1_Score": result.get("f1_score", 0),
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
            print(f" Error exporting CSV files: {e}")

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
            "avg_precision", "avg_recall", "avg_f1_score",
            "avg_search_time"
        ]

        comparison_df = df[comparison_cols].copy()

        # Format percentages and scores
        comparison_df["pass_rate"] = comparison_df["pass_rate"].apply(
            lambda x: f"{x:.1%}")
        comparison_df["average_score"] = comparison_df["average_score"].apply(
            lambda x: f"{x:.3f}")
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
            "Embedding Model", "LLM Model", "Pass Rate", "Avg Score",
            "Precision", "Recall", "F1-Score", "Search Time"
        ]

        return comparison_df
