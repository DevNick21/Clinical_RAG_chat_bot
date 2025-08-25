"""
Visualization module for Clinical RAG evaluation results
Handles all charting, plotting, and visual analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class EvaluationVisualizer:
    """Handles all visualization needs for evaluation results"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("evaluation_reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Set consistent style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_score_heatmap(self, results: Dict, timestamp: str = None) -> str:
        """Create score heatmap by category and metric"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fig, ax = plt.subplots(figsize=(12, 8))
        category_results = results.get("category_results", {})

        # Create score matrix for heatmap
        score_matrix = []
        categories = []
        metrics = ["Overall", "Factual", "Behavior",
                   "Performance", "Context", "Completeness", "Semantic"]

        for category, cat_results in category_results.items():
            categories.append(category)
            row = [
                np.mean([r.get("overall_score", 0) for r in cat_results]),
                np.mean([r.get("factual_accuracy_score", 0)
                        for r in cat_results]),
                np.mean([r.get("behavior_score", 0) for r in cat_results]),
                np.mean([r.get("performance_score", 0) for r in cat_results]),
                np.mean([r.get("context_relevance_score", 0)
                        for r in cat_results]),
                np.mean([r.get("completeness_score", 0) for r in cat_results]),
                np.mean([r.get("semantic_similarity_score", 0)
                        for r in cat_results])
            ]
            score_matrix.append(row)

        sns.heatmap(score_matrix,
                    xticklabels=metrics,
                    yticklabels=categories,
                    annot=True,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1,
                    fmt='.3f')

        plt.title("RAG Evaluation Score Heatmap by Category and Metric")
        plt.tight_layout()

        heatmap_file = self.output_dir / f"score_heatmap_{timestamp}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()

        return str(heatmap_file)

    def create_grade_distribution(self, results: Dict, timestamp: str = None) -> str:
        """Create grade distribution bar chart"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fig, ax = plt.subplots(figsize=(10, 6))
        all_grades = [r.get("pass_grade", "fail")
                      for r in results.get("detailed_results", [])]
        grade_counts = pd.Series(all_grades).value_counts()

        colors = {"excellent": "green", "pass": "lightgreen",
                  "borderline": "orange", "fail": "red"}
        grade_counts.plot(kind='bar',
                          color=[colors.get(x, 'gray')
                                 for x in grade_counts.index],
                          ax=ax)

        plt.title("Distribution of Pass Grades")
        plt.ylabel("Count")
        plt.xlabel("Grade")
        plt.xticks(rotation=45)
        plt.tight_layout()

        grades_file = self.output_dir / f"grade_distribution_{timestamp}.png"
        plt.savefig(grades_file, dpi=300, bbox_inches='tight')
        plt.close()

        return str(grades_file)

    def create_time_vs_score_scatter(self, results: Dict, timestamp: str = None) -> str:
        """Create search time vs score scatter plot"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fig, ax = plt.subplots(figsize=(10, 6))
        detailed_results = results.get("detailed_results", [])

        scores = [r.get("overall_score", 0) for r in detailed_results]
        search_times = [r.get("search_time", 0) for r in detailed_results]
        categories = [r.get("category", "") for r in detailed_results]

        scatter = ax.scatter(search_times, scores, c=range(len(categories)),
                             alpha=0.6, cmap='tab10')

        ax.set_xlabel("Search Time (seconds)")
        ax.set_ylabel("Overall Score")
        ax.set_title("Search Time vs Overall Score")

        # Add trend line
        if len(search_times) > 1:
            z = np.polyfit(search_times, scores, 1)
            p = np.poly1d(z)
            ax.plot(search_times, p(search_times), "r--", alpha=0.8)

        plt.tight_layout()

        scatter_file = self.output_dir / f"time_vs_score_{timestamp}.png"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()

        return str(scatter_file)

    def create_document_analysis(self, results: Dict, timestamp: str = None) -> str:
        """Create document count analysis plots"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        detailed_results = results.get("detailed_results", [])

        # Documents found distribution
        doc_counts = [r.get("documents_found", 0) for r in detailed_results]
        scores = [r.get("overall_score", 0) for r in detailed_results]

        ax1.hist(doc_counts, bins=20, alpha=0.7,
                 color='skyblue', edgecolor='black')
        ax1.set_xlabel("Documents Found")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Documents Found")

        # Documents found vs Score
        ax2.scatter(doc_counts, scores, alpha=0.6, color='coral')
        ax2.set_xlabel("Documents Found")
        ax2.set_ylabel("Overall Score")
        ax2.set_title("Documents Found vs Overall Score")

        # Add trend line if we have data
        if len(doc_counts) > 1:
            z = np.polyfit(doc_counts, scores, 1)
            p = np.poly1d(z)
            ax2.plot(doc_counts, p(doc_counts), "r--", alpha=0.8)

        plt.tight_layout()

        docs_file = self.output_dir / f"document_analysis_{timestamp}.png"
        plt.savefig(docs_file, dpi=300, bbox_inches='tight')
        plt.close()

        return str(docs_file)

    def create_model_comparison_heatmap(self, df: pd.DataFrame, metric: str = "average_score") -> str:
        """Create interactive heatmap for model comparison"""
        if df.empty:
            return ""

        # Create pivot table for heatmap
        heatmap_data = df.pivot(index="embedding_model",
                                columns="llm_model",
                                values=metric)

        # Create the heatmap with plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values,
            texttemplate="%{text:.1%}" if metric == 'average_score' else "%{text:.3f}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=f'Model Performance Heatmap: {metric.replace("_", " ").title()}',
            xaxis_title='LLM Model',
            yaxis_title='Embedding Model',
            width=800,
            height=600
        )

        html_file = self.output_dir / f"heatmap_{metric}.html"
        fig.write_html(str(html_file))

        return str(html_file)

    def create_evaluation_comparison_plot(self, labels: List[str], comparison_data: List[Dict[str, Any]], timestamp: str | None = None) -> str:
        """Create a 2x2 comparison plot for multiple evaluation runs.
        Expects comparison_data items with keys: Pass_Rate (e.g., '80.0%'), Average_Score (e.g., '0.750'),
        Excellent_Count, Pass_Count, Borderline_Count, Fail_Count, Avg_Search_Time (e.g., '1.23s').
        Returns the saved image path.
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not labels or not comparison_data:
            return ""

        # Parse values
        def parse_percent(p: Any) -> float:
            try:
                if isinstance(p, str) and p.endswith('%'):
                    return float(p.strip('%')) / 100.0
                return float(p)
            except Exception:
                return 0.0

        def parse_seconds(s: Any) -> float:
            try:
                if isinstance(s, str) and s.endswith('s'):
                    return float(s[:-1])
                return float(s)
            except Exception:
                return 0.0

        pass_rates = [parse_percent(row.get("Pass_Rate", 0))
                      for row in comparison_data]
        avg_scores = [float(row.get("Average_Score", 0))
                      for row in comparison_data]
        search_times = [parse_seconds(row.get("Avg_Search_Time", 0))
                        for row in comparison_data]

        grades_data = np.array([[row.get("Excellent_Count", 0), row.get("Pass_Count", 0),
                                 row.get("Borderline_Count", 0), row.get("Fail_Count", 0)]
                                for row in comparison_data])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Pass rates
        ax1.bar(labels, pass_rates, color='lightblue', alpha=0.7)
        ax1.set_ylabel('Pass Rate')
        ax1.set_title('Pass Rate Comparison')
        ax1.set_ylim(0, 1)

        # Grades stacked bar
        bottom = np.zeros(len(labels))
        colors = ['green', 'lightgreen', 'orange', 'red']
        grade_labels = ['Excellent', 'Pass', 'Borderline', 'Fail']
        for i, (color, glabel) in enumerate(zip(colors, grade_labels)):
            ax2.bar(labels, grades_data[:, i], bottom=bottom,
                    color=color, alpha=0.7, label=glabel)
            bottom += grades_data[:, i]
        ax2.set_ylabel('Count')
        ax2.set_title('Grade Distribution Comparison')
        ax2.legend()

        # Average scores
        ax3.bar(labels, avg_scores, color='lightcoral', alpha=0.7)
        ax3.set_ylabel('Average Score')
        ax3.set_title('Average Score Comparison')
        ax3.set_ylim(0, 1)

        # Search times
        ax4.bar(labels, search_times, color='lightsteelblue', alpha=0.7)
        ax4.set_ylabel('Search Time (seconds)')
        ax4.set_title('Average Search Time Comparison')

        plt.tight_layout()

        out_file = self.output_dir / f"evaluation_comparison_{timestamp}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()

        return str(out_file)

    def generate_all_visualizations(self, results: Dict, timestamp: str = None) -> Dict[str, str]:
        """Generate all visualizations and return file paths"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        viz_files = {}

        try:
            viz_files["heatmap"] = self.create_score_heatmap(
                results, timestamp)
            viz_files["grades"] = self.create_grade_distribution(
                results, timestamp)
            viz_files["scatter"] = self.create_time_vs_score_scatter(
                results, timestamp)
            viz_files["documents"] = self.create_document_analysis(
                results, timestamp)
        except Exception as e:
            print(f" Error generating visualizations: {e}")

        return viz_files
