"""
Dashboard generation module for Clinical RAG evaluation results
Handles HTML dashboard creation and web-based reporting
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json


class EvaluationDashboard:
    """Handles HTML dashboard generation"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("evaluation_reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def create_performance_dashboard(self, results: Dict, report_files: Dict[str, str],
                                     viz_files: Dict[str, str], timestamp: str = None) -> str:
        """Create comprehensive HTML dashboard"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        dashboard_file = self.output_dir / \
            f"performance_dashboard_{timestamp}.html"

        # Extract summary data
        summary = results.get("summary", {})

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical RAG Performance Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .category-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .category-card {{
            background: #fff;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }}
        .category-name {{
            font-weight: bold;
            color: #007acc;
            margin-bottom: 10px;
        }}
        .file-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }}
        .file-link {{
            background: #007acc;
            color: white;
            padding: 8px 16px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 0.9em;
        }}
        .file-link:hover {{
            background: #005999;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-bad {{ color: #dc3545; }}
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .viz-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .viz-title {{
            background: #f8f9fa;
            padding: 15px;
            margin: 0;
            border-bottom: 1px solid #ddd;
        }}
        .viz-content {{
            padding: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Clinical RAG Performance Dashboard</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2 class="section-title">📊 Overall Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {self._get_status_class(summary.get('pass_rate', 0))}">{summary.get('pass_rate', 0):.1%}</div>
                    <div class="metric-label">Pass Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('average_score', 0):.3f}</div>
                    <div class="metric-label">Average Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('passed', 0)}/{summary.get('total_questions', 0)}</div>
                    <div class="metric-label">Questions Passed</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">📋 Category Breakdown</h2>
            <div class="category-breakdown">
"""

        # Add category cards
        category_breakdown = summary.get("category_breakdown", {})
        for category, stats in category_breakdown.items():
            html_content += f"""
                <div class="category-card">
                    <div class="category-name">{category.title()}</div>
                    <div>Pass Rate: <span class="{self._get_status_class(stats.get('pass_rate', 0))}">{stats.get('pass_rate', 0):.1%}</span></div>
                    <div>Avg Score: {stats.get('average_score', 0):.3f}</div>
                    <div>Questions: {stats.get('count', 0)}</div>
                </div>
"""

        html_content += """
            </div>
        </div>
"""

        # Add retrieval metrics section if available
        retrieval_metrics = summary.get("retrieval_metrics", {})
        if retrieval_metrics.get("questions_evaluated", 0) > 0:
            html_content += f"""
        <div class="section">
            <h2 class="section-title">🔍 Retrieval Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{retrieval_metrics.get('avg_precision', 0):.3f}</div>
                    <div class="metric-label">Average Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{retrieval_metrics.get('avg_recall', 0):.3f}</div>
                    <div class="metric-label">Average Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{retrieval_metrics.get('avg_f1', 0):.3f}</div>
                    <div class="metric-label">Average F1</div>
                </div>
            </div>
        </div>
"""

        # Add visualizations section
        if viz_files:
            html_content += """
        <div class="section">
            <h2 class="section-title">📈 Visualizations</h2>
            <div class="visualization-grid">
"""
            for viz_name, viz_file in viz_files.items():
                if viz_file and Path(viz_file).exists():
                    rel_path = Path(viz_file).name
                    html_content += f"""
                <div class="viz-card">
                    <h3 class="viz-title">{viz_name.title().replace('_', ' ')}</h3>
                    <div class="viz-content">
                        <a href="./{rel_path}" class="file-link">View {viz_name.title()}</a>
                    </div>
                </div>
"""
            html_content += """
            </div>
        </div>
"""

        # Add file downloads section
        if report_files or viz_files:
            html_content += """
        <div class="section">
            <h2 class="section-title">📁 Generated Files</h2>
            <div class="file-links">
"""

            all_files = {**report_files, **viz_files}
            for file_type, file_path in all_files.items():
                if file_path and Path(file_path).exists():
                    rel_path = Path(file_path).name
                    html_content += f"""
                <a href="./{rel_path}" class="file-link">📄 {file_type.replace('_', ' ').title()}</a>
"""

            html_content += """
            </div>
        </div>
"""

        # Close HTML
        html_content += """
    </div>
</body>
</html>
"""

        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(dashboard_file)

    def _get_status_class(self, value: float) -> str:
        """Get CSS class based on value"""
        if value >= 0.8:
            return "status-good"
        elif value >= 0.6:
            return "status-warning"
        else:
            return "status-bad"

    def create_model_comparison_dashboard(self, df, timestamp: str = None) -> str:
        """Create a dashboard for model comparison results"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        dashboard_file = self.output_dir / \
            f"model_comparison_dashboard_{timestamp}.html"

        # Convert DataFrame to HTML table
        table_html = df.to_html(
            classes="table table-striped", table_id="comparison_table")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Model Comparison Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Performance Comparison</h2>
        {table_html}
    </div>
</body>
</html>
"""

        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(dashboard_file)
