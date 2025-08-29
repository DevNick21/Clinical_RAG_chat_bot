#!/usr/bin/env python3
"""
Advanced Statistical Analysis for Clinical RAG Evaluation Results
Implements comprehensive analysis including category breakdown, reliability metrics,
efficiency frontiers, error analysis, and additional heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("tab10")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

class AdvancedClinicalRAGAnalysis:
    def __init__(self, results_path="results/results_dataframe.csv", output_dir="report/chap4_results"):
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.model_stats = None
        
    def load_data(self):
        """Load and preprocess the results data"""
        if not self.results_path.exists():
            print(f"Results file not found at {self.results_path}")
            return False
            
        self.df = pd.read_csv(self.results_path)
        print(f"Loaded {len(self.df)} evaluations from {len(self.df['model_combination'].unique())} model combinations")
        
        # Create aggregated model statistics
        self.model_stats = self.df.groupby(['embedding_model', 'llm_model', 'model_combination']).agg({
            'f1_score': ['mean', 'std', 'count'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'search_time': ['mean', 'std'],
            'documents_found': 'mean'
        }).round(3)
        
        # Flatten column names
        self.model_stats.columns = ['_'.join(col).strip() for col in self.model_stats.columns.values]
        self.model_stats = self.model_stats.reset_index()
        
        return True
    
    def create_precision_recall_heatmaps(self):
        """Create heatmaps for precision and recall"""
        print("Creating precision and recall heatmaps...")
        
        # Precision heatmap
        precision_data = self.model_stats.pivot(
            index='embedding_model', 
            columns='llm_model', 
            values='precision_mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(precision_data, 
                    annot=True, 
                    fmt='.3f',
                    cmap='RdYlGn',
                    center=precision_data.mean().mean(),
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Average Precision"})
        
        plt.title('Model Performance Heatmap: Average Precision', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Language Model', fontsize=12, fontweight='bold')
        plt.ylabel('Embedding Model', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        precision_file = self.images_dir / "heatmap_precision.png"
        plt.savefig(precision_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Recall heatmap
        recall_data = self.model_stats.pivot(
            index='embedding_model', 
            columns='llm_model', 
            values='recall_mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(recall_data, 
                    annot=True, 
                    fmt='.3f',
                    cmap='RdYlGn',
                    center=recall_data.mean().mean(),
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Average Recall"})
        
        plt.title('Model Performance Heatmap: Average Recall', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Language Model', fontsize=12, fontweight='bold')
        plt.ylabel('Embedding Model', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        recall_file = self.images_dir / "heatmap_recall.png"
        plt.savefig(recall_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Created precision heatmap: {precision_file}")
        print(f"Created recall heatmap: {recall_file}")
        
    def analyze_clinical_categories(self):
        """Analyze performance by clinical query categories"""
        print("Analyzing clinical query categories...")
        
        if 'category' not in self.df.columns:
            print("No category column found, creating synthetic categories based on question content")
            # Create categories based on question content
            self.df['category'] = self.df['question'].apply(self._categorize_question)
        
        category_stats = self.df.groupby(['category', 'model_combination']).agg({
            'f1_score': ['mean', 'std', 'count'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'search_time': 'mean'
        }).round(3)
        
        # Overall category performance
        category_overall = self.df.groupby('category').agg({
            'f1_score': ['mean', 'std', 'count'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'search_time': ['mean', 'std']
        }).round(3)
        
        category_overall.columns = ['_'.join(col).strip() for col in category_overall.columns.values]
        category_overall = category_overall.reset_index()
        
        # Create category performance visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1 scores by category
        category_f1 = self.df.groupby('category')['f1_score'].mean().sort_values(ascending=True)
        category_f1.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('Average F1 Score by Clinical Category')
        ax1.set_xlabel('F1 Score')
        
        # Precision by category
        category_precision = self.df.groupby('category')['precision'].mean().sort_values(ascending=True)
        category_precision.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('Average Precision by Clinical Category')
        ax2.set_xlabel('Precision')
        
        # Recall by category
        category_recall = self.df.groupby('category')['recall'].mean().sort_values(ascending=True)
        category_recall.plot(kind='barh', ax=ax3, color='lightgreen')
        ax3.set_title('Average Recall by Clinical Category')
        ax3.set_xlabel('Recall')
        
        # Search time by category
        category_time = self.df.groupby('category')['search_time'].mean().sort_values(ascending=True)
        category_time.plot(kind='barh', ax=ax4, color='gold')
        ax4.set_title('Average Search Time by Clinical Category')
        ax4.set_xlabel('Search Time (seconds)')
        
        plt.tight_layout()
        category_file = self.images_dir / "category_performance_analysis.png"
        plt.savefig(category_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create category performance table
        category_table = self._create_category_table(category_overall)
        with open(self.tables_dir / "category_performance.tex", "w") as f:
            f.write(category_table)
        
        print(f"Created category analysis visualization: {category_file}")
        return category_overall
    
    def _categorize_question(self, question):
        """Simple categorization of questions based on content"""
        question_lower = question.lower()
        if any(word in question_lower for word in ['lab', 'laboratory', 'blood', 'test', 'result']):
            return 'Laboratory'
        elif any(word in question_lower for word in ['medication', 'drug', 'prescription', 'dose']):
            return 'Medication'
        elif any(word in question_lower for word in ['diagnosis', 'diagnose', 'condition', 'disease']):
            return 'Diagnosis'
        elif any(word in question_lower for word in ['admission', 'demographic', 'patient', 'age', 'gender']):
            return 'Demographics'
        elif any(word in question_lower for word in ['procedure', 'surgery', 'operation', 'treatment']):
            return 'Procedures'
        else:
            return 'General'
    
    def analyze_consistency_reliability(self):
        """Analyze model consistency and reliability"""
        print("Analyzing consistency and reliability metrics...")
        
        # Calculate consistency metrics for each model combination
        consistency_stats = []
        
        for combo in self.df['model_combination'].unique():
            combo_data = self.df[self.df['model_combination'] == combo]
            
            # Inter-question consistency (coefficient of variation)
            f1_cv = (combo_data['f1_score'].std() / combo_data['f1_score'].mean()) * 100
            precision_cv = (combo_data['precision'].std() / combo_data['precision'].mean()) * 100
            recall_cv = (combo_data['recall'].std() / combo_data['recall'].mean()) * 100
            
            # Reliability coefficient (1 - CV/100)
            reliability_f1 = max(0, 1 - (f1_cv / 100))
            
            # Count of "good" performances (F1 > 0.6)
            good_performance_rate = (combo_data['f1_score'] > 0.6).mean()
            
            # Outlier rate (performances > 2 std from mean)
            mean_f1 = combo_data['f1_score'].mean()
            std_f1 = combo_data['f1_score'].std()
            outlier_rate = (abs(combo_data['f1_score'] - mean_f1) > 2 * std_f1).mean()
            
            consistency_stats.append({
                'model_combination': combo,
                'f1_cv': f1_cv,
                'precision_cv': precision_cv,
                'recall_cv': recall_cv,
                'reliability_f1': reliability_f1,
                'good_performance_rate': good_performance_rate,
                'outlier_rate': outlier_rate,
                'mean_f1': mean_f1
            })
        
        consistency_df = pd.DataFrame(consistency_stats).round(3)
        consistency_df = consistency_df.sort_values('reliability_f1', ascending=False)
        
        # Create consistency visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top 10 most consistent models
        top_consistent = consistency_df.head(10)
        ax1.barh(range(len(top_consistent)), top_consistent['reliability_f1'], color='lightblue')
        ax1.set_yticks(range(len(top_consistent)))
        ax1.set_yticklabels([combo.replace(' + ', '+') for combo in top_consistent['model_combination']], fontsize=8)
        ax1.set_title('Top 10 Most Consistent Models (Reliability Score)')
        ax1.set_xlabel('Reliability Score')
        
        # CV vs Performance scatter
        ax2.scatter(consistency_df['f1_cv'], consistency_df['mean_f1'], alpha=0.7)
        ax2.set_xlabel('F1 Coefficient of Variation (%)')
        ax2.set_ylabel('Mean F1 Score')
        ax2.set_title('Consistency vs Performance Trade-off')
        
        # Good performance rate distribution
        ax3.hist(consistency_df['good_performance_rate'], bins=15, alpha=0.7, color='lightgreen')
        ax3.set_xlabel('Good Performance Rate (F1 > 0.6)')
        ax3.set_ylabel('Number of Model Combinations')
        ax3.set_title('Distribution of Good Performance Rates')
        
        # Outlier rate analysis
        ax4.hist(consistency_df['outlier_rate'], bins=15, alpha=0.7, color='lightcoral')
        ax4.set_xlabel('Outlier Rate')
        ax4.set_ylabel('Number of Model Combinations')
        ax4.set_title('Distribution of Outlier Rates')
        
        plt.tight_layout()
        consistency_file = self.images_dir / "consistency_reliability_analysis.png"
        plt.savefig(consistency_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create consistency table
        consistency_table = self._create_consistency_table(consistency_df.head(10))
        with open(self.tables_dir / "consistency_reliability.tex", "w") as f:
            f.write(consistency_table)
        
        print(f"Created consistency analysis: {consistency_file}")
        return consistency_df
    
    def analyze_efficiency_frontiers(self):
        """Analyze efficiency frontiers and Pareto optimal solutions"""
        print("Analyzing efficiency frontiers...")
        
        # Calculate efficiency metrics
        self.model_stats['throughput_qpm'] = 60 / self.model_stats['search_time_mean']  # Questions per minute
        self.model_stats['efficiency_score'] = self.model_stats['f1_score_mean'] * self.model_stats['throughput_qpm']
        
        # Find Pareto frontier (maximize F1, minimize time)
        pareto_points = []
        for i, row in self.model_stats.iterrows():
            is_pareto = True
            for j, other_row in self.model_stats.iterrows():
                if i != j:
                    # Check if other point dominates current point
                    if (other_row['f1_score_mean'] >= row['f1_score_mean'] and 
                        other_row['search_time_mean'] <= row['search_time_mean'] and
                        (other_row['f1_score_mean'] > row['f1_score_mean'] or 
                         other_row['search_time_mean'] < row['search_time_mean'])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_points.append(i)
        
        pareto_models = self.model_stats.iloc[pareto_points].sort_values('search_time_mean')
        
        # Create efficiency frontier visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Pareto frontier
        ax1.scatter(self.model_stats['search_time_mean'], self.model_stats['f1_score_mean'], 
                   alpha=0.6, color='lightblue', label='All Models')
        ax1.scatter(pareto_models['search_time_mean'], pareto_models['f1_score_mean'], 
                   color='red', s=100, label='Pareto Optimal', zorder=5)
        ax1.plot(pareto_models['search_time_mean'], pareto_models['f1_score_mean'], 
                'r--', alpha=0.7, zorder=4)
        ax1.set_xlabel('Search Time (seconds)')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Efficiency Frontier (Pareto Optimal Points)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency score ranking
        top_efficient = self.model_stats.nlargest(10, 'efficiency_score')
        ax2.barh(range(len(top_efficient)), top_efficient['efficiency_score'], color='lightgreen')
        ax2.set_yticks(range(len(top_efficient)))
        ax2.set_yticklabels([combo.replace(' + ', '+') for combo in top_efficient['model_combination']], fontsize=8)
        ax2.set_title('Top 10 Most Efficient Models (F1 × QPM)')
        ax2.set_xlabel('Efficiency Score')
        
        # Throughput vs F1
        ax3.scatter(self.model_stats['throughput_qpm'], self.model_stats['f1_score_mean'], alpha=0.7)
        ax3.set_xlabel('Throughput (Questions per Minute)')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Throughput vs Quality Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # Cost-benefit analysis (F1 improvement per second)
        fastest_time = self.model_stats['search_time_mean'].min()
        self.model_stats['time_cost'] = self.model_stats['search_time_mean'] - fastest_time
        self.model_stats['f1_benefit'] = self.model_stats['f1_score_mean'] - self.model_stats['f1_score_mean'].min()
        
        # Filter out points with zero time cost
        cost_benefit = self.model_stats[self.model_stats['time_cost'] > 0]
        if not cost_benefit.empty:
            cost_benefit['benefit_per_second'] = cost_benefit['f1_benefit'] / cost_benefit['time_cost']
            top_cost_benefit = cost_benefit.nlargest(10, 'benefit_per_second')
            
            ax4.barh(range(len(top_cost_benefit)), top_cost_benefit['benefit_per_second'], color='gold')
            ax4.set_yticks(range(len(top_cost_benefit)))
            ax4.set_yticklabels([combo.replace(' + ', '+') for combo in top_cost_benefit['model_combination']], fontsize=8)
            ax4.set_title('Best F1 Improvement per Additional Second')
            ax4.set_xlabel('F1 Benefit per Second')
        
        plt.tight_layout()
        efficiency_file = self.images_dir / "efficiency_frontier_analysis.png"
        plt.savefig(efficiency_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create efficiency tables
        pareto_table = self._create_pareto_table(pareto_models)
        with open(self.tables_dir / "pareto_optimal_models.tex", "w") as f:
            f.write(pareto_table)
        
        print(f"Created efficiency frontier analysis: {efficiency_file}")
        return pareto_models, top_efficient
    
    def analyze_error_patterns(self):
        """Analyze error patterns and failure modes"""
        print("Analyzing error patterns and failure modes...")
        
        # Define failure as F1 < 0.4 (arbitrary threshold)
        failure_threshold = 0.4
        self.df['is_failure'] = self.df['f1_score'] < failure_threshold
        
        # Analyze failure rates by model
        failure_by_model = self.df.groupby('model_combination').agg({
            'is_failure': ['mean', 'sum', 'count'],
            'f1_score': ['mean', 'std']
        }).round(3)
        
        failure_by_model.columns = ['failure_rate', 'failure_count', 'total_questions', 'mean_f1', 'std_f1']
        failure_by_model = failure_by_model.reset_index()
        failure_by_model = failure_by_model.sort_values('failure_rate')
        
        # Find questions with highest variance across models
        question_variance = self.df.groupby('question')['f1_score'].agg(['mean', 'std', 'count']).reset_index()
        question_variance = question_variance[question_variance['count'] >= 10]  # Only questions answered by many models
        high_variance_questions = question_variance.nlargest(10, 'std')
        
        # Analyze performance patterns
        low_performers = self.df[self.df['f1_score'] < 0.3]  # Very low performance
        high_performers = self.df[self.df['f1_score'] > 0.8]  # Very high performance
        
        # Create error analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Failure rates by model
        top_reliable = failure_by_model.head(10)
        ax1.barh(range(len(top_reliable)), top_reliable['failure_rate'], color='lightcoral')
        ax1.set_yticks(range(len(top_reliable)))
        ax1.set_yticklabels([combo.replace(' + ', '+') for combo in top_reliable['model_combination']], fontsize=8)
        ax1.set_title('Top 10 Most Reliable Models (Lowest Failure Rate)')
        ax1.set_xlabel('Failure Rate (F1 < 0.4)')
        
        # F1 distribution comparison
        ax2.hist(low_performers['search_time'], bins=20, alpha=0.7, label='Low F1 (<0.3)', color='red')
        ax2.hist(high_performers['search_time'], bins=20, alpha=0.7, label='High F1 (>0.8)', color='green')
        ax2.set_xlabel('Search Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Search Time Distribution: Low vs High Performance')
        ax2.legend()
        
        # Documents found vs performance
        ax3.scatter(self.df['documents_found'], self.df['f1_score'], alpha=0.5)
        ax3.set_xlabel('Documents Found')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Document Retrieval vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # Precision vs Recall scatter colored by F1
        scatter = ax4.scatter(self.df['precision'], self.df['recall'], 
                             c=self.df['f1_score'], cmap='RdYlGn', alpha=0.6)
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision vs Recall (colored by F1 Score)')
        plt.colorbar(scatter, ax=ax4, label='F1 Score')
        
        plt.tight_layout()
        error_file = self.images_dir / "error_pattern_analysis.png"
        plt.savefig(error_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create failure analysis table
        failure_table = self._create_failure_analysis_table(failure_by_model.head(10))
        with open(self.tables_dir / "failure_analysis.tex", "w") as f:
            f.write(failure_table)
        
        print(f"Created error pattern analysis: {error_file}")
        return failure_by_model, high_variance_questions
    
    def domain_specific_analysis(self):
        """Analyze performance on different types of clinical data"""
        print("Analyzing domain-specific performance...")
        
        # Analyze performance vs document types/complexity
        self.df['doc_complexity'] = pd.cut(self.df['documents_found'], 
                                          bins=[0, 5, 10, 20, float('inf')], 
                                          labels=['Low (1-5)', 'Medium (6-10)', 'High (11-20)', 'Very High (20+)'])
        
        complexity_analysis = self.df.groupby(['doc_complexity', 'model_combination']).agg({
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'search_time': 'mean'
        }).reset_index()
        
        # Overall complexity analysis
        complexity_overall = self.df.groupby('doc_complexity').agg({
            'f1_score': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'search_time': ['mean', 'std']
        }).round(3)
        
        # Create domain-specific visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance by document complexity
        complexity_f1 = self.df.groupby('doc_complexity')['f1_score'].mean()
        complexity_f1.plot(kind='bar', ax=ax1, color='skyblue', rot=45)
        ax1.set_title('F1 Score by Document Retrieval Complexity')
        ax1.set_ylabel('F1 Score')
        ax1.set_xlabel('Document Complexity')
        
        # Search time by complexity
        complexity_time = self.df.groupby('doc_complexity')['search_time'].mean()
        complexity_time.plot(kind='bar', ax=ax2, color='lightcoral', rot=45)
        ax2.set_title('Search Time by Document Complexity')
        ax2.set_ylabel('Search Time (seconds)')
        ax2.set_xlabel('Document Complexity')
        
        # Embedding model performance by complexity
        embedding_complexity = self.df.groupby(['embedding_model', 'doc_complexity'])['f1_score'].mean().unstack()
        embedding_complexity.plot(kind='bar', ax=ax3, rot=45)
        ax3.set_title('Embedding Model Performance by Complexity')
        ax3.set_ylabel('F1 Score')
        ax3.legend(title='Document Complexity', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Precision vs Recall by complexity
        for complexity in self.df['doc_complexity'].cat.categories:
            data = self.df[self.df['doc_complexity'] == complexity]
            ax4.scatter(data['precision'], data['recall'], label=complexity, alpha=0.6)
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision vs Recall by Document Complexity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        domain_file = self.images_dir / "domain_specific_analysis.png"
        plt.savefig(domain_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Created domain-specific analysis: {domain_file}")
        return complexity_overall
    
    def _create_category_table(self, category_data):
        """Create LaTeX table for category performance"""
        table = """\\begin{table}
\\caption{Performance by Clinical Query Category}
\\label{tab:category_performance}
\\begin{tabular}{lrrrr}
\\toprule
Category & F1 Score & Precision & Recall & Search Time (s) \\\\
\\midrule"""
        
        for _, row in category_data.iterrows():
            table += f"\n{row['category']} & {row['f1_score_mean']:.3f} & {row['precision_mean']:.3f} & {row['recall_mean']:.3f} & {row['search_time_mean']:.1f} \\\\"
        
        table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def _create_consistency_table(self, consistency_data):
        """Create LaTeX table for consistency analysis"""
        table = """\\begin{table}
\\caption{Model Consistency and Reliability Analysis}
\\label{tab:consistency_analysis}
\\begin{tabular}{lrrr}
\\toprule
Model Combination & Reliability Score & Good Performance Rate & F1 CV (\\%) \\\\
\\midrule"""
        
        for _, row in consistency_data.iterrows():
            table += f"\n{row['model_combination']} & {row['reliability_f1']:.3f} & {row['good_performance_rate']:.3f} & {row['f1_cv']:.1f} \\\\"
        
        table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def _create_pareto_table(self, pareto_data):
        """Create LaTeX table for Pareto optimal models"""
        table = """\\begin{table}
\\caption{Pareto Optimal Model Configurations}
\\label{tab:pareto_optimal}
\\begin{tabular}{lrrr}
\\toprule
Model Combination & F1 Score & Search Time (s) & Throughput (QPM) \\\\
\\midrule"""
        
        for _, row in pareto_data.iterrows():
            table += f"\n{row['model_combination']} & {row['f1_score_mean']:.3f} & {row['search_time_mean']:.1f} & {row['throughput_qpm']:.1f} \\\\"
        
        table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def _create_failure_analysis_table(self, failure_data):
        """Create LaTeX table for failure analysis"""
        table = """\\begin{table}
\\caption{Model Reliability Analysis (Lowest Failure Rates)}
\\label{tab:failure_analysis}
\\begin{tabular}{lrrr}
\\toprule
Model Combination & Failure Rate & Mean F1 Score & F1 Std Dev \\\\
\\midrule"""
        
        for _, row in failure_data.iterrows():
            table += f"\n{row['model_combination']} & {row['failure_rate']:.3f} & {row['mean_f1']:.3f} & {row['std_f1']:.3f} \\\\"
        
        table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def run_complete_analysis(self):
        """Run all advanced analyses"""
        print("Running complete advanced statistical analysis...")
        
        if not self.load_data():
            return
        
        # Run all analyses
        results = {}
        
        # 1. Create additional heatmaps
        self.create_precision_recall_heatmaps()
        
        # 2. Clinical category analysis
        results['categories'] = self.analyze_clinical_categories()
        
        # 3. Consistency and reliability
        results['consistency'] = self.analyze_consistency_reliability()
        
        # 4. Efficiency frontiers
        results['pareto'], results['efficient'] = self.analyze_efficiency_frontiers()
        
        # 5. Error pattern analysis
        results['failures'], results['variance'] = self.analyze_error_patterns()
        
        # 6. Domain-specific analysis
        results['domain'] = self.domain_specific_analysis()
        
        print("\n=== ADVANCED ANALYSIS SUMMARY ===")
        print(f"Created precision and recall heatmaps")
        print(f"Analyzed {len(results['categories'])} clinical categories")
        print(f"Identified {len(results['pareto'])} Pareto optimal configurations")
        print(f"Generated comprehensive reliability and error analysis")
        print(f"All visualizations and tables saved to {self.output_dir}")
        
        return results

if __name__ == "__main__":
    analyzer = AdvancedClinicalRAGAnalysis()
    results = analyzer.run_complete_analysis()