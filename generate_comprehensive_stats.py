#!/usr/bin/env python3
"""
Generate comprehensive statistics for the Clinical RAG evaluation results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_results():
    """Load the results dataframe"""
    results_path = Path("results/results_dataframe.csv")
    if not results_path.exists():
        print(f"Results file not found at {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} evaluations from {len(df['model_combination'].unique())} model combinations")
    return df

def generate_statistical_significance_tests(df):
    """Perform statistical significance tests"""
    # Test if embedding model choice significantly affects F1 scores
    embedding_groups = [group['f1_score'].values for name, group in df.groupby('embedding_model')]
    f_stat_embedding, p_val_embedding = stats.f_oneway(*embedding_groups)
    
    # Test if LLM choice significantly affects F1 scores
    llm_groups = [group['f1_score'].values for name, group in df.groupby('llm_model')]
    f_stat_llm, p_val_llm = stats.f_oneway(*llm_groups)
    
    return {
        'embedding_f_stat': f_stat_embedding,
        'embedding_p_value': p_val_embedding,
        'llm_f_stat': f_stat_llm,
        'llm_p_value': p_val_llm
    }

def calculate_performance_metrics(df):
    """Calculate comprehensive performance metrics"""
    # Overall statistics
    overall_stats = {
        'total_evaluations': len(df),
        'unique_combinations': len(df['model_combination'].unique()),
        'unique_embeddings': len(df['embedding_model'].unique()),
        'unique_llms': len(df['llm_model'].unique()),
        'avg_precision': df['precision'].mean(),
        'std_precision': df['precision'].std(),
        'avg_recall': df['recall'].mean(),
        'std_recall': df['recall'].std(),
        'avg_f1': df['f1_score'].mean(),
        'std_f1': df['f1_score'].std(),
        'avg_search_time': df['search_time'].mean(),
        'std_search_time': df['search_time'].std(),
        'median_search_time': df['search_time'].median(),
        'avg_docs_found': df['documents_found'].mean(),
        'std_docs_found': df['documents_found'].std()
    }
    
    # Coefficient of variation
    overall_stats['cv_precision'] = (overall_stats['std_precision'] / overall_stats['avg_precision']) * 100
    overall_stats['cv_recall'] = (overall_stats['std_recall'] / overall_stats['avg_recall']) * 100
    overall_stats['cv_f1'] = (overall_stats['std_f1'] / overall_stats['avg_f1']) * 100
    overall_stats['cv_search_time'] = (overall_stats['std_search_time'] / overall_stats['avg_search_time']) * 100
    
    # Percentiles for F1 score
    overall_stats['f1_25th'] = np.percentile(df['f1_score'], 25)
    overall_stats['f1_50th'] = np.percentile(df['f1_score'], 50)
    overall_stats['f1_75th'] = np.percentile(df['f1_score'], 75)
    overall_stats['f1_90th'] = np.percentile(df['f1_score'], 90)
    
    return overall_stats

def analyze_model_combinations(df):
    """Analyze performance by model combinations"""
    combo_stats = df.groupby('model_combination').agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'search_time': ['mean', 'std'],
        'documents_found': 'mean'
    }).round(3)
    
    # Flatten column names
    combo_stats.columns = ['_'.join(col).strip() for col in combo_stats.columns.values]
    combo_stats = combo_stats.reset_index()
    
    # Sort by F1 score
    combo_stats = combo_stats.sort_values('f1_score_mean', ascending=False)
    
    return combo_stats

def analyze_embedding_performance(df):
    """Detailed analysis of embedding model performance"""
    embedding_stats = df.groupby('embedding_model').agg({
        'f1_score': ['mean', 'std', 'count', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'search_time': ['mean', 'std', 'median']
    }).round(3)
    
    # Flatten column names
    embedding_stats.columns = ['_'.join(col).strip() for col in embedding_stats.columns.values]
    embedding_stats = embedding_stats.reset_index()
    
    # Sort by mean F1 score
    embedding_stats = embedding_stats.sort_values('f1_score_mean', ascending=False)
    
    # Calculate efficiency metrics
    embedding_stats['f1_per_second'] = embedding_stats['f1_score_mean'] / embedding_stats['search_time_mean']
    embedding_stats['efficiency_rank'] = embedding_stats['f1_per_second'].rank(ascending=False)
    
    return embedding_stats

def analyze_llm_performance(df):
    """Detailed analysis of LLM performance"""
    llm_stats = df.groupby('llm_model').agg({
        'f1_score': ['mean', 'std', 'count', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'search_time': ['mean', 'std', 'median']
    }).round(3)
    
    # Flatten column names
    llm_stats.columns = ['_'.join(col).strip() for col in llm_stats.columns.values]
    llm_stats = llm_stats.reset_index()
    
    # Sort by mean F1 score
    llm_stats = llm_stats.sort_values('f1_score_mean', ascending=False)
    
    # Calculate efficiency metrics
    llm_stats['f1_per_second'] = llm_stats['f1_score_mean'] / llm_stats['search_time_mean']
    llm_stats['efficiency_rank'] = llm_stats['f1_per_second'].rank(ascending=False)
    
    return llm_stats

def identify_top_performers(combo_stats, threshold=0.65):
    """Identify top performing configurations"""
    # High-quality configurations (top F1)
    top_f1 = combo_stats.nlargest(5, 'f1_score_mean')
    
    # Fast configurations with good performance
    good_performers = combo_stats[combo_stats['f1_score_mean'] >= threshold]
    fast_good = good_performers.nsmallest(5, 'search_time_mean') if not good_performers.empty else pd.DataFrame()
    
    # Balanced configurations (optimize F1/time ratio)
    combo_stats['efficiency_score'] = combo_stats['f1_score_mean'] / (combo_stats['search_time_mean'] / 60)  # F1 per minute
    balanced = combo_stats.nlargest(5, 'efficiency_score')
    
    return {
        'top_f1': top_f1,
        'fast_good': fast_good,
        'balanced': balanced
    }

def analyze_consistency(df):
    """Analyze consistency across different metrics"""
    # Calculate correlation between metrics
    correlations = df[['precision', 'recall', 'f1_score', 'search_time', 'documents_found']].corr()
    
    # Find most consistent models (lowest std relative to mean)
    combo_consistency = df.groupby('model_combination').agg({
        'f1_score': ['mean', 'std']
    })
    combo_consistency.columns = ['f1_mean', 'f1_std']
    combo_consistency['consistency_score'] = combo_consistency['f1_mean'] / combo_consistency['f1_std']
    combo_consistency = combo_consistency.sort_values('consistency_score', ascending=False)
    
    return correlations, combo_consistency

def create_statistical_summary_tables(output_dir="report/chap4_results/tables"):
    """Create additional statistical summary tables"""
    output_dir = Path(output_dir)
    
    df = load_results()
    if df is None:
        return
    
    # Generate all statistics
    overall_stats = calculate_performance_metrics(df)
    combo_stats = analyze_model_combinations(df)
    embedding_stats = analyze_embedding_performance(df)
    llm_stats = analyze_llm_performance(df)
    top_performers = identify_top_performers(combo_stats)
    correlations, consistency = analyze_consistency(df)
    significance_tests = generate_statistical_significance_tests(df)
    
    # Create enhanced system statistics table
    enhanced_system_stats = f"""\\begin{{table}}
\\caption{{Enhanced Evaluation System Statistics}}
\\label{{tab:enhanced_system_stats}}
\\begin{{tabular}}{{ll}}
\\toprule
Metric & Value \\\\
\\midrule
Total Evaluations & {overall_stats['total_evaluations']:,} \\\\
Model Combinations & {overall_stats['unique_combinations']} \\\\
Embedding Models & {overall_stats['unique_embeddings']} \\\\
Language Models & {overall_stats['unique_llms']} \\\\
\\midrule
\\textbf{{Performance Metrics}} & \\\\
Average Precision & {overall_stats['avg_precision']:.3f} ± {overall_stats['std_precision']:.3f} \\\\
Average Recall & {overall_stats['avg_recall']:.3f} ± {overall_stats['std_recall']:.3f} \\\\
Average F1-Score & {overall_stats['avg_f1']:.3f} ± {overall_stats['std_f1']:.3f} \\\\
F1 Median & {overall_stats['f1_50th']:.3f} \\\\
F1 90th Percentile & {overall_stats['f1_90th']:.3f} \\\\
\\midrule
\\textbf{{Efficiency Metrics}} & \\\\
Average Response Time & {overall_stats['avg_search_time']:.1f}s ± {overall_stats['std_search_time']:.1f}s \\\\
Median Response Time & {overall_stats['median_search_time']:.1f}s \\\\
Average Documents Retrieved & {overall_stats['avg_docs_found']:.1f} ± {overall_stats['std_docs_found']:.1f} \\\\
\\midrule
\\textbf{{Variability Analysis}} & \\\\
F1-Score CV & {overall_stats['cv_f1']:.1f}\\% \\\\
Precision CV & {overall_stats['cv_precision']:.1f}\\% \\\\
Search Time CV & {overall_stats['cv_search_time']:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    # Write enhanced system statistics
    with open(output_dir / "enhanced_system_statistics.tex", "w") as f:
        f.write(enhanced_system_stats)
    
    # Create statistical significance table
    significance_table = f"""\\begin{{table}}
\\caption{{Statistical Significance Analysis}}
\\label{{tab:statistical_significance}}
\\begin{{tabular}}{{lrrc}}
\\toprule
Component & F-Statistic & p-value & Significance \\\\
\\midrule
Embedding Model Effect & {significance_tests['embedding_f_stat']:.3f} & {significance_tests['embedding_p_value']:.3f} & {'***' if significance_tests['embedding_p_value'] < 0.001 else '**' if significance_tests['embedding_p_value'] < 0.01 else '*' if significance_tests['embedding_p_value'] < 0.05 else 'ns'} \\\\
LLM Model Effect & {significance_tests['llm_f_stat']:.3f} & {significance_tests['llm_p_value']:.3f} & {'***' if significance_tests['llm_p_value'] < 0.001 else '**' if significance_tests['llm_p_value'] < 0.01 else '*' if significance_tests['llm_p_value'] < 0.05 else 'ns'} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\textbf{{Significance levels:}} *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant"""
    
    with open(output_dir / "statistical_significance.tex", "w") as f:
        f.write(significance_table)
    
    # Create comprehensive model ranking table
    top_10 = combo_stats.head(10)
    ranking_table = f"""\\begin{{table}}
\\caption{{Comprehensive Model Performance Rankings (Top 10)}}
\\label{{tab:comprehensive_rankings}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Model Combination & F1 & Precision & Recall & Time (s) & Efficiency & Rank \\\\
\\midrule"""
    
    for idx, row in top_10.iterrows():
        efficiency = row['f1_score_mean'] / (row['search_time_mean'] / 60)  # F1 per minute
        ranking_table += f"\n{row['model_combination']} & {row['f1_score_mean']:.3f} & {row['precision_mean']:.3f} & {row['recall_mean']:.3f} & {row['search_time_mean']:.1f} & {efficiency:.2f} & {idx + 1} \\\\"
    
    ranking_table += f"""
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    with open(output_dir / "comprehensive_model_rankings.tex", "w") as f:
        f.write(ranking_table)
    
    print(f"Created enhanced statistics tables in {output_dir}")
    
    # Print summary for immediate use
    print("\n=== STATISTICAL SUMMARY FOR REPORT ===")
    print(f"Total evaluations: {overall_stats['total_evaluations']:,}")
    print(f"Model combinations tested: {overall_stats['unique_combinations']}")
    print(f"Average F1: {overall_stats['avg_f1']:.3f} ± {overall_stats['std_f1']:.3f}")
    print(f"F1 coefficient of variation: {overall_stats['cv_f1']:.1f}%")
    print(f"Average search time: {overall_stats['avg_search_time']:.1f}s")
    print(f"Search time CV: {overall_stats['cv_search_time']:.1f}%")
    
    print(f"\nBest F1 combination: {combo_stats.iloc[0]['model_combination']} (F1: {combo_stats.iloc[0]['f1_score_mean']:.3f})")
    print(f"Embedding model significance: F={significance_tests['embedding_f_stat']:.3f}, p={significance_tests['embedding_p_value']:.3f}")
    print(f"LLM model significance: F={significance_tests['llm_f_stat']:.3f}, p={significance_tests['llm_p_value']:.3f}")
    
    return {
        'overall_stats': overall_stats,
        'combo_stats': combo_stats,
        'embedding_stats': embedding_stats,
        'llm_stats': llm_stats,
        'significance_tests': significance_tests,
        'correlations': correlations
    }

if __name__ == "__main__":
    create_statistical_summary_tables()