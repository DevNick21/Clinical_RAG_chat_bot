#!/usr/bin/env python3
"""
Improved visualization script for Clinical RAG evaluation results
Fixes cramped model names and creates comprehensive statistics visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Set up style for better visibility
plt.style.use('default')
sns.set_palette("tab10")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

def load_and_process_data():
    """Load and process the results data"""
    results_path = Path("results/results_dataframe.csv")
    if not results_path.exists():
        print(f"Results file not found at {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} results from {len(df['model_combination'].unique())} model combinations")
    
    # Aggregate by model combination
    model_stats = df.groupby(['embedding_model', 'llm_model', 'model_combination']).agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'search_time': ['mean', 'std'],
        'documents_found': 'mean'
    }).round(3)
    
    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    model_stats = model_stats.reset_index()
    
    return df, model_stats

def create_f1_heatmap(model_stats, output_dir="report/chap4_results/images"):
    """Create heatmap for F1 scores with improved readability"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pivot table
    heatmap_data = model_stats.pivot(
        index='embedding_model', 
        columns='llm_model', 
        values='f1_score_mean'
    )
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with better formatting
    mask = heatmap_data.isnull()
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',
                center=heatmap_data.mean().mean(),
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "F1 Score"},
                mask=mask)
    
    plt.title('Model Performance Heatmap: Average F1 Score', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Language Model', fontsize=12, fontweight='bold')
    plt.ylabel('Embedding Model', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save the plot
    output_file = output_dir / "heatmap_f1_score.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created F1 score heatmap: {output_file}")
    return str(output_file)

def fix_time_vs_score_plot(model_stats, output_dir="report/chap4_results/images"):
    """Recreate time vs score plot with fixed model name display"""
    output_dir = Path(output_dir)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(model_stats['search_time_mean'], 
                         model_stats['f1_score_mean'],
                         s=80, alpha=0.7, c=range(len(model_stats)), cmap='tab20')
    
    # Add labels with improved formatting
    for idx, row in model_stats.iterrows():
        # Use abbreviated model names for readability
        embedding_short = row['embedding_model'].replace('sentence-transformers/', '').replace('microsoft/', '')
        llm_short = row['llm_model'].replace('microsoft/', '').replace('meta-llama/', '')
        label = f"{embedding_short}+{llm_short}"
        
        plt.annotate(label, 
                    (row['search_time_mean'], row['f1_score_mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('Average Search Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    plt.title('Performance vs Efficiency: F1 Score vs Search Time', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = output_dir / "time_vs_score.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Fixed time vs score plot: {output_file}")
    return str(output_file)

def create_quality_vs_throughput_plot(model_stats, output_dir="report/chap4_results/images"):
    """Create quality vs throughput plot with better formatting"""
    output_dir = Path(output_dir)
    
    # Calculate throughput (questions per minute) - assuming 20 questions per run
    # and search_time is per question
    questions_per_run = 20
    model_stats['throughput_qpm'] = questions_per_run * 60 / model_stats['search_time_mean']
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(model_stats['throughput_qpm'], 
                         model_stats['f1_score_mean'],
                         s=80, alpha=0.7, c=range(len(model_stats)), cmap='tab20')
    
    # Add labels with improved formatting
    for idx, row in model_stats.iterrows():
        # Use abbreviated model names for readability
        embedding_short = row['embedding_model'].replace('sentence-transformers/', '').replace('microsoft/', '')
        llm_short = row['llm_model'].replace('microsoft/', '').replace('meta-llama/', '')
        label = f"{embedding_short}+{llm_short}"
        
        plt.annotate(label, 
                    (row['throughput_qpm'], row['f1_score_mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('Throughput (Questions per Minute)', fontsize=12, fontweight='bold')
    plt.ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    plt.title('System Efficiency: F1 Score vs Throughput', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = output_dir / "quality_vs_throughput.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created quality vs throughput plot: {output_file}")
    return str(output_file)

def create_pass_rate_vs_score_plot(df, model_stats, output_dir="report/chap4_results/images"):
    """Create pass rate vs score plot by LLM with better formatting"""
    output_dir = Path(output_dir)
    
    # Calculate pass rate (assuming F1 > 0.6 is passing)
    pass_threshold = 0.6
    pass_rates = df.groupby(['llm_model', 'model_combination']).agg({
        'f1_score': lambda x: (x >= pass_threshold).mean() * 100
    }).reset_index()
    pass_rates.columns = ['llm_model', 'model_combination', 'pass_rate']
    
    # Merge with model stats
    combined_data = pass_rates.merge(
        model_stats[['model_combination', 'f1_score_mean']], 
        on='model_combination'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot colored by LLM
    llm_models = combined_data['llm_model'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(llm_models)))
    
    for i, llm in enumerate(llm_models):
        llm_data = combined_data[combined_data['llm_model'] == llm]
        plt.scatter(llm_data['pass_rate'], llm_data['f1_score_mean'],
                   label=llm.replace('microsoft/', '').replace('meta-llama/', ''),
                   alpha=0.7, s=80, color=colors[i])
    
    plt.xlabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    plt.title('Reliability vs Quality: Pass Rate vs F1 Score by LLM', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    output_file = output_dir / "pass_rate_vs_score.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created pass rate vs score plot: {output_file}")
    return str(output_file)

def generate_comprehensive_statistics(df, model_stats):
    """Generate comprehensive statistics for the report"""
    stats = {}
    
    # Overall statistics
    stats['total_evaluations'] = len(df)
    stats['model_combinations'] = len(df['model_combination'].unique())
    stats['avg_precision'] = df['precision'].mean()
    stats['avg_recall'] = df['recall'].mean()
    stats['avg_f1_score'] = df['f1_score'].mean()
    stats['avg_search_time'] = df['search_time'].mean()
    stats['std_f1_score'] = df['f1_score'].std()
    stats['std_precision'] = df['precision'].std()
    stats['std_recall'] = df['recall'].std()
    stats['std_search_time'] = df['search_time'].std()
    
    # Top performers
    top_f1 = model_stats.nlargest(1, 'f1_score_mean')
    stats['best_f1_combination'] = top_f1['model_combination'].iloc[0]
    stats['best_f1_score'] = top_f1['f1_score_mean'].iloc[0]
    stats['best_f1_precision'] = top_f1['precision_mean'].iloc[0]
    stats['best_f1_time'] = top_f1['search_time_mean'].iloc[0]
    
    # Fastest with good performance (F1 > 0.65)
    good_performers = model_stats[model_stats['f1_score_mean'] > 0.65]
    if not good_performers.empty:
        fastest_good = good_performers.nsmallest(1, 'search_time_mean')
        stats['fastest_good_combination'] = fastest_good['model_combination'].iloc[0]
        stats['fastest_good_f1'] = fastest_good['f1_score_mean'].iloc[0]
        stats['fastest_good_time'] = fastest_good['search_time_mean'].iloc[0]
    
    # Embedding model rankings
    embedding_stats = model_stats.groupby('embedding_model').agg({
        'f1_score_mean': 'mean',
        'precision_mean': 'mean',
        'search_time_mean': 'mean'
    }).sort_values('f1_score_mean', ascending=False)
    
    # LLM model performance
    llm_stats = model_stats.groupby('llm_model').agg({
        'f1_score_mean': 'mean',
        'precision_mean': 'mean',
        'search_time_mean': 'mean'
    }).sort_values('f1_score_mean', ascending=False)
    
    stats['embedding_rankings'] = embedding_stats
    stats['llm_rankings'] = llm_stats
    
    # Statistical significance tests could be added here
    # For now, providing coefficient of variation
    stats['f1_cv'] = (stats['std_f1_score'] / stats['avg_f1_score']) * 100
    stats['time_cv'] = (stats['std_search_time'] / stats['avg_search_time']) * 100
    
    return stats

def main():
    """Main function to generate all improved visualizations"""
    print("Loading and processing data...")
    data = load_and_process_data()
    if data is None:
        return
    
    df, model_stats = data
    
    print(f"\nDataset summary:")
    print(f"- Total evaluations: {len(df)}")
    print(f"- Model combinations: {len(df['model_combination'].unique())}")
    print(f"- Average F1 score: {df['f1_score'].mean():.3f} ± {df['f1_score'].std():.3f}")
    print(f"- Average search time: {df['search_time'].mean():.2f}s ± {df['search_time'].std():.2f}s")
    
    print("\nGenerating improved visualizations...")
    
    # Create F1 heatmap (missing from original)
    create_f1_heatmap(model_stats)
    
    # Fix existing plots with better model name display
    fix_time_vs_score_plot(model_stats)
    create_quality_vs_throughput_plot(model_stats)
    create_pass_rate_vs_score_plot(df, model_stats)
    
    # Generate comprehensive statistics
    stats = generate_comprehensive_statistics(df, model_stats)
    
    print("\nKey Statistics:")
    print(f"- Best F1 combination: {stats['best_f1_combination']} (F1: {stats['best_f1_score']:.3f})")
    if 'fastest_good_combination' in stats:
        print(f"- Fastest good combination: {stats['fastest_good_combination']} (F1: {stats['fastest_good_f1']:.3f}, Time: {stats['fastest_good_time']:.1f}s)")
    print(f"- F1 coefficient of variation: {stats['f1_cv']:.1f}%")
    print(f"- Time coefficient of variation: {stats['time_cv']:.1f}%")
    
    print("\nTop 5 Embedding Models by F1 Score:")
    print(stats['embedding_rankings'].head())
    
    print("\nTop 5 LLM Models by F1 Score:")
    print(stats['llm_rankings'].head())
    
    print("\nAll visualizations created successfully!")

if __name__ == "__main__":
    main()