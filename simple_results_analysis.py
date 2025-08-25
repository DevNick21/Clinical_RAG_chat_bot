"""
Simple Results Analysis for Clinical RAG System
Generates basic statistical analysis from the results data
"""

import csv
import json
from pathlib import Path

# File paths
RESULTS_DIR = Path("RAG_chat_pipeline/results")
TABLES_DIR = Path("report/chap4_results/tables")

# Create directories
TABLES_DIR.mkdir(exist_ok=True, parents=True)

def read_csv_file(filename):
    """Read CSV file and return data"""
    data = []
    file_path = RESULTS_DIR / filename
    
    if file_path.exists():
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    else:
        print(f"Warning: {filename} not found")
    
    return data

def calculate_stats(values):
    """Calculate basic statistics"""
    if not values:
        return {'mean': 0, 'min': 0, 'max': 0, 'count': 0}
    
    numeric_values = [float(v) for v in values if v and v != 'nan']
    
    if not numeric_values:
        return {'mean': 0, 'min': 0, 'max': 0, 'count': 0}
    
    return {
        'mean': sum(numeric_values) / len(numeric_values),
        'min': min(numeric_values),
        'max': max(numeric_values),
        'count': len(numeric_values)
    }

def main():
    print("Enhanced Clinical RAG Results Analysis")
    print("=" * 50)
    
    # Load main results data
    df = read_csv_file("results_dataframe.csv")
    efficiency_df = read_csv_file("run_efficiency_and_safety.csv")
    
    if not df:
        print("Error: Could not load results data")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Basic statistics
    scores = [row['average_score'] for row in df if row['average_score']]
    pass_rates = [row['pass_rate'] for row in df if row['pass_rate']]
    search_times = [row['avg_search_time'] for row in df if row['avg_search_time']]
    
    score_stats = calculate_stats(scores)
    pass_rate_stats = calculate_stats(pass_rates)
    search_time_stats = calculate_stats(search_times)
    
    print("\nOverall Performance Statistics:")
    print(f"  Average Score: {score_stats['mean']:.4f} (min: {score_stats['min']:.4f}, max: {score_stats['max']:.4f})")
    print(f"  Average Pass Rate: {pass_rate_stats['mean']:.4f} (min: {pass_rate_stats['min']:.4f}, max: {pass_rate_stats['max']:.4f})")
    print(f"  Average Search Time: {search_time_stats['mean']:.2f}s (min: {search_time_stats['min']:.2f}s, max: {search_time_stats['max']:.2f}s)")
    
    # Top performers
    print("\nTop 10 Performing Configurations:")
    sorted_configs = sorted(df, key=lambda x: float(x['average_score']) if x['average_score'] else 0, reverse=True)
    
    top_performers_data = []
    for i, config in enumerate(sorted_configs[:10]):
        embedding = config['embedding_model']
        llm = config['llm_model']
        score = float(config['average_score']) if config['average_score'] else 0
        pass_rate = float(config['pass_rate']) if config['pass_rate'] else 0
        search_time = float(config['avg_search_time']) if config['avg_search_time'] else 0
        
        print(f"  {i+1}. {embedding} + {llm}: Score {score:.4f}, Pass Rate {pass_rate:.2%}, Time {search_time:.1f}s")
        
        top_performers_data.append({
            'Rank': i+1,
            'Embedding': embedding,
            'LLM': llm,
            'Score': f"{score:.4f}",
            'Pass Rate': f"{pass_rate:.2%}",
            'Search Time (s)': f"{search_time:.1f}"
        })
    
    # Embedding model analysis
    print("\nEmbedding Model Performance:")
    embedding_performance = {}
    
    for row in df:
        embedding = row['embedding_model']
        score = float(row['average_score']) if row['average_score'] else 0
        
        if embedding not in embedding_performance:
            embedding_performance[embedding] = []
        embedding_performance[embedding].append(score)
    
    embedding_ranking = []
    for embedding, scores in embedding_performance.items():
        stats = calculate_stats(scores)
        print(f"  {embedding}: Avg {stats['mean']:.4f} ({stats['count']} configs)")
        embedding_ranking.append({
            'Embedding': embedding,
            'Avg Score': f"{stats['mean']:.4f}",
            'Configs': stats['count']
        })
    
    # LLM model analysis
    print("\nLLM Model Performance:")
    llm_performance = {}
    
    for row in df:
        llm = row['llm_model']
        score = float(row['average_score']) if row['average_score'] else 0
        
        if llm not in llm_performance:
            llm_performance[llm] = []
        llm_performance[llm].append(score)
    
    llm_ranking = []
    for llm, scores in llm_performance.items():
        stats = calculate_stats(scores)
        print(f"  {llm}: Avg {stats['mean']:.4f} ({stats['count']} configs)")
        llm_ranking.append({
            'LLM': llm,
            'Avg Score': f"{stats['mean']:.4f}",
            'Configs': stats['count']
        })
    
    # Generate LaTeX table for top performers
    latex_content = """\\begin{table}[!htbp]
\\centering
\\begin{small}
\\renewcommand\\arraystretch{1.1}
\\begin{tabular}{|c|l|l|c|c|c|}
\\hline
\\textbf{Rank} & \\textbf{Embedding} & \\textbf{LLM} & \\textbf{Score} & \\textbf{Pass Rate} & \\textbf{Time (s)} \\\\
\\hline
"""
    
    for performer in top_performers_data:
        latex_content += f"{performer['Rank']} & {performer['Embedding']} & {performer['LLM']} & {performer['Score']} & {performer['Pass Rate']} & {performer['Search Time (s)']} \\\\\n"
    
    latex_content += """\\hline
\\end{tabular}
\\end{small}
\\caption{Top 10 Performing Model Combinations}
\\label{tab:enhanced_top_performers}
\\end{table}
"""
    
    # Save LaTeX table
    with open(TABLES_DIR / "enhanced_top_performers.tex", 'w') as f:
        f.write(latex_content)
    
    # Generate embedding ranking table
    embedding_latex = """\\begin{table}[!htbp]
\\centering
\\begin{small}
\\renewcommand\\arraystretch{1.1}
\\begin{tabular}{|l|c|c|}
\\hline
\\textbf{Embedding Model} & \\textbf{Average Score} & \\textbf{Configurations} \\\\
\\hline
"""
    
    # Sort embedding ranking by score
    embedding_ranking.sort(key=lambda x: float(x['Avg Score']), reverse=True)
    
    for embedding in embedding_ranking:
        embedding_latex += f"{embedding['Embedding']} & {embedding['Avg Score']} & {embedding['Configs']} \\\\\n"
    
    embedding_latex += """\\hline
\\end{tabular}
\\end{small}
\\caption{Embedding Model Performance Ranking}
\\label{tab:embedding_ranking}
\\end{table}
"""
    
    with open(TABLES_DIR / "embedding_ranking.tex", 'w') as f:
        f.write(embedding_latex)
    
    # Generate summary statistics table
    stats_latex = """\\begin{table}[!htbp]
\\centering
\\begin{small}
\\renewcommand\\arraystretch{1.1}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Mean} & \\textbf{Min} & \\textbf{Max} \\\\
\\hline
"""
    
    stats_latex += f"Average Score & {score_stats['mean']:.4f} & {score_stats['min']:.4f} & {score_stats['max']:.4f} \\\\\n"
    stats_latex += f"Pass Rate & {pass_rate_stats['mean']:.4f} & {pass_rate_stats['min']:.4f} & {pass_rate_stats['max']:.4f} \\\\\n"
    stats_latex += f"Search Time (s) & {search_time_stats['mean']:.2f} & {search_time_stats['min']:.2f} & {search_time_stats['max']:.2f} \\\\\n"
    
    stats_latex += """\\hline
\\end{tabular}
\\end{small}
\\caption{Overall System Performance Statistics}
\\label{tab:system_stats}
\\end{table}
"""
    
    with open(TABLES_DIR / "system_statistics.tex", 'w') as f:
        f.write(stats_latex)
    
    print(f"\nAnalysis complete!")
    print(f"Generated LaTeX tables saved to: {TABLES_DIR}")
    print("  - enhanced_top_performers.tex")
    print("  - embedding_ranking.tex") 
    print("  - system_statistics.tex")

if __name__ == "__main__":
    main()