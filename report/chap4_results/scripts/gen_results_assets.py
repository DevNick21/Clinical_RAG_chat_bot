import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "RAG_chat_pipeline", "results")
TABLES_DIR = os.path.join(PROJECT_ROOT, "report", "chap4_results", "tables")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "report", "chap4_results", "images")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load data
rf = os.path.join(RESULTS_DIR, "results_dataframe.csv")
ef = os.path.join(RESULTS_DIR, "run_efficiency_and_safety.csv")

df = pd.read_csv(rf)
if os.path.exists(ef):
    eff = pd.read_csv(ef)
else:
    eff = pd.DataFrame(columns=[
                       "experiment_id", "throughput_qpm", "disclaimer_rate", "hallucination_rate"])

# Merge for convenience
merged = pd.merge(df, eff, on="experiment_id", how="left")

# ---------- All runs table (tabularx) ----------
# Select and sort columns
all_runs = merged.copy()
all_runs["pass_rate_pct"] = (
    all_runs["pass_rate"].astype(float) * 100.0).round(1)
all_runs["average_score"] = all_runs["average_score"].astype(float).round(3)
all_runs["avg_search_time"] = all_runs["avg_search_time"].astype(
    float).round(2)
all_runs["throughput_qpm"] = all_runs.get(
    "throughput_qpm", pd.Series([math.nan]*len(all_runs))).round(3)

cols = [
    "embedding_model", "llm_model", "pass_rate_pct", "average_score", "avg_search_time", "throughput_qpm"
]
all_runs = all_runs[cols].sort_values(
    by=["average_score", "pass_rate_pct"], ascending=[False, False])

# Build LaTeX tabularx content
lines = []
lines.append(
    "% Auto-generated from results_dataframe.csv and run_efficiency_and_safety.csv")
lines.append("\\begin{tabularx}{\\textwidth}{l l c c c c}")
lines.append("  \\toprule")
lines.append(
    "  Embedding & LLM & Pass rate (\\%) & Avg. score & Search time (s) & QPM \\")
lines.append("  \\midrule")
for _, r in all_runs.iterrows():
    emb = str(r["embedding_model"]).replace("_", "\\_")
    llm = str(r["llm_model"]).replace("_", "\\_")
    pr = f"{r['pass_rate_pct']:.1f}" if not math.isnan(
        r["pass_rate_pct"]) else "--"
    sc = f"{r['average_score']:.3f}" if not math.isnan(
        r["average_score"]) else "--"
    tm = f"{r['avg_search_time']:.2f}" if not math.isnan(
        r["avg_search_time"]) else "--"
    qpm = f"{r['throughput_qpm']:.3f}" if not pd.isna(
        r["throughput_qpm"]) else "--"
    lines.append(f"  {emb} & {llm} & {pr} & {sc} & {tm} & {qpm} \\")
lines.append("  \\bottomrule")
lines.append("\\end{tabularx}")

with open(os.path.join(TABLES_DIR, "all_runs_table.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ---------- Category averages table ----------
# Map category to column pairs
cats = [
    ("Header", "header_pass_rate", "header_avg_score"),
    ("Diagnoses", "diagnoses_pass_rate", "diagnoses_avg_score"),
    ("Procedures", "procedures_pass_rate", "procedures_avg_score"),
    ("Labs", "labs_pass_rate", "labs_avg_score"),
    ("Microbiology", "microbiology_pass_rate", "microbiology_avg_score"),
    ("Prescriptions", "prescriptions_pass_rate", "prescriptions_avg_score"),
    ("Comprehensive", "comprehensive_pass_rate", "comprehensive_avg_score"),
]

rows = []
for name, pass_col, score_col in cats:
    if pass_col in df.columns and score_col in df.columns:
        pr_mean = float(pd.to_numeric(
            df[pass_col], errors='coerce').mean() * 100.0)
        sc_mean = float(pd.to_numeric(df[score_col], errors='coerce').mean())
        rows.append((name, round(pr_mean, 1), round(sc_mean, 3)))
    else:
        rows.append((name, float('nan'), float('nan')))

lines2 = []
lines2.append(
    "% Auto-generated per-category averages from results_dataframe.csv")
lines2.append("\\begin{tabularx}{0.7\\textwidth}{l c c}")
lines2.append("  \\toprule")
lines2.append("  Category & Mean pass rate (\\%) & Mean avg. score \\")
lines2.append("  \\midrule")
for name, pr, sc in rows:
    pr_s = f"{pr:.1f}" if not math.isnan(pr) else "--"
    sc_s = f"{sc:.3f}" if not math.isnan(sc) else "--"
    lines2.append(f"  {name} & {pr_s} & {sc_s} \\")
lines2.append("  \\bottomrule")
lines2.append("\\end{tabularx}")

with open(os.path.join(TABLES_DIR, "category_averages.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines2))

# ---------- Figures ----------
plt.figure(figsize=(6.5, 4.0), dpi=150)
sns.scatterplot(data=df, x="avg_search_time", y="average_score",
                hue="embedding_model", s=45, alpha=0.9, edgecolor="none")
plt.xlabel("Average search time (s)")
plt.ylabel("Average score")
plt.legend(loc="best", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "time_vs_score.png"))
plt.close()

if "throughput_qpm" in merged.columns:
    plt.figure(figsize=(6.5, 4.0), dpi=150)
    sns.scatterplot(data=merged, x="throughput_qpm", y="average_score",
                    hue="llm_model", s=45, alpha=0.9, edgecolor="none")
    plt.xlabel("Throughput (QPM)")
    plt.ylabel("Average score")
    plt.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "quality_vs_throughput.png"))
    plt.close()

plt.figure(figsize=(6.5, 4.0), dpi=150)
sns.scatterplot(data=df, x="average_score", y="pass_rate",
                hue="llm_model", s=45, alpha=0.9, edgecolor="none")
plt.xlabel("Average score")
plt.ylabel("Pass rate")
plt.legend(loc="best", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "pass_rate_vs_score.png"))
plt.close()

print("Generated tables and figures:")
print(" -", os.path.join(TABLES_DIR, "all_runs_table.tex"))
print(" -", os.path.join(TABLES_DIR, "category_averages.tex"))
print(" -", os.path.join(IMAGES_DIR, "time_vs_score.png"))
print(" -", os.path.join(IMAGES_DIR, "quality_vs_throughput.png"))
print(" -", os.path.join(IMAGES_DIR, "pass_rate_vs_score.png"))
