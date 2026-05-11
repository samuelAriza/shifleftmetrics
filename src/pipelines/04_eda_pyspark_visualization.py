# Databricks notebook source
# MAGIC %md
# MAGIC # [EDA] PySpark + matplotlib
# MAGIC
# MAGIC **Layer:** Silver (read) + Gold (read KPIs)
# MAGIC **Author:** ShiftMetrics Analytics — EAFIT SI7006 Trabajo 3
# MAGIC
# MAGIC ## What this notebook delivers
# MAGIC 1. Statistical profiling of silver tables (PySpark `summary()`)
# MAGIC 2. **8 publication-quality visualizations** of the gold KPIs:
# MAGIC    - Defect rate per Apache project
# MAGIC    - Bug density distribution per project (boxplot)
# MAGIC    - Top-15 most buggy modules
# MAGIC    - Feature correlation matrix (heatmap)
# MAGIC    - Issue throughput evolution (Red Hat 2010-2024)
# MAGIC    - Cycle time bands per year
# MAGIC    - Cycle time by issue type
# MAGIC    - Cross-domain comparison (PROMISE vs Red Hat)
# MAGIC
# MAGIC ## Methodology notes
# MAGIC - PySpark for all aggregations (distributed). `.toPandas()` only on small KPI tables.
# MAGIC - Helper `to_pandas_numeric()` normalizes `DecimalType` → `DoubleType` so matplotlib
# MAGIC   arithmetic works without errors (SparkSQL ROUND returns Decimal by default).

# COMMAND ----------

# Uncomment if seaborn is not pre-installed in your runtime (Free Edition Serverless ships it):
# %pip install seaborn
# dbutils.library.restartPython()

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, DoubleType

SILVER = "workspace.shiftmetrics_silver"
GOLD   = "workspace.shiftmetrics_gold"

# Reproducible plot style — academic-grade defaults
mpl.rcParams.update({
    "figure.dpi":           120,
    "savefig.dpi":           120,
    "savefig.bbox":          "tight",
    "font.family":           "DejaVu Sans",
    "font.size":             10,
    "axes.titlesize":        13,
    "axes.titleweight":      "bold",
    "axes.labelsize":        11,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.alpha":            0.25,
    "legend.frameon":        False,
    "legend.fontsize":       10,
})
sns.set_palette("deep")

print(f"  matplotlib: {mpl.__version__}")
print(f"  seaborn:    {sns.__version__}")
print(f"  spark:      {spark.version}")

# COMMAND ----------

# MAGIC %md ## [0] Helper — normalize Decimal -> Double before toPandas()

# COMMAND ----------

def to_pandas_numeric(sdf):
    """
    Convert a Spark DataFrame to Pandas, casting all DecimalType columns to DoubleType first.

    Why: SparkSQL ROUND() returns decimal(N,M). When read into Pandas, those columns
    become Python `decimal.Decimal` objects, which crash matplotlib (no Decimal × float).
    This helper casts them to double so the resulting Pandas df is fully numpy-friendly.
    """
    decimal_cols = [f.name for f in sdf.schema.fields if isinstance(f.dataType, DecimalType)]
    if decimal_cols:
        for c in decimal_cols:
            sdf = sdf.withColumn(c, F.col(c).cast(DoubleType()))
    return sdf.toPandas()

# COMMAND ----------

# MAGIC %md ## [1] Statistical Profiling — PySpark `summary()`

# COMMAND ----------

print("-> silver.code_metrics_clean — numerical summary:")
display(
    spark.table(f"{SILVER}.code_metrics_clean")
        .select("wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc", "bug",
                "is_buggy", "bug_density")
        .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
)

# COMMAND ----------

print("-> silver.process_issues_clean — temporal & cycle-time profile:")
display(
    spark.table(f"{SILVER}.process_issues_clean")
        .select("year_created", "is_resolved", "cycle_time_days", "age_days")
        .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
)

# COMMAND ----------

# MAGIC %md ## [2] Viz 1 — Defect rate per project (PROMISE)

# COMMAND ----------

pdf = to_pandas_numeric(spark.table(f"{GOLD}.kpi_defect_distribution_by_project"))

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#d62728" if p > 50 else "#ff7f0e" if p > 30 else "#2ca02c" for p in pdf["pct_buggy"]]
bars = ax.bar(pdf["project"], pdf["pct_buggy"], color=colors, edgecolor="black", linewidth=0.5)

for b, v in zip(bars, pdf["pct_buggy"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(50, ls="--", color="gray", alpha=0.6, label="50% threshold")
ax.set_title("Defect Rate per Apache Project (PROMISE)\nProportion of modules with at least 1 bug")
ax.set_ylabel("% buggy modules")
ax.set_xlabel("Project")
ax.set_ylim(0, float(pdf["pct_buggy"].max()) * 1.15)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## [3] Viz 2 — Bug density distribution (boxplot per project)

# COMMAND ----------

density_pdf = to_pandas_numeric(
    spark.table(f"{SILVER}.code_metrics_clean")
        .filter(F.col("bug_density") > 0)
        .select("_project", "bug_density")
)

fig, ax = plt.subplots(figsize=(13, 6))
order = density_pdf.groupby("_project")["bug_density"].median().sort_values().index.tolist()
sns.boxplot(data=density_pdf, x="_project", y="bug_density", order=order,
            palette="RdYlGn_r", ax=ax, showfliers=False, linewidth=0.8)

ax.set_yscale("log")
ax.set_title("Bug Density Distribution per Apache Project (log scale)\nBugs ÷ LOC, only modules with ≥1 bug")
ax.set_ylabel("Bug density (log scale)")
ax.set_xlabel("Project — sorted by median density")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## [4] Viz 3 — Top-15 most buggy modules

# COMMAND ----------

top_pdf = to_pandas_numeric(
    spark.table(f"{GOLD}.kpi_top_buggy_modules").limit(15)
)
top_pdf["module_short"] = top_pdf["module"].apply(lambda s: s.split(".")[-1] if "." in s else s)
top_pdf["label"]        = top_pdf["project"] + " • " + top_pdf["module_short"]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(top_pdf["label"][::-1], top_pdf["bugs"][::-1],
               color=sns.color_palette("Reds_r", n_colors=15), edgecolor="black", linewidth=0.4)

for b, v in zip(bars, top_pdf["bugs"][::-1]):
    ax.text(float(v) + 0.3, b.get_y() + b.get_height() / 2, f"{int(v)}",
            va="center", fontsize=9, fontweight="bold")

ax.set_title("Top-15 Most Buggy Modules across Apache Projects\nAbsolute bug count from PROMISE dataset")
ax.set_xlabel("Number of bugs")
ax.set_ylabel("Project • Module")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## [5] Viz 4 — Feature correlation heatmap with target

# COMMAND ----------

corr_pdf = to_pandas_numeric(spark.table(f"{GOLD}.kpi_metric_correlations_with_target"))
corr_pdf = corr_pdf.sort_values("corr_with_is_buggy", key=lambda s: s.abs(), ascending=False)

fig, ax = plt.subplots(figsize=(11, 9))
data = corr_pdf[["corr_with_bug", "corr_with_is_buggy"]].values
sns.heatmap(data, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
            xticklabels=["bug (count)", "is_buggy (binary)"],
            yticklabels=corr_pdf["feature"].tolist(),
            cbar_kws={"label": "Pearson correlation"}, ax=ax,
            linewidths=0.5, linecolor="white")

ax.set_title("Feature <-> ML Target Correlation\nSorted by |corr_with_is_buggy| descending")
ax.set_xlabel("Target")
ax.set_ylabel("Feature (C&K + Halstead + McCabe)")
plt.tight_layout()
plt.show()

print("\n-> Top 5 features by |correlation with is_buggy|:")
print(corr_pdf.head(5).to_string(index=False))

# COMMAND ----------

# MAGIC %md ## [6] Viz 5 — Issue throughput evolution (Red Hat 2010-2024)

# COMMAND ----------

thr_pdf = to_pandas_numeric(spark.table(f"{GOLD}.kpi_throughput_yearly"))

fig, ax1 = plt.subplots(figsize=(13, 6))
color1 = "#1f77b4"
ax1.bar(thr_pdf["year"], thr_pdf["issues_resolved"], color=color1,
        edgecolor="black", linewidth=0.4, label="Resolved")
ax1.bar(thr_pdf["year"], thr_pdf["issues_created"] - thr_pdf["issues_resolved"],
        bottom=thr_pdf["issues_resolved"], color="#aec7e8", edgecolor="black",
        linewidth=0.4, label="Open / unresolved")
ax1.set_xlabel("Year")
ax1.set_ylabel("Issues created", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
color2 = "#d62728"
ax2.plot(thr_pdf["year"], thr_pdf["pct_resolved"], color=color2, marker="o",
         linewidth=2.2, label="% resolved")
ax2.set_ylabel("% resolved", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(40, 100)
ax2.grid(False)

fig.suptitle("Red Hat Issue Throughput & Resolution Rate (2010–2024)\n"
             "8× volume growth; resolution rate degrading from 96% → 63%",
             fontsize=13, fontweight="bold")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## [7] Viz 6 — Cycle time percentiles per year

# COMMAND ----------

fig, ax = plt.subplots(figsize=(13, 6))
ax.fill_between(thr_pdf["year"], thr_pdf["median_cycle_days"], thr_pdf["p95_cycle_days"],
                alpha=0.25, color="#ff7f0e", label="median <-> p95 band")
ax.plot(thr_pdf["year"], thr_pdf["avg_cycle_days"], marker="o", linewidth=2,
        color="#d62728", label="avg cycle (days)")
ax.plot(thr_pdf["year"], thr_pdf["median_cycle_days"], marker="s", linewidth=2,
        color="#1f77b4", label="median cycle")

ax.set_title("Red Hat — Cycle Time Distribution per Year\nMean, median, and 95th percentile band")
ax.set_xlabel("Year")
ax.set_ylabel("Days from created → resolved")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## [8] Viz 7 — Cycle time by issue type

# COMMAND ----------

ct_pdf = to_pandas_numeric(spark.table(f"{GOLD}.kpi_cycle_time_by_issue_type"))
ct_pdf = ct_pdf.sort_values("median_cycle_days")

fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(ct_pdf))))
bars = ax.barh(ct_pdf["issue_type"], ct_pdf["median_cycle_days"],
               color=sns.color_palette("viridis_r", n_colors=len(ct_pdf)),
               edgecolor="black", linewidth=0.4)
for b, v, n in zip(bars, ct_pdf["median_cycle_days"], ct_pdf["issues"]):
    ax.text(float(v) + 1, b.get_y() + b.get_height() / 2,
            f"{int(v)}d  (n={int(n):,})", va="center", fontsize=9)

ax.set_title("Median Cycle Time by Jira Issue Type (Red Hat)\nLong-tail filtered: only types with ≥100 issues")
ax.set_xlabel("Median cycle time (days)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## [9] Viz 8 — Cross-domain summary (the showcase)

# COMMAND ----------

cross_pdf = to_pandas_numeric(spark.table(f"{GOLD}.kpi_cross_domain_summary"))

apache_top = cross_pdf[cross_pdf["source_system"] == "apache"].nlargest(11, "records")
redhat_top = cross_pdf[cross_pdf["source_system"] == "redhat"].nlargest(10, "records")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# LEFT — Apache: pct_buggy ranking
ax = axes[0]
sub = apache_top.sort_values("positive_rate_pct", ascending=True)
bars = ax.barh(sub["project_name"], sub["positive_rate_pct"],
               color=sns.color_palette("Reds_r", n_colors=len(sub)),
               edgecolor="black", linewidth=0.4)
for b, v in zip(bars, sub["positive_rate_pct"]):
    ax.text(float(v) + 0.5, b.get_y() + b.get_height() / 2, f"{float(v):.1f}%",
            va="center", fontsize=9, fontweight="bold")
ax.set_title("Apache (PROMISE) — % buggy modules\nCode-level signal: defect proneness")
ax.set_xlabel("% buggy modules")

# RIGHT — Red Hat: avg cycle days
ax = axes[1]
sub = redhat_top.sort_values("avg_cycle_days", ascending=True)
bars = ax.barh(sub["project_name"].str.slice(0, 30),
               sub["avg_cycle_days"],
               color=sns.color_palette("Blues_r", n_colors=len(sub)),
               edgecolor="black", linewidth=0.4)
for b, v in zip(bars, sub["avg_cycle_days"]):
    ax.text(float(v) + 1, b.get_y() + b.get_height() / 2, f"{float(v):.0f}d",
            va="center", fontsize=9, fontweight="bold")
ax.set_title("Red Hat (Jira) — Avg cycle time per system (top 10 by volume)\nProcess-level signal: throughput friction")
ax.set_xlabel("Average cycle time (days)")

fig.suptitle("Cross-Domain SDLC View — Code Defects <-> Process Friction",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## [10] Summary — what the EDA tells us

# COMMAND ----------

print("\n[*] EDA KEY FINDINGS (auto-generated from gold KPIs)\n")

top3 = pdf.nlargest(3, "pct_buggy")
print("-> PROMISE — Most defect-prone Apache projects:")
for _, r in top3.iterrows():
    print(f"  {r['project']:10s}  pct_buggy={float(r['pct_buggy']):>5.1f}%  "
          f"({int(r['buggy_modules'])}/{int(r['modules'])} modules, total bugs={int(r['total_bugs'])})")

print("\n-> PROMISE — Top 3 features correlated with is_buggy:")
for _, r in corr_pdf.head(3).iterrows():
    print(f"  {r['feature']:10s}  corr={float(r['corr_with_is_buggy']):>+.3f}  "
          f"(corr_with_bug={float(r['corr_with_bug']):>+.3f})")

recent = thr_pdf.tail(5)
print("\n-> Red Hat — Throughput trend last 5 years:")
for _, r in recent.iterrows():
    print(f"  {int(r['year'])}  issues={int(r['issues_created']):>6,}  "
          f"resolved={float(r['pct_resolved']):>4.1f}%  avg_cycle={float(r['avg_cycle_days']):>5.1f}d")

n_apache = (cross_pdf["source_system"] == "apache").sum()
n_redhat = (cross_pdf["source_system"] == "redhat").sum()
print("\n-> Cross-domain — projects analyzed via dim_project:")
print(f"  Apache:  {n_apache} projects, {int(cross_pdf[cross_pdf.source_system=='apache']['records'].sum()):>7,} code modules")
print(f"  Red Hat: {n_redhat} systems,  {int(cross_pdf[cross_pdf.source_system=='redhat']['records'].sum()):>7,} process issues")

print("\n[OK] EDA finished. KPIs in workspace.shiftmetrics_gold are ready for ML.")
