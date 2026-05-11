# Databricks notebook source
# MAGIC %md
# MAGIC # [BRONZE] Ingestion — PROMISE + Red Hat Jira
# MAGIC
# MAGIC **Layer:** Bronze (raw, immutable)
# MAGIC **Author:** ShiftMetrics Analytics — EAFIT SI7006 Trabajo 3
# MAGIC
# MAGIC ## Sources
# MAGIC | Source | Path | Files | Schema |
# MAGIC |---|---|---|---|
# MAGIC | PROMISE Defect | `landing/promise/<project>/<project>-<version>.csv` | 41 | 22 cols (C&K + Halstead + McCabe) |
# MAGIC | Red Hat Public Jira | `landing/redhat/<system>.csv` | 250 | 9 cols (Jira issue lifecycle) |
# MAGIC
# MAGIC ## Sinks (Delta)
# MAGIC - `workspace.shiftmetrics_bronze.code_metrics_raw`     — partitioned by `_project`
# MAGIC - `workspace.shiftmetrics_bronze.process_issues_raw`   — partitioned by `_system_bucket`
# MAGIC
# MAGIC ## Audit columns
# MAGIC `_ingestion_ts`, `_source_file`, `_project`/`_system`, `_version` (PROMISE only)
# MAGIC
# MAGIC > **Note:** Unity Catalog forbids `input_file_name()`. We use the modern
# MAGIC > `_metadata.file_path` API, which is also strictly typed and richer.

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)

VOLUME_LANDING = "/Volumes/workspace/shiftmetrics_bronze/lakehouse_vol/landing"
BRONZE         = "workspace.shiftmetrics_bronze"

run_started = datetime.utcnow().isoformat() + "Z"
print(f"-> Bronze ingestion started: {run_started}")
print(f"  Spark version: {spark.version}")
print(f"  Catalog:       {spark.catalog.currentCatalog()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## [1] PROMISE — Code metrics (41 files → 1 Delta table)

# COMMAND ----------

promise_schema = StructType([
    StructField("name",    StringType(),  False),
    StructField("wmc",     DoubleType(),  True),
    StructField("dit",     DoubleType(),  True),
    StructField("noc",     DoubleType(),  True),
    StructField("cbo",     DoubleType(),  True),
    StructField("rfc",     DoubleType(),  True),
    StructField("lcom",    DoubleType(),  True),
    StructField("ca",      DoubleType(),  True),
    StructField("ce",      DoubleType(),  True),
    StructField("npm",     DoubleType(),  True),
    StructField("lcom3",   DoubleType(),  True),
    StructField("loc",     DoubleType(),  True),
    StructField("dam",     DoubleType(),  True),
    StructField("moa",     DoubleType(),  True),
    StructField("mfa",     DoubleType(),  True),
    StructField("cam",     DoubleType(),  True),
    StructField("ic",      DoubleType(),  True),
    StructField("cbm",     DoubleType(),  True),
    StructField("amc",     DoubleType(),  True),
    StructField("max_cc",  DoubleType(),  True),
    StructField("avg_cc",  DoubleType(),  True),
    StructField("bug",     IntegerType(), True),
])

# Note: select("*", "_metadata.file_path") replaces the legacy input_file_name()
promise_df = (
    spark.read
        .schema(promise_schema)
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .csv(f"{VOLUME_LANDING}/promise/*/*.csv")
        .select("*", F.col("_metadata.file_path").alias("_source_file"))
        .withColumn("_ingestion_ts", F.current_timestamp())
        # Path: .../landing/promise/<project>/<project>-<version>.csv
        .withColumn("_project",
                    F.regexp_extract("_source_file", r"/promise/([^/]+)/", 1))
        .withColumn("_version",
                    F.regexp_extract("_source_file", r"-([0-9][0-9.a-z]*)\.csv$", 1))
)

promise_count = promise_df.count()
print(f"  PROMISE rows: {promise_count:,}")
promise_df.printSchema()

# COMMAND ----------

(
    promise_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("_project")
        .saveAsTable(f"{BRONZE}.code_metrics_raw")
)

spark.sql(f"""
    ALTER TABLE {BRONZE}.code_metrics_raw
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = true,
        'delta.autoOptimize.autoCompact'   = true
    )
""")

spark.sql(f"OPTIMIZE {BRONZE}.code_metrics_raw ZORDER BY (_version)")

spark.sql(f"""
    COMMENT ON TABLE {BRONZE}.code_metrics_raw IS
    'PROMISE defect dataset — C&K + Halstead + McCabe code metrics for 11 Apache projects across multiple versions. Target: bug (defect count, binarizable). Partitioned by _project, Z-Ordered by _version.'
""")

print(f"  [OK] Wrote {BRONZE}.code_metrics_raw")

display(spark.sql(f"""
    SELECT _project,
           COUNT(*)                AS rows,
           COUNT(DISTINCT _version) AS versions,
           SUM(CASE WHEN bug > 0 THEN 1 ELSE 0 END) AS buggy_modules,
           ROUND(100.0 * SUM(CASE WHEN bug > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_buggy
    FROM {BRONZE}.code_metrics_raw
    GROUP BY _project
    ORDER BY _project
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## [2] Red Hat — Process issues (250 files → 1 Delta table)

# COMMAND ----------

# Schema validated in FASE 5 against 250 files (all match this header)
redhat_raw = (
    spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .csv(f"{VOLUME_LANDING}/redhat/*.csv")
        .select("*", F.col("_metadata.file_path").alias("_source_file"))
)

# Normalize column names → snake_case
column_map = {
    "Issue key":     "issue_key",
    "Issue Type":    "issue_type",
    "Status":        "status",
    "Project key":   "project_key",
    "Project name":  "project_name",
    "Project type":  "project_type",
    "Resolution":    "resolution",
    "Created":       "created_str",
    "Resolved":      "resolved_str",
}
for src, tgt in column_map.items():
    if src in redhat_raw.columns:
        redhat_raw = redhat_raw.withColumnRenamed(src, tgt)

# Cast dates: format dd/MM/yyyy HH:mm  (e.g. "16/06/2022 23:13")
DATE_FMT = "dd/MM/yyyy HH:mm"

redhat_df = (
    redhat_raw
        .withColumn("created",  F.to_timestamp("created_str",  DATE_FMT))
        .withColumn("resolved", F.to_timestamp("resolved_str", DATE_FMT))
        .drop("created_str", "resolved_str")
        .withColumn("_ingestion_ts", F.current_timestamp())
        .withColumn("_system",
                    F.regexp_extract("_source_file", r"/redhat/([^/]+)\.csv$", 1))
        # Bucket by first letter for partition pruning (250 distinct systems → too many)
        .withColumn("_system_bucket",
                    F.upper(F.substring("project_key", 1, 1)))
)

redhat_count = redhat_df.count()
print(f"  Red Hat rows: {redhat_count:,}")
redhat_df.printSchema()

# COMMAND ----------

(
    redhat_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("_system_bucket")
        .saveAsTable(f"{BRONZE}.process_issues_raw")
)

spark.sql(f"""
    ALTER TABLE {BRONZE}.process_issues_raw
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = true,
        'delta.autoOptimize.autoCompact'   = true
    )
""")

spark.sql(f"OPTIMIZE {BRONZE}.process_issues_raw ZORDER BY (project_key, status)")

spark.sql(f"""
    COMMENT ON TABLE {BRONZE}.process_issues_raw IS
    'Red Hat Public Jira PBIs 2001-2024 — process metrics raw. 250 systems × ~490k issues. Excludes RHD system (schema outlier, see DD-002). Partitioned by _system_bucket (first-letter), Z-Ordered by project_key + status.'
""")

print(f"  [OK] Wrote {BRONZE}.process_issues_raw")

display(spark.sql(f"""
    SELECT issue_type,
           COUNT(*) AS issues,
           ROUND(AVG(DATEDIFF(resolved, created)), 1) AS avg_days_to_resolve
    FROM {BRONZE}.process_issues_raw
    GROUP BY issue_type
    ORDER BY issues DESC
"""))

# COMMAND ----------

# MAGIC %md ## [3] Bronze layer summary

# COMMAND ----------

display(spark.sql(f"""
    SELECT table_name,
           table_type,
           comment
    FROM workspace.information_schema.tables
    WHERE table_schema = 'shiftmetrics_bronze'
      AND table_type   = 'MANAGED'
    ORDER BY table_name
"""))

print(f"\n-> Bronze ingestion finished at {datetime.utcnow().isoformat()}Z")
print(f"  PROMISE rows ingested: {promise_count:,}")
print(f"  Red Hat rows ingested: {redhat_count:,}")
print(f"  Total rows in Bronze:  {promise_count + redhat_count:,}")
