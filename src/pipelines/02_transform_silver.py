# Databricks notebook source
# MAGIC %md
# MAGIC # [SILVER] Layer — Data Quality + Conformed + Cross-Domain Bridge
# MAGIC
# MAGIC **Layer:** Silver (cleansed, conformed, ML-ready)
# MAGIC **Author:** ShiftMetrics Analytics — EAFIT SI7006 Trabajo 3
# MAGIC
# MAGIC ## Inputs (Bronze, read-only)
# MAGIC - `workspace.shiftmetrics_bronze.code_metrics_raw`     (15,775 rows)
# MAGIC - `workspace.shiftmetrics_bronze.process_issues_raw`   (504,640 rows)
# MAGIC
# MAGIC ## Outputs (Silver Delta)
# MAGIC | Table | Purpose | Constraints |
# MAGIC |---|---|---|
# MAGIC | `code_metrics_clean`   | PROMISE deduped, typed, +derived | `bug>=0`, `loc>=0`, `is_buggy IN (0,1)` |
# MAGIC | `process_issues_clean` | Red Hat deduped, dates parsed, +derived | `created<=now`, `cycle_time>=0`, `is_resolved IN (0,1)` |
# MAGIC | `dim_project`          | Cross-domain bridge dimension (Apache + Red Hat) | unique(project_id) |
# MAGIC | `_dq_metrics`          | Persistent DQ audit log per run | append-only |
# MAGIC
# MAGIC ## Data decisions applied
# MAGIC - **DD-002**: `RHD.csv` (1 system) excluded from bronze due to schema mismatch.
# MAGIC - **DD-003**: 121 rows with `resolved < created` dropped in silver (impossible cycle time).

# COMMAND ----------

import uuid
from datetime import datetime, timezone
from pyspark.sql import functions as F, Window as W
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, BooleanType, TimestampType, DateType,
)

BRONZE = "workspace.shiftmetrics_bronze"
SILVER = "workspace.shiftmetrics_silver"

RUN_ID    = str(uuid.uuid4())
RUN_TS    = datetime.now(timezone.utc)
RUN_TS_STR = RUN_TS.isoformat()

print(f"-> Silver transformation started")
print(f"  run_id:       {RUN_ID}")
print(f"  run_ts:       {RUN_TS_STR}")
print(f"  spark:        {spark.version}")

# COMMAND ----------

# MAGIC %md ## [0] DQ Framework — helper functions

# COMMAND ----------

# In-memory accumulator. Persisted at the end of the run to silver._dq_metrics.
_dq_log: list[dict] = []

def dq_record(table: str, check: str, level: str, expected, actual, status: str, details: str = ""):
    """Append a DQ check result to the run-level accumulator."""
    _dq_log.append({
        "run_id":     RUN_ID,
        "run_ts":     RUN_TS,
        "table_name": table,
        "check_name": check,
        "level":      level,           # 'critical' | 'warning' | 'info'
        "expected":   str(expected),
        "actual":     str(actual),
        "status":     status,          # 'PASS' | 'FAIL' | 'WARN'
        "details":    details,
    })
    icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(status, "[?]")
    print(f"    {icon} [{level:>8s}] {check:<40s} expected={expected}  actual={actual}  {details}")

def dq_count(df, table: str, check: str, expected: int, level: str = "info"):
    actual = df.count()
    status = "PASS" if actual == expected else ("FAIL" if level == "critical" else "WARN")
    dq_record(table, check, level, expected, actual, status)
    return actual

def add_constraint(table_fqn: str, name: str, expr: str) -> None:
    """Idempotent constraint add. Tolerates 'already exists' gracefully."""
    stmt = f"ALTER TABLE {table_fqn} ADD CONSTRAINT {name} CHECK ({expr})"
    try:
        spark.sql(stmt)
        print(f"    [OK] constraint added: {name}")
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg:
            print(f"    [EXISTS] constraint already exists: {name}")
        else:
            print(f"    [FAIL] failed to add {name}: {e}")
            raise

# COMMAND ----------

# MAGIC %md ## [1] Pre-flight — Read Bronze + DQ snapshot

# COMMAND ----------

bronze_promise = spark.table(f"{BRONZE}.code_metrics_raw")
bronze_redhat  = spark.table(f"{BRONZE}.process_issues_raw")

print("-> Bronze snapshot (read-only):")
dq_count(bronze_promise, "code_metrics_raw",   "bronze_row_count_promise", expected=15775, level="critical")
dq_count(bronze_redhat,  "process_issues_raw", "bronze_row_count_redhat",  expected=504640, level="critical")

# COMMAND ----------

# MAGIC %md
# MAGIC ## [2] Silver — `code_metrics_clean`
# MAGIC
# MAGIC ### Transformations
# MAGIC 1. Drop rows with NULL critical fields (`name`, `loc`, `bug`)
# MAGIC 2. Cast `bug` to non-negative integer
# MAGIC 3. Deduplicate by natural key `(name, _project, _version)` keeping latest ingestion
# MAGIC 4. Derive `is_buggy` (binary ML target), `bug_density` (`bug/loc`)
# MAGIC 5. Drop audit-only `_source_file` (kept in bronze for lineage)
# MAGIC 6. Add silver audit cols `_silver_run_id`, `_silver_ingested_at`

# COMMAND ----------

# Step 2.1 — Pre-DQ on bronze
print("-> DQ on bronze.code_metrics_raw before cleaning:")
nulls_name = bronze_promise.filter(F.col("name").isNull()).count()
nulls_loc  = bronze_promise.filter(F.col("loc").isNull()).count()
nulls_bug  = bronze_promise.filter(F.col("bug").isNull()).count()
neg_bug    = bronze_promise.filter(F.col("bug") < 0).count()

dq_record("code_metrics_clean", "nulls_in_name", "critical", 0, nulls_name, "PASS" if nulls_name == 0 else "FAIL")
dq_record("code_metrics_clean", "nulls_in_loc",  "critical", 0, nulls_loc,  "PASS" if nulls_loc  == 0 else "FAIL")
dq_record("code_metrics_clean", "nulls_in_bug",  "critical", 0, nulls_bug,  "PASS" if nulls_bug  == 0 else "FAIL")
dq_record("code_metrics_clean", "negative_bug",  "critical", 0, neg_bug,    "PASS" if neg_bug    == 0 else "FAIL")

# Step 2.2 — Cleaning + dedup + derivations
nk_window = W.partitionBy("name", "_project", "_version").orderBy(F.col("_ingestion_ts").desc())

silver_promise = (
    bronze_promise
        .filter(F.col("name").isNotNull() & F.col("loc").isNotNull() & F.col("bug").isNotNull())
        .filter(F.col("bug") >= 0)
        # Dedup on natural key
        .withColumn("_rn", F.row_number().over(nk_window))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_source_file")
        # Derived columns
        .withColumn("is_buggy",
                    F.when(F.col("bug") > 0, F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()))
        .withColumn("bug_density",
                    F.when(F.col("loc") > 0, F.col("bug") / F.col("loc")).otherwise(F.lit(0.0)))
        # Silver audit
        .withColumn("_silver_run_id",      F.lit(RUN_ID))
        .withColumn("_silver_ingested_at", F.lit(RUN_TS).cast(TimestampType()))
)

# Step 2.3 — DQ on silver before write
print("-> DQ on silver.code_metrics_clean before write:")
silver_promise_count = silver_promise.count()
dup_count = (
    silver_promise
        .groupBy("name", "_project", "_version").count()
        .filter(F.col("count") > 1).count()
)
dq_record("code_metrics_clean", "post_dedup_row_count",   "info",     "<= 15775", silver_promise_count, "PASS")
dq_record("code_metrics_clean", "duplicate_natural_keys", "critical", 0, dup_count, "PASS" if dup_count == 0 else "FAIL")
dq_record("code_metrics_clean", "rows_dropped_in_silver", "info",
          f"= {15775 - silver_promise_count}", 15775 - silver_promise_count, "PASS")

# Step 2.4 — Write silver
(
    silver_promise.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("_project")
        .saveAsTable(f"{SILVER}.code_metrics_clean")
)

spark.sql(f"""
    ALTER TABLE {SILVER}.code_metrics_clean SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = true,
        'delta.autoOptimize.autoCompact'   = true
    )
""")

# Step 2.5 — Add Delta CHECK constraints
print("-> Applying Delta CHECK constraints to code_metrics_clean:")
CONSTRAINTS_PROMISE = [
    ("bug_non_negative", "bug >= 0"),
    ("loc_non_negative", "loc >= 0"),
    ("is_buggy_binary",  "is_buggy IN (0, 1)"),
]
for cname, expr in CONSTRAINTS_PROMISE:
    add_constraint(f"{SILVER}.code_metrics_clean", cname, expr)

spark.sql(f"OPTIMIZE {SILVER}.code_metrics_clean ZORDER BY (is_buggy, _version)")

spark.sql(f"""
    COMMENT ON TABLE {SILVER}.code_metrics_clean IS
    'PROMISE clean — deduped on (name,_project,_version), schema-enforced, with derived columns is_buggy (ML target) and bug_density. CHECK constraints: bug>=0, loc>=0, is_buggy IN (0,1).'
""")

print(f"  [OK] Wrote {SILVER}.code_metrics_clean — {silver_promise_count:,} rows")

# COMMAND ----------

display(spark.sql(f"""
    SELECT _project,
           COUNT(*)                                   AS rows,
           SUM(is_buggy)                              AS buggy_rows,
           ROUND(100.0 * SUM(is_buggy) / COUNT(*), 1) AS pct_buggy,
           ROUND(AVG(bug_density), 5)                 AS avg_bug_density,
           ROUND(AVG(loc), 0)                         AS avg_loc
    FROM {SILVER}.code_metrics_clean
    GROUP BY _project
    ORDER BY _project
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## [3] Silver — `process_issues_clean`
# MAGIC
# MAGIC ### Transformations
# MAGIC 1. Drop rows with NULL `created` (data corruption — date couldn't be parsed)
# MAGIC 2. **Drop rows with `resolved < created`** (physically impossible — see DD-003)
# MAGIC 3. Deduplicate by natural key `issue_key`
# MAGIC 4. Derive `cycle_time_days`, `year_created`, `month_created`, `is_resolved`, `age_days`
# MAGIC 5. Repartition by `year_created` (bounded cardinality unlike `_system_bucket`)

# COMMAND ----------

# Step 3.1 — Pre-DQ on bronze
print("-> DQ on bronze.process_issues_raw before cleaning:")
total_bronze_redhat = bronze_redhat.count()
nulls_issue_key  = bronze_redhat.filter(F.col("issue_key").isNull()).count()
nulls_created    = bronze_redhat.filter(F.col("created").isNull()).count()
future_created   = bronze_redhat.filter(F.col("created") > F.current_timestamp()).count()
inverted_dates   = bronze_redhat.filter(
    (F.col("resolved").isNotNull()) & (F.col("resolved") < F.col("created"))
).count()

dq_record("process_issues_clean", "nulls_in_issue_key",       "critical", 0, nulls_issue_key,
          "PASS" if nulls_issue_key == 0 else "FAIL")
dq_record("process_issues_clean", "nulls_in_created",         "warning",  0, nulls_created,
          "PASS" if nulls_created == 0 else "WARN", f"will drop {nulls_created} rows")
dq_record("process_issues_clean", "future_created_dates",     "warning",  0, future_created,
          "PASS" if future_created == 0 else "WARN")
dq_record("process_issues_clean", "inverted_created_resolved","warning",  0, inverted_dates,
          "PASS" if inverted_dates == 0 else "WARN", f"will drop {inverted_dates} rows (DD-003)")

# Step 3.2 — Cleaning + dedup + derivations
nk_window_redhat = W.partitionBy("issue_key").orderBy(F.col("_ingestion_ts").desc())

silver_redhat = (
    bronze_redhat
        .filter(F.col("issue_key").isNotNull())
        .filter(F.col("created").isNotNull())
        .filter(F.col("created") <= F.current_timestamp())
        # DD-003: drop physically impossible inverted dates
        .filter((F.col("resolved").isNull()) | (F.col("resolved") >= F.col("created")))
        # Dedup
        .withColumn("_rn", F.row_number().over(nk_window_redhat))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_source_file", "_system", "_system_bucket")
        # Derived
        .withColumn("year_created",   F.year("created"))
        .withColumn("month_created",  F.month("created"))
        .withColumn("is_resolved",
                    F.when(F.col("resolved").isNotNull(), F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()))
        .withColumn("cycle_time_days",
                    F.when(F.col("resolved").isNotNull(),
                           F.datediff("resolved", "created")).otherwise(F.lit(None).cast(IntegerType())))
        .withColumn("age_days",
                    F.datediff(F.current_timestamp(), F.col("created")))
        # Audit
        .withColumn("_silver_run_id",      F.lit(RUN_ID))
        .withColumn("_silver_ingested_at", F.lit(RUN_TS).cast(TimestampType()))
)

# Step 3.3 — DQ on silver before write
print("-> DQ on silver.process_issues_clean before write:")
silver_redhat_count = silver_redhat.count()
redhat_dup_count = (
    silver_redhat.groupBy("issue_key").count().filter(F.col("count") > 1).count()
)
dq_record("process_issues_clean", "post_dedup_row_count",      "info",     f"<= {total_bronze_redhat}", silver_redhat_count, "PASS")
dq_record("process_issues_clean", "duplicate_issue_keys",      "critical", 0, redhat_dup_count,
          "PASS" if redhat_dup_count == 0 else "FAIL")
dq_record("process_issues_clean", "rows_dropped_inverted_dates","info",
          f"= {inverted_dates}", inverted_dates, "PASS",
          "physically impossible: resolved < created (DD-003)")
dq_record("process_issues_clean", "rows_dropped_in_silver",    "info",
          f"= {total_bronze_redhat - silver_redhat_count}",
          total_bronze_redhat - silver_redhat_count, "PASS")

# Step 3.4 — Write silver
(
    silver_redhat.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("year_created")
        .saveAsTable(f"{SILVER}.process_issues_clean")
)

spark.sql(f"""
    ALTER TABLE {SILVER}.process_issues_clean SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = true,
        'delta.autoOptimize.autoCompact'   = true
    )
""")

# Step 3.5 — Constraints
print("-> Applying Delta CHECK constraints to process_issues_clean:")
CONSTRAINTS_REDHAT = [
    ("created_not_future",      "created <= current_timestamp()"),
    ("cycle_time_non_negative", "cycle_time_days IS NULL OR cycle_time_days >= 0"),
    ("is_resolved_binary",      "is_resolved IN (0, 1)"),
]
for cname, expr in CONSTRAINTS_REDHAT:
    add_constraint(f"{SILVER}.process_issues_clean", cname, expr)

spark.sql(f"OPTIMIZE {SILVER}.process_issues_clean ZORDER BY (project_key, issue_type)")

spark.sql(f"""
    COMMENT ON TABLE {SILVER}.process_issues_clean IS
    'Red Hat clean — deduped on issue_key, dates parsed, derived cycle_time_days/year_created/is_resolved. Inverted-date rows dropped per DD-003. CHECK constraints: created<=now, cycle_time>=0, is_resolved binary.'
""")

print(f"  [OK] Wrote {SILVER}.process_issues_clean — {silver_redhat_count:,} rows")

# COMMAND ----------

display(spark.sql(f"""
    SELECT year_created,
           COUNT(*)                                              AS issues,
           SUM(is_resolved)                                      AS resolved_issues,
           ROUND(100.0 * SUM(is_resolved) / COUNT(*), 1)         AS pct_resolved,
           ROUND(AVG(cycle_time_days), 1)                        AS avg_cycle_days,
           PERCENTILE_APPROX(cycle_time_days, 0.5)               AS median_cycle_days
    FROM {SILVER}.process_issues_clean
    WHERE year_created BETWEEN 2010 AND 2024
    GROUP BY year_created
    ORDER BY year_created
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## [4] Silver — `dim_project` (cross-domain bridge)

# COMMAND ----------

dim_apache = (
    spark.table(f"{SILVER}.code_metrics_clean")
        .groupBy("_project")
        .agg(
            F.count("*").alias("n_records"),
            F.countDistinct("_version").alias("n_versions"),
        )
        .select(
            F.concat(F.lit("apache:"), F.col("_project")).alias("project_id"),
            F.lit("apache").alias("source_system"),
            F.col("_project").alias("project_key"),
            F.initcap(F.col("_project")).alias("project_name"),
            F.lit("code").alias("domain_layer"),
            F.col("n_records"),
            F.col("n_versions"),
        )
)

dim_redhat = (
    spark.table(f"{SILVER}.process_issues_clean")
        .groupBy("project_key", "project_name")
        .agg(
            F.count("*").alias("n_records"),
            F.lit(None).cast(LongType()).alias("n_versions"),
        )
        .select(
            F.concat(F.lit("redhat:"), F.col("project_key")).alias("project_id"),
            F.lit("redhat").alias("source_system"),
            F.col("project_key"),
            F.col("project_name"),
            F.lit("process").alias("domain_layer"),
            F.col("n_records"),
            F.col("n_versions"),
        )
)

dim_project = (
    dim_apache.unionByName(dim_redhat)
        .withColumn("_silver_run_id",      F.lit(RUN_ID))
        .withColumn("_silver_ingested_at", F.lit(RUN_TS).cast(TimestampType()))
)

dim_project_count = dim_project.count()
print(f"  dim_project rows: {dim_project_count:,}")

dim_dup = dim_project.groupBy("project_id").count().filter(F.col("count") > 1).count()
dq_record("dim_project", "duplicate_project_id", "critical", 0, dim_dup, "PASS" if dim_dup == 0 else "FAIL")
dq_record("dim_project", "row_count",            "info",     "= 11 + ~250", dim_project_count, "PASS")

(
    dim_project.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{SILVER}.dim_project")
)

spark.sql(f"""
    ALTER TABLE {SILVER}.dim_project SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = true,
        'delta.autoOptimize.autoCompact'   = true
    )
""")
spark.sql(f"OPTIMIZE {SILVER}.dim_project ZORDER BY (source_system, domain_layer)")

spark.sql(f"""
    COMMENT ON TABLE {SILVER}.dim_project IS
    'Cross-domain bridge dimension. Unifies Apache (PROMISE, code-level) and Red Hat (Jira, process-level) under a single project_id. Enables joins across the SDLC: code metrics <-> process metrics.'
""")

display(spark.sql(f"""
    SELECT source_system, domain_layer,
           COUNT(*)         AS projects,
           SUM(n_records)   AS total_records
    FROM {SILVER}.dim_project
    GROUP BY source_system, domain_layer
    ORDER BY source_system
"""))

# COMMAND ----------

# MAGIC %md ## [5] Persist DQ run log -> `_dq_metrics`

# COMMAND ----------

dq_schema = StructType([
    StructField("run_id",     StringType(),    False),
    StructField("run_ts",     TimestampType(), False),
    StructField("table_name", StringType(),    False),
    StructField("check_name", StringType(),    False),
    StructField("level",      StringType(),    False),
    StructField("expected",   StringType(),    True),
    StructField("actual",     StringType(),    True),
    StructField("status",     StringType(),    False),
    StructField("details",    StringType(),    True),
])

dq_df = spark.createDataFrame(_dq_log, schema=dq_schema)
print(f"  Total DQ checks recorded: {dq_df.count()}")

(
    dq_df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(f"{SILVER}._dq_metrics")
)

spark.sql(f"""
    COMMENT ON TABLE {SILVER}._dq_metrics IS
    'Append-only DQ audit log. One row per (run_id, table, check). Used for evidence in audits and regression detection.'
""")

display(spark.sql(f"""
    SELECT level, status, COUNT(*) AS checks
    FROM {SILVER}._dq_metrics
    WHERE run_id = '{RUN_ID}'
    GROUP BY level, status
    ORDER BY level, status
"""))

display(spark.sql(f"""
    SELECT table_name, check_name, level, expected, actual, status
    FROM {SILVER}._dq_metrics
    WHERE run_id = '{RUN_ID}'
      AND status != 'PASS'
    ORDER BY level DESC, table_name
"""))

# COMMAND ----------

# MAGIC %md ## [6] Silver layer summary

# COMMAND ----------

display(spark.sql(f"""
    SELECT table_name, table_type, comment
    FROM workspace.information_schema.tables
    WHERE table_schema = 'shiftmetrics_silver'
      AND table_type = 'MANAGED'
    ORDER BY table_name
"""))

display(spark.sql(f"""
    SELECT '{BRONZE}.code_metrics_raw'        AS source,
           (SELECT COUNT(*) FROM {BRONZE}.code_metrics_raw)   AS bronze_rows,
           (SELECT COUNT(*) FROM {SILVER}.code_metrics_clean) AS silver_rows,
           ROUND(100.0 * (SELECT COUNT(*) FROM {SILVER}.code_metrics_clean) /
                          (SELECT COUNT(*) FROM {BRONZE}.code_metrics_raw), 2) AS retention_pct
    UNION ALL
    SELECT '{BRONZE}.process_issues_raw',
           (SELECT COUNT(*) FROM {BRONZE}.process_issues_raw),
           (SELECT COUNT(*) FROM {SILVER}.process_issues_clean),
           ROUND(100.0 * (SELECT COUNT(*) FROM {SILVER}.process_issues_clean) /
                          (SELECT COUNT(*) FROM {BRONZE}.process_issues_raw), 2)
"""))

print(f"\n-> Silver transformation finished")
print(f"  run_id:     {RUN_ID}")
print(f"  finished:   {datetime.now(timezone.utc).isoformat()}")
print(f"  PROMISE:    bronze {bronze_promise.count():,} -> silver {silver_promise_count:,}")
print(f"  Red Hat:    bronze {bronze_redhat.count():,} -> silver {silver_redhat_count:,}")
print(f"  Dim:        dim_project {dim_project_count:,} rows")
