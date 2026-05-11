-- =============================================================================
-- ShiftMetrics — EDA with SparkSQL
-- Author: ShiftMetrics Analytics — EAFIT SI7006 Trabajo 3
-- Layer: Silver (read) → Gold (write)
-- =============================================================================
-- 8 analytical queries materialized as gold KPI tables.
-- Each KPI is a CREATE OR REPLACE TABLE so the notebook is idempotent.
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 1 — Defect distribution by project (PROMISE)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_defect_distribution_by_project
COMMENT 'Defect rate, density and total bugs per Apache project (PROMISE).'
AS
SELECT
    _project                                                                   AS project,
    COUNT(*)                                                                   AS modules,
    SUM(is_buggy)                                                              AS buggy_modules,
    ROUND(100.0 * SUM(is_buggy) / COUNT(*), 2)                                 AS pct_buggy,
    SUM(bug)                                                                   AS total_bugs,
    ROUND(AVG(bug_density), 5)                                                 AS avg_bug_density,
    ROUND(PERCENTILE_APPROX(bug_density, 0.5), 5)                              AS median_bug_density,
    ROUND(AVG(loc), 1)                                                         AS avg_loc,
    ROUND(STDDEV(loc), 1)                                                      AS std_loc
FROM workspace.shiftmetrics_silver.code_metrics_clean
GROUP BY _project
ORDER BY pct_buggy DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 2 — Metric correlations with the ML target (Pearson via SQL)
-- Spark SQL ships CORR; we compute correlation of each candidate feature with bug.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_metric_correlations_with_target
COMMENT 'Pearson correlation of each numeric feature with bug count and is_buggy. Used to prioritize ML features.'
AS
SELECT 'wmc'   AS feature, ROUND(CORR(wmc,   bug), 4) AS corr_with_bug, ROUND(CORR(wmc,   is_buggy), 4) AS corr_with_is_buggy FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'dit',                ROUND(CORR(dit,   bug), 4),                    ROUND(CORR(dit,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'noc',                ROUND(CORR(noc,   bug), 4),                    ROUND(CORR(noc,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'cbo',                ROUND(CORR(cbo,   bug), 4),                    ROUND(CORR(cbo,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'rfc',                ROUND(CORR(rfc,   bug), 4),                    ROUND(CORR(rfc,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'lcom',               ROUND(CORR(lcom,  bug), 4),                    ROUND(CORR(lcom,  is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'ca',                 ROUND(CORR(ca,    bug), 4),                    ROUND(CORR(ca,    is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'ce',                 ROUND(CORR(ce,    bug), 4),                    ROUND(CORR(ce,    is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'npm',                ROUND(CORR(npm,   bug), 4),                    ROUND(CORR(npm,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'lcom3',              ROUND(CORR(lcom3, bug), 4),                    ROUND(CORR(lcom3, is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'loc',                ROUND(CORR(loc,   bug), 4),                    ROUND(CORR(loc,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'dam',                ROUND(CORR(dam,   bug), 4),                    ROUND(CORR(dam,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'moa',                ROUND(CORR(moa,   bug), 4),                    ROUND(CORR(moa,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'mfa',                ROUND(CORR(mfa,   bug), 4),                    ROUND(CORR(mfa,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'cam',                ROUND(CORR(cam,   bug), 4),                    ROUND(CORR(cam,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'ic',                 ROUND(CORR(ic,    bug), 4),                    ROUND(CORR(ic,    is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'cbm',                ROUND(CORR(cbm,   bug), 4),                    ROUND(CORR(cbm,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'amc',                ROUND(CORR(amc,   bug), 4),                    ROUND(CORR(amc,   is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'max_cc',             ROUND(CORR(max_cc,bug), 4),                    ROUND(CORR(max_cc,is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean UNION ALL
SELECT 'avg_cc',             ROUND(CORR(avg_cc,bug), 4),                    ROUND(CORR(avg_cc,is_buggy), 4) FROM workspace.shiftmetrics_silver.code_metrics_clean;

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 3 — Top-25 buggiest modules (worst offenders for refactoring)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_top_buggy_modules
COMMENT 'Top 25 modules ranked by absolute bug count and bug density. Targets for refactoring.'
AS
SELECT
    _project                AS project,
    _version                AS version,
    name                    AS module,
    bug                     AS bugs,
    loc                     AS lines_of_code,
    ROUND(bug_density, 5)   AS bug_density,
    wmc, rfc, cbo, lcom,
    DENSE_RANK() OVER (ORDER BY bug DESC, bug_density DESC) AS rank_global
FROM workspace.shiftmetrics_silver.code_metrics_clean
WHERE bug > 0
ORDER BY bug DESC, bug_density DESC
LIMIT 25;

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 4 — Defect density quintile distribution (skewness profiling)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_defect_density_quintiles
COMMENT 'Quintile breakdown of bug_density per project. Reveals whether bugs are concentrated or spread.'
AS
WITH ranked AS (
    SELECT _project, bug_density,
           NTILE(5) OVER (PARTITION BY _project ORDER BY bug_density) AS quintile
    FROM workspace.shiftmetrics_silver.code_metrics_clean
    WHERE bug_density > 0
)
SELECT
    _project       AS project,
    quintile,
    COUNT(*)       AS modules,
    ROUND(MIN(bug_density), 5) AS min_density,
    ROUND(MAX(bug_density), 5) AS max_density,
    ROUND(AVG(bug_density), 5) AS avg_density
FROM ranked
GROUP BY _project, quintile
ORDER BY _project, quintile;

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 5 — Throughput evolution (Red Hat, yearly process metrics)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_throughput_yearly
COMMENT 'Issue throughput, resolution rate and cycle time evolution by year (2010-2024).'
AS
SELECT
    year_created                                                  AS year,
    COUNT(*)                                                      AS issues_created,
    SUM(is_resolved)                                              AS issues_resolved,
    ROUND(100.0 * SUM(is_resolved) / COUNT(*), 2)                 AS pct_resolved,
    ROUND(AVG(cycle_time_days), 1)                                AS avg_cycle_days,
    PERCENTILE_APPROX(cycle_time_days, 0.5)                       AS median_cycle_days,
    PERCENTILE_APPROX(cycle_time_days, 0.95)                      AS p95_cycle_days,
    COUNT(DISTINCT project_key)                                   AS active_systems
FROM workspace.shiftmetrics_silver.process_issues_clean
WHERE year_created BETWEEN 2010 AND 2024
GROUP BY year_created
ORDER BY year_created;

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 6 — Cycle time by issue type (effort signature per Jira type)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_cycle_time_by_issue_type
COMMENT 'Cycle time distribution per Jira issue_type. Identifies which work items dominate cycle.'
AS
SELECT
    issue_type,
    COUNT(*)                                          AS issues,
    SUM(is_resolved)                                  AS resolved,
    ROUND(100.0 * SUM(is_resolved) / COUNT(*), 1)     AS pct_resolved,
    ROUND(AVG(cycle_time_days), 1)                    AS avg_cycle_days,
    PERCENTILE_APPROX(cycle_time_days, 0.5)           AS median_cycle_days,
    PERCENTILE_APPROX(cycle_time_days, 0.9)           AS p90_cycle_days
FROM workspace.shiftmetrics_silver.process_issues_clean
WHERE is_resolved = 1
GROUP BY issue_type
HAVING issues >= 100        -- filter long-tail noise
ORDER BY issues DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 7 — Top-20 Red Hat systems by volume (where the action is)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_redhat_top_systems_by_volume
COMMENT 'Top 20 Red Hat systems by issue volume. Production hotspots.'
AS
SELECT
    project_key                                       AS system_code,
    project_name                                      AS system_name,
    COUNT(*)                                          AS total_issues,
    SUM(CASE WHEN issue_type = 'Bug' THEN 1 ELSE 0 END)   AS bugs,
    ROUND(100.0 * SUM(CASE WHEN issue_type='Bug' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_bugs,
    ROUND(AVG(cycle_time_days), 1)                    AS avg_cycle_days,
    MIN(year_created)                                 AS first_year,
    MAX(year_created)                                 AS last_year
FROM workspace.shiftmetrics_silver.process_issues_clean
GROUP BY project_key, project_name
ORDER BY total_issues DESC
LIMIT 20;

-- ─────────────────────────────────────────────────────────────────────────────
-- KPI 8 — Cross-domain summary via dim_project (the showcase query)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE workspace.shiftmetrics_gold.kpi_cross_domain_summary
COMMENT 'Cross-domain SDLC view: code-level (PROMISE) and process-level (RedHat) projects unified via dim_project. Demonstrates lakehouse value.'
AS
WITH apache_view AS (
    SELECT d.project_id, d.source_system, d.domain_layer, d.project_name,
           COUNT(c.name)                                    AS records,
           SUM(c.is_buggy)                                  AS positive_target,
           ROUND(100.0 * SUM(c.is_buggy)/COUNT(c.name), 2)  AS positive_rate_pct,
           ROUND(AVG(c.bug_density), 5)                     AS avg_bug_density,
           NULL                                             AS avg_cycle_days
    FROM workspace.shiftmetrics_silver.dim_project d
    JOIN workspace.shiftmetrics_silver.code_metrics_clean c
      ON d.project_key = c._project AND d.source_system = 'apache'
    GROUP BY d.project_id, d.source_system, d.domain_layer, d.project_name
),
redhat_view AS (
    SELECT d.project_id, d.source_system, d.domain_layer, d.project_name,
           COUNT(p.issue_key)                               AS records,
           SUM(p.is_resolved)                               AS positive_target,
           ROUND(100.0 * SUM(p.is_resolved)/COUNT(p.issue_key), 2) AS positive_rate_pct,
           NULL                                             AS avg_bug_density,
           ROUND(AVG(p.cycle_time_days), 1)                 AS avg_cycle_days
    FROM workspace.shiftmetrics_silver.dim_project d
    JOIN workspace.shiftmetrics_silver.process_issues_clean p
      ON d.project_key = p.project_key AND d.source_system = 'redhat'
    GROUP BY d.project_id, d.source_system, d.domain_layer, d.project_name
)
SELECT * FROM apache_view
UNION ALL
SELECT * FROM redhat_view;

-- ─────────────────────────────────────────────────────────────────────────────
-- Inspection — final Gold catalog state
-- ─────────────────────────────────────────────────────────────────────────────
SELECT table_name, table_type, comment
FROM workspace.information_schema.tables
WHERE table_schema = 'shiftmetrics_gold'
  AND table_type = 'MANAGED'
ORDER BY table_name;
