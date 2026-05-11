-- ============================================================================
-- ShiftMetrics Lakehouse — Bootstrap (idempotent, run-many-times-safe)
-- Layer: Foundation
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS workspace.shiftmetrics_bronze
  COMMENT 'Raw ingested data — immutable source of truth (PROMISE + RedHat Jira)';

CREATE SCHEMA IF NOT EXISTS workspace.shiftmetrics_silver
  COMMENT 'Cleansed and conformed data — schema enforced, deduplicated, typed';

CREATE SCHEMA IF NOT EXISTS workspace.shiftmetrics_gold
  COMMENT 'Business KPIs, EDA outputs and ML model artifacts';

CREATE VOLUME IF NOT EXISTS workspace.shiftmetrics_bronze.lakehouse_vol
  COMMENT 'Unified Volume: /landing for raw files, /_checkpoints, /_artifacts';
