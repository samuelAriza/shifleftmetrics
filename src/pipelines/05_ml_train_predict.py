# Databricks notebook source
# MAGIC %md
# MAGIC # [ML] ML Pipeline — Defect Prediction (Hybrid SparkML + sklearn)
# MAGIC
# MAGIC **Layer:** Silver (read) + Gold (write) + Volume (model artifact)
# MAGIC **Author:** ShiftMetrics Analytics — EAFIT SI7006 Trabajo 3
# MAGIC
# MAGIC ## Runtime adaptation — DD-004
# MAGIC Databricks Free Edition Serverless imposes three restrictions:
# MAGIC 1. **Py4J whitelist** blocks classic `pyspark.ml` constructors.
# MAGIC 2. **`.cache()` / `.persist()`** are disabled (`NOT_SUPPORTED_WITH_SERVERLESS`).
# MAGIC 3. **`pyspark.ml.connect.LogisticRegression`** is torch-backed; torch is not preinstalled.
# MAGIC
# MAGIC We address (1) and (2) by removing the offending calls. We address (3) by
# MAGIC installing torch via `%pip install` in the first cell.
# MAGIC
# MAGIC ## Pipeline
# MAGIC 1. Install torch (required by `pyspark.ml.connect.LogisticRegression`)
# MAGIC 2. Feature engineering via Spark SQL (no VectorAssembler)
# MAGIC 3. Stratified 80/20 split (no `.cache()`)
# MAGIC 4. 3 candidates: SparkLR + sklearn RF + sklearn GBT
# MAGIC 5. Champion selection by holdout test AUC
# MAGIC 6. Persist 4 gold metric tables + champion model in Volume

# COMMAND ----------

# Install torch (required by Spark Connect ML's LogisticRegression backend)
%pip install torch --quiet
dbutils.library.restartPython()

# COMMAND ----------

import os
import time
import uuid
import joblib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType,
)

# Spark Connect ML (torch-backed LogisticRegression — the only whitelisted classifier)
from pyspark.ml.connect.classification import LogisticRegression as SparkLR

# scikit-learn (driver-side benchmarks)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix as sk_confusion_matrix,
)

SILVER       = "workspace.shiftmetrics_silver"
GOLD         = "workspace.shiftmetrics_gold"
MODEL_VOLUME = "/Volumes/workspace/shiftmetrics_bronze/lakehouse_vol/_artifacts/models"

RUN_ID     = str(uuid.uuid4())
RUN_TS     = datetime.now(timezone.utc)
RUN_TS_STR = RUN_TS.strftime("%Y%m%d_%H%M%S")
SEED       = 42
np.random.seed(SEED)

FEATURE_COLS = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3",
    "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc",
]
TARGET_COL = "is_buggy"

print(f"-> ML pipeline started")
print(f"  run_id:    {RUN_ID}")
print(f"  run_ts:    {RUN_TS.isoformat()}")
print(f"  spark:     {spark.version}")

try:
    import torch
    print(f"  torch:     {torch.__version__}")
except ImportError:
    print(f"  torch:     [WARN] NOT AVAILABLE — SparkLR will be skipped")

# COMMAND ----------

# MAGIC %md ## [1] Load training data & inspect class balance

# COMMAND ----------

ds = (
    spark.table(f"{SILVER}.code_metrics_clean")
        .select(TARGET_COL, *FEATURE_COLS, "_project", "_version", "name")
        .dropna(subset=FEATURE_COLS + [TARGET_COL])
)

n_total = ds.count()
print(f"  Total rows:    {n_total:,}")
print(f"  Class balance:")
for row in ds.groupBy(TARGET_COL).count().orderBy(TARGET_COL).collect():
    pct = 100.0 * row["count"] / n_total
    print(f"    is_buggy={row[TARGET_COL]}  →  {row['count']:>6,}  ({pct:5.1f}%)")
display(ds.limit(5))

# COMMAND ----------

# MAGIC %md ## [2] Stratified 80/20 split (no `.cache()` per DD-004)

# COMMAND ----------

fractions = {0: 0.8, 1: 0.8}
train_sdf = ds.sampleBy(TARGET_COL, fractions=fractions, seed=SEED)
test_sdf  = ds.subtract(train_sdf)

n_train = train_sdf.count()
n_test  = test_sdf.count()
print(f"  Train: {n_train:>6,}  Test: {n_test:>6,}")

for label, df in [("Train", train_sdf), ("Test", test_sdf)]:
    print(f"  {label} class balance:")
    for r in df.groupBy(TARGET_COL).count().orderBy(TARGET_COL).collect():
        print(f"    is_buggy={r[TARGET_COL]}: {r['count']:>6,}  ({100.0*r['count']/df.count():.1f}%)")

# COMMAND ----------

# MAGIC %md ## [3] Feature engineering — Spark SQL z-score normalization

# COMMAND ----------

agg_exprs = []
for c in FEATURE_COLS:
    agg_exprs += [F.mean(c).alias(f"{c}_mean"), F.stddev(c).alias(f"{c}_std")]
scaling_params = train_sdf.agg(*agg_exprs).collect()[0].asDict()
print(f"  Scaling params from {n_train:,} training rows")
print(f"  Sample (wmc): mean={scaling_params['wmc_mean']:.3f}  std={scaling_params['wmc_std']:.3f}")

def apply_zscore(sdf):
    scaled_cols = []
    for c in FEATURE_COLS:
        mu  = float(scaling_params[f"{c}_mean"])
        std = float(scaling_params[f"{c}_std"]) or 1.0
        scaled_cols.append(((F.col(c) - F.lit(mu)) / F.lit(std)).alias(f"{c}_z"))
    return sdf.select(
        TARGET_COL, "_project", "_version", "name", *FEATURE_COLS, *scaled_cols
    ).withColumn(
        "features",
        F.array(*[F.col(f"{c}_z") for c in FEATURE_COLS]),
    )

train_scaled = apply_zscore(train_sdf)
test_scaled  = apply_zscore(test_sdf)
print(f"  [OK] z-score applied to train ({n_train:,}) and test ({n_test:,})")
display(train_scaled.select("name", "_project", TARGET_COL, "features").limit(3))

# COMMAND ----------

# MAGIC %md ## [4] Pull data to driver for sklearn models

# COMMAND ----------

train_pdf = train_scaled.select(*FEATURE_COLS, TARGET_COL).toPandas()
test_pdf  = test_scaled.select(*FEATURE_COLS, TARGET_COL, "name", "_project", "_version").toPandas()

X_train = train_pdf[FEATURE_COLS].values.astype(np.float64)
y_train = train_pdf[TARGET_COL].values.astype(np.int32)
X_test  = test_pdf[FEATURE_COLS].values.astype(np.float64)
y_test  = test_pdf[TARGET_COL].values.astype(np.int32)

sk_scaler = SkScaler().fit(X_train)
X_train_sk = sk_scaler.transform(X_train)
X_test_sk  = sk_scaler.transform(X_test)

print(f"  Driver tensors:")
print(f"    X_train: {X_train_sk.shape}  y_train: {y_train.shape}  (positives: {int(y_train.sum())})")
print(f"    X_test:  {X_test_sk.shape}   y_test:  {y_test.shape}   (positives: {int(y_test.sum())})")

# COMMAND ----------

# MAGIC %md ## [5] Helpers — evaluation + sklearn training

# COMMAND ----------

def evaluate_predictions(y_true, y_pred, y_proba):
    return {
        "auc_test":       float(roc_auc_score(y_true, y_proba)),
        "f1_test":        float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy_test":  float(accuracy_score(y_true, y_pred)),
        "precision_test": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_test":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def train_sklearn(name, estimator, param_grid):
    t0 = time.time()
    print(f"\n-> Training {name} (sklearn)...")
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train_sk, y_train)
    best = grid.best_estimator_

    y_pred_train  = best.predict(X_train_sk)
    y_proba_train = best.predict_proba(X_train_sk)[:, 1]
    auc_train     = float(roc_auc_score(y_train, y_proba_train))

    y_pred  = best.predict(X_test_sk)
    y_proba = best.predict_proba(X_test_sk)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    metrics.update({
        "auc_train":   auc_train,
        "name":        name,
        "model":       best,
        "y_pred":      y_pred,
        "y_proba":     y_proba,
        "elapsed_s":   time.time() - t0,
        "best_params": grid.best_params_,
        "framework":   "sklearn",
    })

    print(f"    Best params: {grid.best_params_}")
    print(f"    AUC train: {auc_train:.4f}  AUC test: {metrics['auc_test']:.4f}")
    print(f"    F1: {metrics['f1_test']:.4f}  Accuracy: {metrics['accuracy_test']:.4f}")
    print(f"    Precision: {metrics['precision_test']:.4f}  Recall: {metrics['recall_test']:.4f}")
    print(f"    Elapsed: {metrics['elapsed_s']:.1f}s")
    return metrics

# COMMAND ----------

# MAGIC %md ## [6] Candidate 1 — SparkMLlib LogisticRegression (torch-backed)

# COMMAND ----------

print(f"\n-> Training LogisticRegression (Spark Connect MLlib, torch-backed)...")
t0 = time.time()

try:
    import torch
    SPARK_LR_AVAILABLE = True
    print(f"  torch {torch.__version__} available")
except ImportError:
    SPARK_LR_AVAILABLE = False
    print(f"  [WARN] torch not available — SparkLR will be skipped.")

spark_lr_train = train_scaled.select(F.col(TARGET_COL).alias("label"), "features")
spark_lr_test  = test_scaled.select(F.col(TARGET_COL).alias("label"), "features",
                                    "name", "_project", "_version")

def to_proba_pos(col):
    return col.apply(lambda v: float(v[1]) if v is not None else 0.0)

result_spark_lr = None
if SPARK_LR_AVAILABLE:
    spark_grid = [(0.001, 50), (0.001, 100), (0.0005, 100)]
    best_spark = None
    for lr_val, mi_val in spark_grid:
        print(f"  Trying learningRate={lr_val}, maxIter={mi_val}...")
        try:
            lr_local = SparkLR(
                featuresCol="features",
                labelCol="label",
                learningRate=lr_val,
                maxIter=mi_val,
            )
            model = lr_local.fit(spark_lr_train)
            pred_train = model.transform(spark_lr_train)
            pred_test  = model.transform(spark_lr_test)
            pdf_train  = pred_train.select("label", "prediction", "probability").toPandas()
            pdf_test   = pred_test.select("label", "prediction", "probability",
                                          "name", "_project", "_version").toPandas()

            auc_tr  = roc_auc_score(pdf_train["label"].values,
                                     to_proba_pos(pdf_train["probability"]).values)
            y_pr    = pdf_test["prediction"].astype(int).values
            y_pb    = to_proba_pos(pdf_test["probability"]).values
            metrics = evaluate_predictions(pdf_test["label"].values, y_pr, y_pb)
            print(f"    AUC train: {auc_tr:.4f}  AUC test: {metrics['auc_test']:.4f}")

            if best_spark is None or metrics["auc_test"] > best_spark["auc_test"]:
                best_spark = {
                    "auc_train":   float(auc_tr),
                    **metrics,
                    "name":        "LogisticRegression (SparkMLlib)",
                    "model":       model,
                    "y_pred":      y_pr,
                    "y_proba":     y_pb,
                    "elapsed_s":   time.time() - t0,
                    "best_params": {"learningRate": lr_val, "maxIter": mi_val},
                    "framework":   "spark.ml.connect",
                    "test_pdf":    pdf_test,
                }
        except Exception as e:
            print(f"    [WARN] failed: {str(e)[:200]}")
            continue

    if best_spark is not None:
        result_spark_lr = best_spark
        print(f"\n  [OK] Spark LR best test AUC: {result_spark_lr['auc_test']:.4f}  "
              f"({result_spark_lr['best_params']})  elapsed: {result_spark_lr['elapsed_s']:.1f}s")
    else:
        print(f"\n  [WARN] All SparkLR attempts failed. Continuing with sklearn-only.")

if result_spark_lr is None:
    print(f"\n  [SKIP] SparkLR skipped. Pipeline continues with sklearn models.")

# COMMAND ----------

# MAGIC %md ## [7] Candidate 2 — sklearn RandomForestClassifier

# COMMAND ----------

result_rf = train_sklearn(
    "RandomForest (sklearn)",
    RandomForestClassifier(random_state=SEED, n_jobs=1),
    {
        "n_estimators": [50, 100],
        "max_depth":    [5, 10, None],
    },
)

# COMMAND ----------

# MAGIC %md ## [8] Candidate 3 — sklearn GradientBoostingClassifier

# COMMAND ----------

result_gbt = train_sklearn(
    "GradientBoosting (sklearn)",
    GradientBoostingClassifier(random_state=SEED),
    {
        "n_estimators":  [50, 100],
        "max_depth":     [3, 5],
        "learning_rate": [0.1],
    },
)

# COMMAND ----------

# MAGIC %md ## [9] Champion selection

# COMMAND ----------

all_results = [r for r in [result_spark_lr, result_rf, result_gbt] if r is not None]

if not all_results:
    raise RuntimeError("No models trained successfully. Aborting.")

champion = max(all_results, key=lambda r: r["auc_test"])

print("\n[*] MODEL COMPARISON (holdout test set)")
hdr = f"  {'Model':<32s} {'AUC':>7s} {'F1':>7s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'Framework':>20s}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in all_results:
    mark = " [CHAMPION]" if r["name"] == champion["name"] else "   "
    print(f"  {r['name']:<32s} {r['auc_test']:>7.4f} {r['f1_test']:>7.4f} "
          f"{r['accuracy_test']:>7.4f} {r['precision_test']:>7.4f} {r['recall_test']:>7.4f} "
          f"{r['framework']:>20s}{mark}")
print(f"\n  [CHAMPION] {champion['name']} (test AUC = {champion['auc_test']:.4f})")

# COMMAND ----------

# MAGIC %md ## [10] Persist `gold.ml_model_metrics`

# COMMAND ----------

metrics_rows = [
    (
        RUN_ID, RUN_TS, r["name"], r["framework"],
        float(r["auc_train"]), float(r["auc_test"]),
        float(r["f1_test"]), float(r["accuracy_test"]),
        float(r["precision_test"]), float(r["recall_test"]),
        float(r["elapsed_s"]), str(r["best_params"]),
        str(r["name"] == champion["name"]),
    )
    for r in all_results
]
metrics_schema = StructType([
    StructField("run_id",          StringType(),    False),
    StructField("run_ts",          TimestampType(), False),
    StructField("model_name",      StringType(),    False),
    StructField("framework",       StringType(),    False),
    StructField("auc_train",       DoubleType(),    False),
    StructField("auc_test",        DoubleType(),    False),
    StructField("f1_test",         DoubleType(),    False),
    StructField("accuracy_test",   DoubleType(),    False),
    StructField("precision_test", DoubleType(),    False),
    StructField("recall_test",    DoubleType(),    False),
    StructField("training_seconds", DoubleType(),  False),
    StructField("best_params",    StringType(),    False),
    StructField("is_champion",    StringType(),    False),
])
metrics_df = spark.createDataFrame(metrics_rows, schema=metrics_schema)
(
    metrics_df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(f"{GOLD}.ml_model_metrics")
)
spark.sql(f"""
    COMMENT ON TABLE {GOLD}.ml_model_metrics IS
    'Append-only ML model comparison log. One row per (run_id, model_name). is_champion=true marks the winner.'
""")
print(f"  [OK] Appended {len(metrics_rows)} model metrics rows")
display(spark.sql(f"""
    SELECT model_name, framework, ROUND(auc_test,4) AS auc_test,
           ROUND(f1_test,4) AS f1_test, ROUND(precision_test,4) AS precision_w,
           ROUND(recall_test,4) AS recall_w, is_champion
    FROM {GOLD}.ml_model_metrics
    WHERE run_id = '{RUN_ID}'
    ORDER BY auc_test DESC
"""))

# COMMAND ----------

# MAGIC %md ## [11] Persist `gold.ml_predictions`

# COMMAND ----------

if champion["framework"] == "spark.ml.connect":
    pred_pdf = champion["test_pdf"].copy()
    pred_pdf = pred_pdf.rename(columns={"name": "module_name", "_project": "project",
                                        "_version": "version"})
    pred_pdf["prob_buggy"] = champion["y_proba"]
    pred_pdf = pred_pdf[["module_name", "project", "version", "label", "prediction", "prob_buggy"]]
else:
    pred_pdf = pd.DataFrame({
        "module_name": test_pdf["name"].values,
        "project":     test_pdf["_project"].values,
        "version":     test_pdf["_version"].values,
        "label":       y_test,
        "prediction":  champion["y_pred"].astype(int),
        "prob_buggy":  champion["y_proba"],
    })

pred_pdf.insert(0, "model_name", champion["name"])
pred_pdf.insert(0, "run_ts",     RUN_TS)
pred_pdf.insert(0, "run_id",     RUN_ID)
pred_pdf["label"]      = pred_pdf["label"].astype("int32")
pred_pdf["prediction"] = pred_pdf["prediction"].astype("int32")
pred_pdf["prob_buggy"] = pred_pdf["prob_buggy"].astype("float64")

pred_schema = StructType([
    StructField("run_id",      StringType(),    False),
    StructField("run_ts",      TimestampType(), False),
    StructField("model_name",  StringType(),    False),
    StructField("module_name", StringType(),    False),
    StructField("project",     StringType(),    False),
    StructField("version",     StringType(),    True),
    StructField("label",       IntegerType(),   False),
    StructField("prediction",  IntegerType(),   False),
    StructField("prob_buggy",  DoubleType(),    False),
])
pred_sdf = spark.createDataFrame(pred_pdf, schema=pred_schema)
(
    pred_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{GOLD}.ml_predictions")
)
spark.sql(f"""
    ALTER TABLE {GOLD}.ml_predictions SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = true,
        'delta.autoOptimize.autoCompact'   = true
    )
""")
spark.sql(f"""
    COMMENT ON TABLE {GOLD}.ml_predictions IS
    'Champion model test-set predictions with probabilities for is_buggy.'
""")
print(f"  [OK] Wrote {len(pred_pdf):,} predictions")
display(pred_sdf.limit(10))

# COMMAND ----------

# MAGIC %md ## [12] Persist `gold.ml_confusion_matrix`

# COMMAND ----------

tn, fp, fn, tp = sk_confusion_matrix(
    pred_pdf["label"].values, pred_pdf["prediction"].values, labels=[0, 1]
).ravel()
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_pos    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_pos = (2*precision_pos*recall_pos / (precision_pos+recall_pos)) if (precision_pos+recall_pos) > 0 else 0.0
print(f"  Positive class (is_buggy=1):")
print(f"    Precision: {precision_pos:.4f}")
print(f"    Recall:    {recall_pos:.4f}")
print(f"    F1:        {f1_pos:.4f}")

cm_rows = [
    (RUN_ID, RUN_TS, champion["name"], "TP", int(tp)),
    (RUN_ID, RUN_TS, champion["name"], "FP", int(fp)),
    (RUN_ID, RUN_TS, champion["name"], "TN", int(tn)),
    (RUN_ID, RUN_TS, champion["name"], "FN", int(fn)),
]
cm_schema = StructType([
    StructField("run_id",     StringType(),    False),
    StructField("run_ts",     TimestampType(), False),
    StructField("model_name", StringType(),    False),
    StructField("cell",       StringType(),    False),
    StructField("count",      IntegerType(),   False),
])
cm_df = spark.createDataFrame(cm_rows, schema=cm_schema)
(
    cm_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{GOLD}.ml_confusion_matrix")
)
spark.sql(f"""
    COMMENT ON TABLE {GOLD}.ml_confusion_matrix IS
    'Confusion matrix counts (TP/FP/TN/FN) of the champion model on the holdout test set.'
""")
print(f"  [OK] Wrote confusion matrix")
display(cm_df.orderBy("cell"))

# COMMAND ----------

# MAGIC %md ## [13] Persist `gold.ml_feature_importance`

# COMMAND ----------

clf = champion["model"]
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_.tolist()
    importance_type = "feature_importance"
elif hasattr(clf, "coef_"):
    importances = [abs(float(c)) for c in clf.coef_.ravel()]
    importance_type = "abs_coefficient"
else:
    try:
        importances = [abs(float(c)) for c in clf.coefficients.toArray()]
        importance_type = "abs_coefficient"
    except Exception:
        importances = [0.0] * len(FEATURE_COLS)
        importance_type = "n/a"

fi_pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
fi_rows = [
    (RUN_ID, RUN_TS, champion["name"], importance_type, name, float(imp), rank+1)
    for rank, (name, imp) in enumerate(fi_pairs)
]
fi_schema = StructType([
    StructField("run_id",          StringType(),    False),
    StructField("run_ts",          TimestampType(), False),
    StructField("model_name",      StringType(),    False),
    StructField("importance_type", StringType(),    False),
    StructField("feature",         StringType(),    False),
    StructField("importance",      DoubleType(),    False),
    StructField("rank",            IntegerType(),   False),
])
fi_df = spark.createDataFrame(fi_rows, schema=fi_schema)
(
    fi_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{GOLD}.ml_feature_importance")
)
spark.sql(f"""
    COMMENT ON TABLE {GOLD}.ml_feature_importance IS
    'Feature ranking of champion model. Tree-based: feature_importances_; LR: |coefficient|.'
""")
print(f"  [OK] Wrote feature importance ({importance_type})")
display(fi_df.orderBy("rank"))

# COMMAND ----------

# MAGIC %md ## [14] Persist champion model artifact → Volume

# COMMAND ----------

artifact_dir = f"{MODEL_VOLUME}/champion_{RUN_TS_STR}"
os.makedirs(artifact_dir, exist_ok=True)

if champion["framework"] == "spark.ml.connect":
    model_path = f"{artifact_dir}/spark_lr_model"
    champion["model"].save(model_path)
    print(f"  [OK] Saved SparkLR model to: {model_path}")
else:
    model_path = f"{artifact_dir}/sklearn_model.joblib"
    joblib.dump(champion["model"], model_path)
    print(f"  [OK] Saved sklearn model to: {model_path}")

scaler_path = f"{artifact_dir}/scaler.joblib"
joblib.dump(sk_scaler, scaler_path)
print(f"  [OK] Saved feature scaler to: {scaler_path}")

meta = {
    "run_id":      RUN_ID,
    "run_ts":      RUN_TS.isoformat(),
    "champion":    champion["name"],
    "framework":   champion["framework"],
    "metrics": {
        "auc_test":   champion["auc_test"],
        "f1_test":    champion["f1_test"],
        "accuracy":   champion["accuracy_test"],
    },
    "feature_cols":  FEATURE_COLS,
    "best_params":   {k: str(v) for k, v in champion["best_params"].items()},
}
with open(f"{artifact_dir}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  [OK] Saved metadata.json")
print(f"\n  [ARTIFACT] Dir: {artifact_dir}")

# COMMAND ----------

# MAGIC %md ## [15] Final summary

# COMMAND ----------

display(spark.sql(f"""
    SELECT table_name, comment
    FROM workspace.information_schema.tables
    WHERE table_schema = 'shiftmetrics_gold'
      AND table_name LIKE 'ml_%'
    ORDER BY table_name
"""))

print("\n[*] ML PIPELINE FINISHED — " + datetime.now(timezone.utc).isoformat())
print(f"  run_id:       {RUN_ID}")
print(f"  champion:     {champion['name']} ({champion['framework']})")
print(f"  test AUC:     {champion['auc_test']:.4f}")
print(f"  test F1:      {champion['f1_test']:.4f}")
print(f"  artifact:     {artifact_dir}")
print(f"")
print(f"  Models trained:  {len(all_results)} of 3 attempted")
for r in all_results:
    print(f"    - {r['name']:<32s}  AUC={r['auc_test']:.4f}  ({r['framework']})")
print(f"")
print(f"  Top 3 features for the champion:")
for r in fi_rows[:3]:
    print(f"    {r[4]:>8s}  importance={r[5]:.4f}  ({r[3]})")
