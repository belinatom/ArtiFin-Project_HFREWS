# ----------- TRAINING: TANZANIAN PRIMARY HEALTHCARE FACILITIES INACCESSIBILITY RISK —-----------------------
# --ANOMALY & RISK DETECTION PIPELINE FROM RAINFALL AND ANTECEDENT ENVIRONMENT CONDITIONS──────────
''' Runs 4 experiments to detect health facilities at risk of being cut off 
due to heavy rainfall and antecedent environmental conditions.''''

#-------------- DEFINITION OF FEATURES BEING USED FOR MODEL TRAINING ---------------
''''Dynamic Features - this includes all the features that change daily or from time to time. It include features derived from rainfall and soil moisture
Terrain or Static Features - Includes all features that stay mostly constant for a long period of time. It includes the elevation where a facility is placed,
the slope and the drainage capacity of the soil in the environment the facility is located.''''

# --------------------- MODEL SELECTION JUSTIFICATION ------------------------------------'
'''' Isolation Forest - 
One Class SVM - '''

# ------------------ PRE-DEFINING THE EXPERIMENTS THAT WILL BE RUN IN THIS TRAINING PIPELINE ------
''' Two models are used in this training pipeline and 2 experiments are conducted for each model varying the type of parameters 
included in the model. The variation includes adding or removing static features to see how antecedent static features affect or not affect the risk score 
and the resulting labelling / anomaly detection ''''

#   Experiment 1 — Isolation Forest   | Dynamic features only
#   Experiment 2 — Isolation Forest   | Dynamic + Terrain features
#   Experiment 3 — One-Class SVM      | Dynamic features only
#   Experiment 4 — One-Class SVM      | Dynamic + Terrain features

# ----------- ANOMALY DETECTION RISK SCORES DEFINITION 
''''Anomaly scores are normalised to [0, 1] and binned into three risk classes
# using percentile thresholds (more statistically robust than fixed cut-offs):
#   Green  — 0 – 70th percentile
#   Orange — 70th – 90th percentile
#   Red    — > 90th percentile ''''

# ------------- ENVIRONMENT PREPARATION ---------------------
'''' All the packages have been installed via command line and therefore only importing them
here instead of installing them again ''''

import argparse
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────────────────────

DRAINAGE_RISK_SCORES = {
    "SE": 1,
    "MW": 2,
    "I":  3,
    "P":  4,
    "VP": 5,
}
DRAINAGE_RISK_DEFAULT = 2

# Clean 13-feature set — best balance of signal and redundancy
DYNAMIC_FEATURES = [
    "rain_1d",
    "rain_3d_sum",
    "rain_7d_sum",
    "rain_percentile",
    "rain_anomaly",
    "rain_days_above_20mm",
    "sm_1d",
    "sm_7d_mean",
    "sm_percentile",
    "sm_change_3d",
]

TERRAIN_FEATURES = [
    "elevation_m",   # elevation
    "slope_deg",     # slope
    "drainage_risk", # drainage_pattern (encoded from DRAINAGE column)
]

STATIC_FEATURES = TERRAIN_FEATURES  # alias for data loading

# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "id":       1,
        "name":     "IsolationForest_Dynamic",
        "model":    "IsolationForest",
        "features": DYNAMIC_FEATURES,
        "desc":     "Test if climate variables alone detect risk",
    },
    {
        "id":       2,
        "name":     "IsolationForest_Dynamic_Terrain",
        "model":    "IsolationForest",
        "features": DYNAMIC_FEATURES + TERRAIN_FEATURES,
        "desc":     "Does terrain improve anomaly detection?",
    },
    {
        "id":       3,
        "name":     "OneClassSVM_Dynamic",
        "model":    "OneClassSVM",
        "features": DYNAMIC_FEATURES,
        "desc":     "Compare anomaly detection methods — dynamic only",
    },
    {
        "id":       4,
        "name":     "OneClassSVM_Dynamic_Terrain",
        "model":    "OneClassSVM",
        "features": DYNAMIC_FEATURES + TERRAIN_FEATURES,
        "desc":     "Best performing configuration candidate",
    },
]

# Percentile thresholds for risk classification
RISK_THRESHOLDS = {
    "p_orange": 70,
    "p_red":    90,
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load each data source from disk
# ─────────────────────────────────────────────────────────────────────────────

def load_static_features(path):
    """Loads hfr_data.csv and prepares static + terrain features per facility."""
    df = pd.read_csv(path)

    df["facility_id"] = (
        df["lat"].round(5).astype(str)
        + "_"
        + df["lon"].round(5).astype(str)
    )

    # Encode soil drainage property → ordinal drainage risk score
    df["drainage_risk"] = (
        df["DRAINAGE"]
        .map(DRAINAGE_RISK_SCORES)
        .fillna(DRAINAGE_RISK_DEFAULT)
        .astype(int)
    )

    # Region-median imputation for continuous terrain features
    for col in ["elevation_m", "slope_deg"]:
        df[col] = df.groupby("region")[col].transform(
            lambda x: x.fillna(x.median())
        )

    static_df = df[
        ["facility_id", "name", "region", "facility_class", "lat", "lon"]
        + STATIC_FEATURES
    ].copy()

    print(f"Loaded {len(static_df):,} facilities from {path}")
    print(f"  DRAINAGE risk distribution : {df['drainage_risk'].value_counts().sort_index().to_dict()}")
    print(f"  Missing after imputation   — elevation: {df['elevation_m'].isna().sum()}, slope: {df['slope_deg'].isna().sum()}")

    return static_df


def load_rainfall_features(path):
    """Loads pre-computed rainfall features CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded rainfall features      : {df.shape} from {path}")
    return df


def load_soil_moisture_features(path):
    """Loads pre-computed soil moisture features CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded soil moisture features : {df.shape} from {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Merge all three sources
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_sources(static_df, rain_df, sm_df):
    """Inner-joins rainfall + soil moisture on facility_id/date, then left-joins static."""
    # Coerce date columns to same dtype (handles mixed date format strings)
    rain_df = rain_df.copy()
    sm_df   = sm_df.copy()
    rain_df["date"] = pd.to_datetime(rain_df["date"], format="mixed")
    sm_df["date"]   = pd.to_datetime(sm_df["date"],   format="mixed")

    dynamic = rain_df.merge(sm_df, on=["facility_id", "date"], how="inner")
    merged  = dynamic.merge(
        static_df[["facility_id"] + STATIC_FEATURES + ["name", "region"]],
        on="facility_id",
        how="left",
    )

    unmatched = merged[STATIC_FEATURES[0]].isna().sum()
    print(f"\nMerge summary:")
    print(f"  Rainfall rows      : {len(rain_df):,}")
    print(f"  Soil moisture rows : {len(sm_df):,}")
    print(f"  After joining      : {len(merged):,}  (unmatched static: {unmatched})")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Reference labels (evaluation only — NOT used in model training)
# ─────────────────────────────────────────────────────────────────────────────

def assign_reference_labels(df):
    """
    Rule-based labels to evaluate how well unsupervised scores separate risk tiers.
    Models are trained without these labels (fully unsupervised).

      CUTOFF_HIGH     — saturated soils + heavy recent rain
      CUTOFF_MODERATE — elevated antecedent conditions + meaningful rain
      CUTOFF_WATCH    — positive rain anomaly + soil above median
      NO_ALERT        — no concerning conditions
    """
    def label_row(row):
        if (row["rain_1d"]        >= 15
                and row["rain_7d_sum"]   >= 60
                and row["sm_p90_flag"]   == 1
                and row["rain_p90_flag"] == 1):
            return "CUTOFF_HIGH"
        if (row["rain_1d"]        >= 8
                and row["rain_7d_sum"]   >= 35
                and row["sm_percentile"] >  60):
            return "CUTOFF_MODERATE"
        if (row["rain_1d"]        >= 3
                and row["rain_anomaly"]  >   5
                and row["sm_percentile"] >  50):
            return "CUTOFF_WATCH"
        return "NO_ALERT"

    df = df.copy()
    df["reference_label"] = df.apply(label_row, axis=1)

    print(f"\nReference label distribution ({len(df):,} rows):")
    for label, count in df["reference_label"].value_counts().items():
        print(f"  {label:<22} {count:>6}  ({count / len(df) * 100:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Anomaly scoring and risk classification
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str) -> Pipeline:
    """Returns a StandardScaler + anomaly detector pipeline."""
    if model_type == "IsolationForest":
        detector = IsolationForest(
            n_estimators=200,
            contamination=0.10,   # ~10% of rows expected to be anomalous
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "OneClassSVM":
        detector = OneClassSVM(
            kernel="rbf",
            nu=0.10,              # upper bound on fraction of anomalies
            gamma="scale",
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ("scaler",   StandardScaler()),
        ("detector", detector),
    ])


def compute_anomaly_scores(pipeline: Pipeline, X: np.ndarray, model_type: str) -> np.ndarray:
    """
    Returns raw anomaly scores where HIGHER = MORE anomalous.

    IsolationForest  → negate score_samples()      (higher score = more normal)
    OneClassSVM      → negate decision_function()  (positive = inside boundary = normal)
    """
    detector = pipeline.named_steps["detector"]
    X_scaled = pipeline.named_steps["scaler"].transform(X)

    if model_type == "IsolationForest":
        return -detector.score_samples(X_scaled)
    else:
        return -detector.decision_function(X_scaled)


def normalise_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Min-max normalise anomaly scores to [0, 1]."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()


def classify_risk(norm_scores: np.ndarray):
    """
    Percentile-based risk classification.
      0 – 70th pct  → Green
      70 – 90th pct → Orange
      > 90th pct    → Red
    """
    p_orange = np.percentile(norm_scores, RISK_THRESHOLDS["p_orange"])
    p_red    = np.percentile(norm_scores, RISK_THRESHOLDS["p_red"])

    classes = np.where(
        norm_scores > p_red,
        "Red",
        np.where(norm_scores > p_orange, "Orange", "Green")
    )
    return classes, p_orange, p_red


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Evaluate alignment with reference labels
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_alignment(risk_classes: np.ndarray, ref_labels: np.ndarray) -> dict:
    """
    Measures how well unsupervised risk classes align with rule-based labels.
    Key metrics:
      high_red_rate        — fraction of CUTOFF_HIGH rows assigned Red
      no_alert_green_rate  — fraction of NO_ALERT rows assigned Green
      alignment_score      — harmonic mean of the two (penalises imbalance)
    """
    results = {}
    df_eval = pd.DataFrame({"risk": risk_classes, "label": ref_labels})

    for label in ["CUTOFF_HIGH", "CUTOFF_MODERATE", "CUTOFF_WATCH", "NO_ALERT"]:
        mask    = df_eval["label"] == label
        n_total = mask.sum()
        if n_total == 0:
            continue
        dist = df_eval.loc[mask, "risk"].value_counts(normalize=True).to_dict()
        results[label] = {
            "n":      int(n_total),
            "Red":    round(dist.get("Red",    0.0), 4),
            "Orange": round(dist.get("Orange", 0.0), 4),
            "Green":  round(dist.get("Green",  0.0), 4),
        }

    high_red = results.get("CUTOFF_HIGH", {}).get("Red",   0.0)
    no_green = results.get("NO_ALERT",    {}).get("Green", 0.0)

    results["high_red_rate"]        = high_red
    results["no_alert_green_rate"]  = no_green
    results["alignment_score"] = (
        round(2 * high_red * no_green / (high_red + no_green), 4)
        if (high_red + no_green) > 0 else 0.0
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Run one experiment and log to MLflow
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(df: pd.DataFrame, exp: dict) -> dict:
    """Fits one anomaly model, scores all rows, classifies risk, logs to MLflow."""
    exp_id     = exp["id"]
    model_type = exp["model"]
    features   = exp["features"]
    run_name   = exp["name"]

    print(f"\n{'='*62}")
    print(f"  Experiment {exp_id} — {run_name}")
    print(f"  {exp['desc']}")
    print(f"  Model    : {model_type}")
    print(f"  Features : {len(features)}  →  {features}")
    print(f"{'='*62}")

    X          = df[features].fillna(0).values
    ref_labels = df["reference_label"].values

    pipeline = build_model(model_type)
    pipeline.fit(X)

    raw_scores               = compute_anomaly_scores(pipeline, X, model_type)
    norm_scores              = normalise_scores(raw_scores)
    risk_classes, p_orange, p_red = classify_risk(norm_scores)

    alignment   = evaluate_alignment(risk_classes, ref_labels)
    risk_counts = pd.Series(risk_classes).value_counts()
    n_total     = len(risk_classes)

    print(f"\n  Risk class distribution:")
    for rc in ["Red", "Orange", "Green"]:
        n = risk_counts.get(rc, 0)
        print(f"    {rc:<8} {n:>6}  ({n / n_total * 100:.1f}%)")

    print(f"\n  Alignment with reference labels:")
    for label, stats in alignment.items():
        if isinstance(stats, dict):
            print(f"    {label:<22}  n={stats['n']:>5}  "
                  f"Red={stats['Red']:.0%}  Orange={stats['Orange']:.0%}  Green={stats['Green']:.0%}")
    print(f"\n  CUTOFF_HIGH → Red rate     : {alignment['high_red_rate']:.0%}")
    print(f"  NO_ALERT    → Green rate   : {alignment['no_alert_green_rate']:.0%}")
    print(f"  Alignment score (harmonic) : {alignment['alignment_score']:.4f}")

    # ── MLflow logging ──────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "experiment_id":   exp_id,
            "model_type":      model_type,
            "feature_set":     "dynamic+terrain" if len(features) > len(DYNAMIC_FEATURES) else "dynamic_only",
            "n_features":      len(features),
            "features":        ", ".join(features),
            "n_rows":          len(df),
            "contamination":   0.10,
            "p_orange_thresh": RISK_THRESHOLDS["p_orange"],
            "p_red_thresh":    RISK_THRESHOLDS["p_red"],
        })

        mlflow.log_metrics({
            "anomaly_score_mean": round(float(norm_scores.mean()),              6),
            "anomaly_score_std":  round(float(norm_scores.std()),               6),
            "anomaly_score_p90":  round(float(np.percentile(norm_scores, 90)), 6),
            "anomaly_score_p95":  round(float(np.percentile(norm_scores, 95)), 6),
            "p_orange_cutoff":    round(float(p_orange), 6),
            "p_red_cutoff":       round(float(p_red),    6),
        })

        for rc in ["Red", "Orange", "Green"]:
            n = risk_counts.get(rc, 0)
            mlflow.log_metric(f"n_{rc.lower()}",   int(n))
            mlflow.log_metric(f"pct_{rc.lower()}", round(n / n_total, 6))

        mlflow.log_metrics({
            "high_red_rate":       alignment["high_red_rate"],
            "no_alert_green_rate": alignment["no_alert_green_rate"],
            "alignment_score":     alignment["alignment_score"],
        })
        for label, stats in alignment.items():
            if isinstance(stats, dict):
                safe = label.lower()
                mlflow.log_metric(f"align_{safe}_red",    stats["Red"])
                mlflow.log_metric(f"align_{safe}_orange", stats["Orange"])
                mlflow.log_metric(f"align_{safe}_green",  stats["Green"])

        input_example = pd.DataFrame([X[0]], columns=features)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=f"model_exp{exp_id}",
            input_example=input_example,
            registered_model_name=f"CutoffRisk_{run_name}",
        )

        print(f"\n  MLflow run ID : {run.info.run_id}")

    result_df = df[["facility_id", "date", "reference_label"]].copy()
    result_df["anomaly_score_raw"]  = raw_scores
    result_df["anomaly_score_norm"] = norm_scores
    result_df["risk_class"]         = risk_classes
    result_df["experiment"]         = run_name

    return {
        "experiment":      run_name,
        "pipeline":        pipeline,
        "result_df":       result_df,
        "alignment_score": alignment["alignment_score"],
        "high_red_rate":   alignment["high_red_rate"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="4-experiment anomaly detection pipeline for HFR cutoff risk"
    )
    parser.add_argument("--static",    required=True, help="Path to hfr_data.csv")
    parser.add_argument("--rain",      required=True, help="Path to rainfall_features.csv")
    parser.add_argument("--sm",        required=True, help="Path to sm_features.csv")
    parser.add_argument("--mlflow_db", default="sqlite:///mlflow_cutoff_alerts.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_db)
    mlflow.set_experiment("HFR_CutoffRisk_AnomalyDetection")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n── LOADING DATA ───────────────────────────────────────────")
    static_df = load_static_features(args.static)
    rain_df   = load_rainfall_features(args.rain)
    sm_df     = load_soil_moisture_features(args.sm)

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("\n── MERGING ────────────────────────────────────────────────")
    merged = merge_all_sources(static_df, rain_df, sm_df)

    # ── Reference labels (evaluation only) ────────────────────────────────────
    print("\n── ASSIGNING REFERENCE LABELS ─────────────────────────────")
    merged = assign_reference_labels(merged)

    # ── Run all 4 experiments ─────────────────────────────────────────────────
    print("\n── RUNNING EXPERIMENTS ────────────────────────────────────")
    results = []
    for exp in EXPERIMENTS:
        res = run_experiment(merged, exp)
        results.append(res)

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  EXPERIMENT COMPARISON SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Experiment':<40} {'Alignment':>10}  {'High→Red':>10}")
    print(f"  {'─'*40} {'─'*10}  {'─'*10}")
    for res in sorted(results, key=lambda r: r["alignment_score"], reverse=True):
        print(f"  {res['experiment']:<40} {res['alignment_score']:>10.4f}  {res['high_red_rate']:>10.0%}")

    best = max(results, key=lambda r: r["alignment_score"])
    print(f"\n  ✓ Best experiment : {best['experiment']}")
    print(f"    Alignment score : {best['alignment_score']:.4f}")
    print(f"    CUTOFF_HIGH → Red rate : {best['high_red_rate']:.0%}")
    print(f"\n  View in MLflow:")
    print(f"  mlflow ui --backend-store-uri {args.mlflow_db}")


if __name__ == "__main__":
    main()
