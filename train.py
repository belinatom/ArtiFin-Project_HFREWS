# ----------- TRAINING: TANZANIAN PRIMARY HEALTHCARE FACILITIES INACCESSIBILITY RISK -----------
# ------ ANOMALY & RISK DETECTION PIPELINE FROM RAINFALL AND ANTECEDENT ENVIRONMENT CONDITIONS ------

import argparse
import os
import warnings
from pathlib import Path
from datetime import date, timedelta

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")


# ── STEP 1: FEATURE SETS ──────────────────────────────────────────────────────────────────────
DYNAMIC = [
    "rain_1d", "rain_3d_sum", "rain_7d_sum", "rain_percentile", "rain_anomaly",
    "river_discharge_m3s", "discharge_percentile",
    "swi_1d", "swi_7d_mean", "swi_3d_trend", "swi_percentile",
    "runoff_1d", "runoff_3d_sum", "runoff_7d_sum", "runoff_percentile",
]
STATIC = ["water_accumulation_index", "dist_to_river_km"]
ALL = DYNAMIC + STATIC


# ── STEP 2: MLFLOW SETUP ──────────────────────────────────────────────────────────────────────
def setup_mlflow(project_dir: Path, experiment_name: str = "ArtifinProj_HFREWS_fresh"):
    """
    Point MLflow at a local SQLite file in the project directory.
    Using a plain file path (no as_posix(), no absolute URI construction)
    ensures the UI command works on Windows without path issues.
    """
    # Use as_posix() so Windows backslashes don't break the SQLite URI
    # e.g. sqlite:///C:/Users/belin/.../mlflow.db  (forward slashes required)
    db_path      = (project_dir / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return {
        "tracking_uri": tracking_uri,
        "db_path": db_path,
        "experiment_name": experiment_name,
    }


# ── STEP 3: FEATURE NORMALISATION ────────────────────────────────────────────────────────────
def mm(s):
    """Min-max normalise a Series; returns 0.5 for zero-variance/all-NaN columns."""
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or (mx - mn) < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


# ── STEP 4: INDEX COMPUTATION (dynamic = facility-specific, static = global) ─────────────────
def compute_indices(df):
    d = df.copy()

    rain_1d_n = d.groupby("facility_id")["rain_1d"].transform(mm).fillna(0.5)
    rain_3d_n = d.groupby("facility_id")["rain_3d_sum"].transform(mm).fillna(0.5)
    rain_7d_n = d.groupby("facility_id")["rain_7d_sum"].transform(mm).fillna(0.5)

    swi_1d_n = d.groupby("facility_id")["swi_1d"].transform(mm).fillna(0.5)
    swi_7d_n = d.groupby("facility_id")["swi_7d_mean"].transform(mm).fillna(0.5)
    swi_pct_n = d.groupby("facility_id")["swi_percentile"].transform(mm).fillna(0.5)

    dis_pct_n = d.groupby("facility_id")["discharge_percentile"].transform(mm).fillna(0.5)
    runoff_7d_n = d.groupby("facility_id")["runoff_7d_sum"].transform(mm).fillna(0.5)

    d["rainfall_index"] = (
        (1 / 3) * rain_1d_n + (1 / 3) * rain_3d_n + (1 / 3) * rain_7d_n
    ).clip(0, 1)

    d["sm_index"] = (
        (1 / 3) * swi_1d_n + (1 / 3) * swi_7d_n + (1 / 3) * swi_pct_n
    ).clip(0, 1)

    d["river_index"] = (
        (1 / 3) * dis_pct_n + (1 / 3) * swi_1d_n + (1 / 3) * runoff_7d_n
    ).clip(0, 1)

    if "water_accumulation_index" in d.columns and "dist_to_river_km" in d.columns:
        wai = d["water_accumulation_index"].fillna(0.5).clip(0, 1)
        prox = (1 - mm(d["dist_to_river_km"])).fillna(0.5).clip(0, 1)
        d["terrain_index"] = (0.5 * wai + 0.5 * prox).clip(0, 1)
    elif "water_accumulation_index" in d.columns:
        d["terrain_index"] = d["water_accumulation_index"].fillna(0.5).clip(0, 1)
    else:
        d["terrain_index"] = 0.5

    d["composite_score"] = (
        0.25 * d["rainfall_index"] + 0.25 * d["sm_index"] +
        0.25 * d["river_index"] + 0.25 * d["terrain_index"]
    ).clip(0, 1)

    return d


# ── STEP 5: PROXY LABELS (rule-based, per-facility percentiles) ───────────────────────────────
def label_rows(df):
    d = df.copy()

    q90_rain = d.groupby("facility_id")["rain_1d"].transform(lambda x: x.quantile(0.90))
    q90_dis = d.groupby("facility_id")["river_discharge_m3s"].transform(lambda x: x.quantile(0.90))
    q80_swi = d.groupby("facility_id")["swi_1d"].transform(lambda x: x.quantile(0.80))

    if "water_accumulation_index" in d.columns:
        wai = d["water_accumulation_index"].fillna(0.5).clip(0, 1)
        wai_high_thresh = float(wai.quantile(0.70))
    else:
        wai = pd.Series(0.5, index=d.index)
        wai_high_thresh = 0.7

    rain_spike = d["rain_1d"] >= q90_rain
    if "rain_percentile" in d.columns:
        rain_spike = rain_spike | (d["rain_percentile"] >= 0.90)

    river_spike = d["river_discharge_m3s"] >= q90_dis
    saturated = d["swi_1d"] >= q80_swi

    high = (rain_spike | river_spike) & saturated & (wai >= wai_high_thresh)
    moderate = (rain_spike | river_spike) & saturated & ~high

    labels = pd.Series("NO_ALERT", index=d.index, dtype="object")
    labels.loc[moderate] = "MODERATE"
    labels.loc[high] = "HIGH"
    d["label"] = labels

    summary = d.groupby("facility_id")["label"].value_counts().unstack(fill_value=0)
    print("\n--- Alert Summary Per Facility ---")
    print(summary)
    print("----------------------------------\n")

    return d


# ── STEP 6: METRICS ───────────────────────────────────────────────────────────────────────────
def compute_metrics(labels, flags, final_scores):
    y_true_high = (labels == "HIGH").astype(int).values
    y_true_any = (labels != "NO_ALERT").astype(int).values

    metrics = {
        "anomaly_rate": float(flags.mean()),
        "mean_score": float(final_scores.mean()),
        "max_score": float(final_scores.max()),
    }
    for suffix, y_true in [("high", y_true_high), ("any", y_true_any)]:
        if y_true.sum() > 0:
            metrics[f"f1_{suffix}"] = float(f1_score(y_true, flags, zero_division=0))
            metrics[f"precision_{suffix}"] = float(precision_score(y_true, flags, zero_division=0))
            metrics[f"recall_{suffix}"] = float(recall_score(y_true, flags, zero_division=0))
        else:
            metrics[f"f1_{suffix}"] = 0.0
            metrics[f"precision_{suffix}"] = 0.0
            metrics[f"recall_{suffix}"] = 0.0

    return metrics


# ── STEP 7: OBJECTIVE EVALUATION ─────────────────────────────────────────────────────────────
def evaluate_model_performance(d):
    index_cols = ["rainfall_index", "sm_index", "river_index"]
    if "terrain_index" in d.columns:
        index_cols.append("terrain_index")

    print("\n" + "=" * 55)
    print("         OBJECTIVE UNSUPERVISED METRICS")
    print("=" * 55)

    eval_df = d[index_cols + ["label", "composite_score"]].dropna().copy()

    if len(eval_df) == 0:
        print("[METRIC] No complete rows available for evaluation.")
        print("=" * 55 + "\n")
        return pd.DataFrame()

    if eval_df["label"].nunique() > 1:
        sample_size = min(len(eval_df), 5000)
        df_samp = eval_df.sample(sample_size, random_state=42)
        score = silhouette_score(df_samp[index_cols], df_samp["label"])
        print(f"[METRIC] Silhouette Score : {score:.4f}")
        print("         (>0.25 reasonable structure | <0.10 overlapping/noisy)")
    else:
        print("[METRIC] Silhouette Score : N/A (need >= 2 label types)")

    corr = eval_df[index_cols].corr()
    print("\n[METRIC] Internal Feature Correlation:")
    print(corr.round(3))

    mean_score = eval_df["composite_score"].mean()
    cv = eval_df["composite_score"].std() / mean_score if mean_score != 0 else 0.0
    at_min = (eval_df["composite_score"] <= 0.01).sum() / len(eval_df) * 100
    at_max = (eval_df["composite_score"] >= 0.99).sum() / len(eval_df) * 100
    print(f"\n[METRIC] Coefficient of Variation : {cv:.4f}")
    print(f"[METRIC] Boundary Saturation       : floor(0): {at_min:.1f}% | ceiling(1): {at_max:.1f}%")

    pca = PCA(n_components=1)
    pca.fit(eval_df[index_cols])
    print(f"[METRIC] PC1 Explained Variance    : {pca.explained_variance_ratio_[0]:.4f}")
    print("=" * 55 + "\n")

    return corr


# ── STEP 8: GLOBAL EXPERIMENT RUNNER ─────────────────────────────────────────────────────────
def run_experiment(df, feats, contamination, out_dir, exp_label):
    """
    Train one IsolationForest on the full dataset, score every row, log
    everything to MLflow, and save the pipeline locally.
    """
    use_terrain = any(c in feats for c in STATIC)
    feature_tag = "dynamic+terrain" if use_terrain else "dynamic"
    X = df[feats].fillna(0)

    detector = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

    print(f"  Fitting IsolationForest on {len(X):,} rows x {len(feats)} features...")
    pipe = Pipeline([("scaler", StandardScaler()), ("detector", detector)])
    pipe.fit(X)

    Xs = pipe.named_steps["scaler"].transform(X)
    raw = -pipe.named_steps["detector"].score_samples(Xs)
    norm = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).flatten()

    if use_terrain:
        phys = df["composite_score"].values
    else:
        phys = ((df["rainfall_index"] + df["sm_index"] + df["river_index"]) / 3).values

    final_scores = (0.8 * phys + 0.2 * norm).clip(0, 1)
    thresh = np.percentile(norm, 100 * (1 - contamination))
    flags = (norm >= thresh).astype(int)

    scored_df = df.copy()
    scored_df["anomaly_score"] = norm
    scored_df["final_score"] = final_scores
    scored_df["flagged"] = flags

    m = compute_metrics(df["label"], flags, final_scores)

    rows = []
    print(f"\n  Per-facility flag breakdown ({exp_label}):")
    print(f"  {'Facility':<22} | {'Total rows':>10} | {'Flagged':>8} | {'Flag %':>7}")
    print("  " + "-" * 55)
    for f_id, grp in scored_df.groupby("facility_id"):
        n_rows = len(grp)
        n_flagged = int(grp["flagged"].sum())
        pct = n_flagged / n_rows * 100
        print(f"  {str(f_id):<22} | {n_rows:>10,} | {n_flagged:>8} | {pct:>6.1f}%")
        rows.append({
            "facility_id": f_id,
            "total_rows": n_rows,
            "n_flagged": n_flagged,
            "flag_pct": round(pct, 2),
        })

    facility_breakdown = pd.DataFrame(rows)

    print(
        f"\n  Global — F1-High: {m['f1_high']:.3f} | "
        f"F1-Any: {m['f1_any']:.3f} | "
        f"Anomaly rate: {m['anomaly_rate'] * 100:.1f}%"
    )

    artifact_staging = os.path.join(out_dir, f"_staging_{exp_label}")
    os.makedirs(artifact_staging, exist_ok=True)

    pipeline_path = os.path.join(artifact_staging, f"{exp_label}.pkl")
    breakdown_path = os.path.join(artifact_staging, f"{exp_label}_facility_breakdown.csv")
    scored_path = os.path.join(artifact_staging, f"{exp_label}_scored.parquet")

    joblib.dump(pipe, pipeline_path)
    facility_breakdown.to_csv(breakdown_path, index=False)
    scored_df.to_parquet(scored_path, index=False, compression="snappy")

    with mlflow.start_run(run_name=exp_label):
        mlflow.log_params({
            "model": "IsolationForest",
            "features": feature_tag,
            "n_features": len(feats),
            "contamination": contamination,
            "total_rows": len(df),
            "n_facilities": int(df["facility_id"].nunique()),
        })
        mlflow.log_metrics(m)
        mlflow.log_artifact(breakdown_path, artifact_path="facility_breakdown")
        mlflow.log_artifact(scored_path, artifact_path="scored_predictions")
        mlflow.log_artifact(pipeline_path, artifact_path="models")

    local_model_path = os.path.join(out_dir, f"{exp_label}.pkl")
    joblib.dump(pipe, local_model_path)
    print(f"  Pipeline saved -> {local_model_path}")

    return {
        "exp_label": exp_label,
        "model_name": "IsolationForest",
        "features": feature_tag,
        "metrics": m,
        "pipe": pipe,
        "scored_df": scored_df,
    }


# ── MAIN ────────────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Tanzanian PHC inaccessibility risk — anomaly & risk detection pipeline"
    )
    p.add_argument("--static", default="hfr_data.csv")
    p.add_argument("--hazard", default="daily_hazard.parquet")
    p.add_argument("--contamination", type=float, default=0.15)
    p.add_argument("--output_dir", default="models")
    p.add_argument("--cutoff_days", type=int, default=7)
    args = p.parse_args()

    project_dir = Path(__file__).resolve().parent
    os.makedirs(args.output_dir, exist_ok=True)

    mlflow_info = setup_mlflow(project_dir)
    print(f"MLflow tracking : {mlflow_info['tracking_uri']}")

    print("Loading data...")
    static = pd.read_csv(args.static, encoding="latin1")
    hazard = pd.read_parquet(args.hazard)

    static["facility_id"] = pd.to_numeric(static["facility_id"], errors="raise").astype(int)
    hazard["facility_id"] = pd.to_numeric(hazard["facility_id"], errors="raise").astype(int)
    hazard["date"] = pd.to_datetime(hazard["date"])

    cutoff = pd.Timestamp(date.today() - timedelta(days=args.cutoff_days))
    merged = (
        hazard[hazard["date"] <= cutoff]
        .merge(static, on="facility_id", how="left")
    )

    print(f"Merged dataset: {len(merged):,} rows | {merged['facility_id'].nunique()} facilities")

    print("Computing facility-specific objective indices...")
    merged = compute_indices(merged)

    print("Applying per-facility rule-based labelling...")
    merged = label_rows(merged)

    print("Running objective evaluation on indices...")
    evaluate_model_performance(merged)

    dyn_cols = [c for c in DYNAMIC if c in merged.columns]
    all_cols = [c for c in ALL if c in merged.columns]

    experiments = [
        (dyn_cols, "Exp1_IF_Dynamic"),
        (all_cols, "Exp2_IF_AllFeatures"),
    ]

    all_results = {}
    for feats, exp_label in experiments:
        print(f"\n{'=' * 58}")
        print(f"  {exp_label}  |  model=IsolationForest  |  n_features={len(feats)}")
        print(f"{'=' * 58}")
        all_results[exp_label] = run_experiment(
            df=merged,
            feats=feats,
            contamination=args.contamination,
            out_dir=args.output_dir,
            exp_label=exp_label,
        )

    print("\n" + "=" * 65)
    print("  RUN SUMMARY")
    print("=" * 65)
    print(f"  {'Experiment':<28} | {'F1-High':>8} | {'F1-Any':>8} | {'Anomaly%':>9}")
    print("  " + "-" * 62)
    for exp_label, res in all_results.items():
        m = res["metrics"]
        print(
            f"  {exp_label:<28} | "
            f"{m.get('f1_high', 0):>8.3f} | "
            f"{m.get('f1_any', 0):>8.3f} | "
            f"{m.get('anomaly_rate', 0) * 100:>8.1f}%"
        )

    print("\n✓ Done.")
    print(f"  MLflow DB : {mlflow_info['db_path']}")
    print(f"\n  Open the MLflow UI — paste this into a NEW terminal:")
    print(f'  mlflow ui --backend-store-uri "{mlflow_info["tracking_uri"]}"')
    print(f"  Then open: http://localhost:5000")
    print("  NOTE: do NOT just run 'mlflow ui' — it will open an empty database")


if __name__ == "__main__":
    main()