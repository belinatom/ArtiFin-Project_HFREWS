import glob
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import psycopg
from scipy.stats import ks_2samp

FEATURES = [
    "rain_1d","rain_3d_sum","rain_7d_sum","rain_percentile","rain_anomaly",
    "river_discharge_m3s","discharge_percentile",
    "swi_1d","swi_7d_mean","swi_3d_trend","swi_percentile",
    "runoff_1d","runoff_3d_sum","runoff_7d_sum","runoff_percentile",
    "water_accumulation_index","dist_to_river_km",
]

BASE_DIR = Path(__file__).resolve().parent

def get_db_conn():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "flood_ews"),
        user=os.getenv("POSTGRES_USER", "flood_ews"),
        password=os.getenv("POSTGRES_PASSWORD", "flood_ews_pw"),
    )

def main():
    # Load only needed columns from reference — saves ~80% memory
    ref_cols = FEATURES + ["final_score"]
    reference = pd.read_parquet(
        BASE_DIR / "monitoring" / "reference.parquet",
        columns=[c for c in ref_cols]
    )

    batch_files = glob.glob(str(BASE_DIR / "monitoring" / "current_batches" / "*.parquet"))
    if not batch_files:
        raise FileNotFoundError("No batch files found. Run batchgenerate.py first.")

    latest_file = max(batch_files, key=os.path.getmtime)
    current = pd.read_parquet(latest_file)
    batch_id = os.path.basename(latest_file).replace(".parquet", "")

    batch_date = None
    if "date" in current.columns and len(current) > 0:
        batch_date = pd.to_datetime(current["date"]).iloc[0].date()

    print(f"Running metrics for batch: {batch_id}")

    # KS test per feature
    drifted_features = 0
    feature_drift_details = {}
    for feature in FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            feature_drift_details[feature] = None
            continue
        ref_vals = reference[feature].dropna().values
        cur_vals = current[feature].dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            feature_drift_details[feature] = None
            continue
        _, p_value = ks_2samp(ref_vals, cur_vals)
        p_value = float(p_value)
        feature_drift_details[feature] = p_value
        if p_value < 0.05:
            drifted_features += 1

    share_drifted = drifted_features / len(FEATURES)

    score_drift_pvalue = None
    if "final_score" in reference.columns and "final_score" in current.columns:
        _, score_drift_pvalue = ks_2samp(
            reference["final_score"].dropna().values,
            current["final_score"].dropna().values
        )
        score_drift_pvalue = float(score_drift_pvalue)

    # Free reference from memory now
    del reference

    mean_final_score = float(current["final_score"].mean()) if "final_score" in current.columns else None
    max_final_score  = float(current["final_score"].max())  if "final_score" in current.columns else None
    std_final_score  = float(current["final_score"].std())  if "final_score" in current.columns else None

    risk_share = {}
    if "risk_class" in current.columns:
        risk_share = current["risk_class"].value_counts(normalize=True).to_dict()
    high_risk_share     = float(risk_share.get("High Risk", 0.0))
    moderate_risk_share = float(risk_share.get("Moderate Risk", 0.0))
    no_risk_share       = float(risk_share.get("No Risk", 0.0))

    available = [f for f in FEATURES if f in current.columns]
    total = len(current) * len(available) if available else 1
    missing = current[available].isna().sum().sum() if available else 0
    missing_feature_share = float(missing / total)

    conn = get_db_conn()
    now = datetime.utcnow()
    try:
        with conn.cursor() as cur:
            # Write predictions so all Grafana stat cards update
            if batch_date and "final_score" in current.columns:
                cur.execute("DELETE FROM predictions WHERE date = %s", (batch_date,))
                rows = [(int(r.get("facility_id", 0)), batch_date,
                         float(r["final_score"]), str(r["risk_class"]))
                        for _, r in current.iterrows()]
                cur.executemany(
                    "INSERT INTO predictions (facility_id, date, final_score, risk_class) VALUES (%s,%s,%s,%s)",
                    rows)

            # Write monitoring metrics
            cur.execute("DELETE FROM monitoring_metrics WHERE batch_date = %s", (batch_date,))
            cur.execute("""
                INSERT INTO monitoring_metrics (
                    timestamp, batch_id, batch_date, batch_size,
                    num_drifted_features, share_drifted_features,
                    score_drift_pvalue,
                    mean_final_score, max_final_score, std_final_score,
                    high_risk_share, moderate_risk_share, no_risk_share,
                    missing_feature_share
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (now, batch_id, batch_date, len(current),
                  drifted_features, share_drifted, score_drift_pvalue,
                  mean_final_score, max_final_score, std_final_score,
                  high_risk_share, moderate_risk_share, no_risk_share,
                  missing_feature_share))

            # Write feature drift
            cur.execute("DELETE FROM feature_drift WHERE batch_id = %s", (batch_id,))
            for feature, p_value in feature_drift_details.items():
                drifted = (p_value is not None and p_value < 0.05)
                cur.execute("""
                    INSERT INTO feature_drift (recorded_at, batch_id, batch_date, feature, ks_pvalue, drifted)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (now, batch_id, batch_date, feature, p_value, drifted))

        conn.commit()
    finally:
        conn.close()

    counts = current["risk_class"].value_counts().to_dict() if "risk_class" in current.columns else {}
    print(f"Monitoring metrics saved  |  batch={batch_id}  |  size={len(current)}")
    print(f"High Risk        : {counts.get('High Risk', 0)}")
    print(f"Moderate Risk    : {counts.get('Moderate Risk', 0)}")
    print(f"No Risk          : {counts.get('No Risk', 0)}")
    print(f"Drifted features : {drifted_features}/{len(FEATURES)}  ({share_drifted:.1%})")
    print(f"Score drift p    : {score_drift_pvalue}")
    print(f"Mean score       : {mean_final_score:.4f}")
    print(f"Grafana updated.")

if __name__ == "__main__":
    main()
