import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_FILE = BASE_DIR / "hfr_data.csv"
HAZARD_FILE = BASE_DIR / "daily_hazard.parquet"
PRODUCTION_MODEL = "Exp2_IF_AllFeatures.pkl"

FEATURES = [
    "rain_1d","rain_3d_sum","rain_7d_sum","rain_percentile","rain_anomaly",
    "river_discharge_m3s","discharge_percentile",
    "swi_1d","swi_7d_mean","swi_3d_trend","swi_percentile",
    "runoff_1d","runoff_3d_sum","runoff_7d_sum","runoff_percentile",
    "water_accumulation_index","dist_to_river_km",
]

def score_features(pipeline, X):
    Xs = pipeline.named_steps["scaler"].transform(X)
    raw = -pipeline.named_steps["detector"].score_samples(Xs)
    mn, mx = raw.min(), raw.max()
    if mx - mn < 1e-9:
        return np.full(len(raw), 0.5)
    return np.clip((raw - mn) / (mx - mn), 0, 1)

def classify(score):
    if score >= 0.65: return "High Risk"
    if score >= 0.35: return "Moderate Risk"
    return "No Risk"

def main():
    out_dir = BASE_DIR / "monitoring" / "current_batches"
    os.makedirs(out_dir, exist_ok=True)

    pipeline = joblib.load(MODELS_DIR / PRODUCTION_MODEL)

    static_df = pd.read_csv(STATIC_FILE, encoding="latin1")
    if "facility_id" not in static_df.columns:
        static_df.insert(0, "facility_id", range(len(static_df)))
    static_df["facility_id"] = static_df["facility_id"].astype(int)

    # Read only the single date needed — much less memory
    target_date = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else None

    hazard_df = pd.read_parquet(HAZARD_FILE)
    hazard_df["facility_id"] = pd.to_numeric(hazard_df["facility_id"], errors="raise").astype(int)
    hazard_df["date"] = pd.to_datetime(hazard_df["date"])

    available = sorted(hazard_df["date"].unique())
    if target_date:
        nearest = min(available, key=lambda d: abs(d - target_date))
    else:
        nearest = max(available)

    current = hazard_df[hazard_df["date"] == nearest].copy()
    del hazard_df  # free memory immediately

    if current.empty:
        raise ValueError(f"No data for {nearest}")

    df = current.merge(static_df, on="facility_id", how="left")
    feats_avail = [f for f in FEATURES if f in df.columns]
    X = df[feats_avail].fillna(0)

    df["final_score"] = score_features(pipeline, X)
    df["risk_class"] = df["final_score"].apply(classify)
    df["date"] = nearest.date().isoformat()

    keep = ["facility_id","name","region","facility_type","lat","lon","date",
            *FEATURES,"final_score","risk_class"]
    batch = df[[c for c in keep if c in df.columns]].sort_values("final_score", ascending=False)

    batch_id = nearest.date().isoformat()
    out_file = out_dir / f"{batch_id}.parquet"
    batch.to_parquet(out_file, index=False)

    counts = batch["risk_class"].value_counts().to_dict()
    print(f"Current batch saved: {out_file}")
    print(f"Batch date: {batch_id}")
    print(f"Rows: {len(batch)}")
    print(f"High Risk: {counts.get('High Risk', 0)}")
    print(f"Moderate Risk: {counts.get('Moderate Risk', 0)}")
    print(f"No Risk: {counts.get('No Risk', 0)}")

if __name__ == "__main__":
    main()
