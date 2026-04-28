import os
from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

# Paths (same as your project)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_FILE = BASE_DIR / "hfr_data.csv"
HAZARD_FILE = BASE_DIR / "daily_hazard.parquet"

PRODUCTION_MODEL = "Exp2_IF_AllFeatures.pkl"

FEATURES = [
    "rain_1d",
    "rain_3d_sum",
    "rain_7d_sum",
    "rain_percentile",
    "rain_anomaly",
    "river_discharge_m3s",
    "discharge_percentile",
    "swi_1d",
    "swi_7d_mean",
    "swi_3d_trend",
    "swi_percentile",
    "runoff_1d",
    "runoff_3d_sum",
    "runoff_7d_sum",
    "runoff_percentile",
    "water_accumulation_index",
    "dist_to_river_km",
]


def score_features(pipeline, X: np.ndarray) -> np.ndarray:
    Xs = pipeline.named_steps["scaler"].transform(X)
    raw = -pipeline.named_steps["detector"].score_samples(Xs)

    mn, mx = raw.min(), raw.max()
    if mx - mn < 1e-9:
        return np.full(len(raw), 0.5)

    return np.clip((raw - mn) / (mx - mn), 0, 1)


def classify(score: float) -> str:
    if score >= 0.65:
        return "High Risk"
    if score >= 0.35:
        return "Moderate Risk"
    return "No Risk"


def main():
    # -------------------------------------------------------------------------
    # 1. Create monitoring folder
    # -------------------------------------------------------------------------
    out_dir = BASE_DIR / "monitoring"
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Load model and data
    # -------------------------------------------------------------------------
    model_path = MODELS_DIR / PRODUCTION_MODEL
    pipeline = joblib.load(model_path)

    static_df = pd.read_csv(STATIC_FILE, encoding="latin1")
    if "facility_id" not in static_df.columns:
        static_df.insert(0, "facility_id", range(len(static_df)))
    static_df["facility_id"] = static_df["facility_id"].astype(int)

    hazard_df = pd.read_parquet(HAZARD_FILE)
    hazard_df["facility_id"] = pd.to_numeric(hazard_df["facility_id"], errors="raise").astype(int)
    hazard_df["date"] = pd.to_datetime(hazard_df["date"])

    # -------------------------------------------------------------------------
    # 3. Define reference window (exclude recent data)
    # -------------------------------------------------------------------------
    cutoff_days = 30  # last 30 days excluded from baseline
    cutoff_date = pd.Timestamp(date.today() - timedelta(days=cutoff_days))

    reference_df = hazard_df[hazard_df["date"] <= cutoff_date].copy()

    if reference_df.empty:
        raise ValueError("Reference dataset is empty. Not enough historical data.")

    # -------------------------------------------------------------------------
    # 4. Merge static data
    # -------------------------------------------------------------------------
    df = reference_df.merge(static_df, on="facility_id", how="left")

    # -------------------------------------------------------------------------
    # 5. Build feature matrix
    # -------------------------------------------------------------------------
    feats_avail = [f for f in FEATURES if f in df.columns]
    if len(feats_avail) != len(FEATURES):
        missing = sorted(set(FEATURES) - set(feats_avail))
        raise ValueError(f"Missing required features: {missing}")

    X = df[feats_avail].fillna(0).values

    # -------------------------------------------------------------------------
    # 6. Score reference dataset
    # -------------------------------------------------------------------------
    df["final_score"] = score_features(pipeline, X)
    df["risk_class"] = df["final_score"].apply(classify)
    df["date"] = df["date"].dt.date.astype(str)

    # -------------------------------------------------------------------------
    # 7. Select columns for monitoring
    # -------------------------------------------------------------------------
    keep_cols = [
        "facility_id",
        "name",
        "region",
        "facility_type",
        "lat",
        "lon",
        "date",
        *FEATURES,
        "final_score",
        "risk_class",
    ]

    reference = df[[c for c in keep_cols if c in df.columns]].copy()
    reference = reference.sort_values("date").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 8. Save reference dataset
    # -------------------------------------------------------------------------
    out_file = out_dir / "reference.parquet"
    reference.to_parquet(out_file, index=False)

    # -------------------------------------------------------------------------
    # 9. Print summary
    # -------------------------------------------------------------------------
    print(f"Reference dataset saved: {out_file}")
    print(f"Rows: {len(reference)}")
    print(f"Facilities: {reference['facility_id'].nunique()}")
    print(f"Date range: {reference['date'].min()} → {reference['date'].max()}")


if __name__ == "__main__":
    main()