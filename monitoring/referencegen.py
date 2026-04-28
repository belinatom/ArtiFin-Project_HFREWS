import os
from pathlib import Path

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

# ── Reference window: 2 years of data, Jan 2023 → Dec 2024 ───────────────────
# Batches in 2025 will be compared against this baseline.
REFERENCE_START = "2023-01-01"
REFERENCE_END   = "2024-12-31"


def score_features(pipeline, X) -> np.ndarray:
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
    # 3. Filter to reference window: 2023-01-01 → 2024-12-31
    # -------------------------------------------------------------------------
    mask = (hazard_df["date"] >= REFERENCE_START) & (hazard_df["date"] <= REFERENCE_END)
    reference_df = hazard_df[mask].copy()

    if reference_df.empty:
        raise ValueError(
            f"Reference dataset is empty for {REFERENCE_START} → {REFERENCE_END}. "
            f"Available range: {hazard_df['date'].min().date()} → {hazard_df['date'].max().date()}"
        )

    print(f"Reference window : {REFERENCE_START} → {REFERENCE_END}")
    print(f"Rows in window   : {len(reference_df):,}")

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

    X = df[feats_avail].fillna(0)

    # -------------------------------------------------------------------------
    # 6. Score reference dataset
    # -------------------------------------------------------------------------
    print("Scoring reference data...")
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
    print(f"\nReference dataset saved: {out_file}")
    print(f"Rows       : {len(reference):,}")
    print(f"Facilities : {reference['facility_id'].nunique()}")
    print(f"Date range : {reference['date'].min()} → {reference['date'].max()}")
    print(f"\nNow run demo batches from 2025:")
    print(f"  python batchgenerate.py 2025-03-15")
    print(f"  python monitormetrics.py")


if __name__ == "__main__":
    main()
