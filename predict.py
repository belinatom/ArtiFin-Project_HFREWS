# predict.py
#
# Runs the full prediction pipeline:
#   1. Pulls fresh rainfall data from GEE (uses cached baseline parquet)
#   2. Pulls fresh soil moisture data from GEE (uses cached baseline parquet)
#   3. Loads static facility features from hfr_data.csv
#   4. Merges all three sources
#   5. Loads the trained model from MLflow
#   6. Scores all facilities and saves predictions to a CSV
#
# Usage:
#   python predict.py \
#       --static      hfr_data.csv \
#       --rain_cache  chirps_baseline.parquet \
#       --sm_cache    era5_baseline.parquet \
#       --ee_project  mhews-tz
#
# Output:
#   predictions_YYYY-MM-DD.csv


import argparse
import os
import tempfile
import warnings
from datetime import date
from pathlib import Path

import mlflow.sklearn
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

MLFLOW_DB     = "sqlite:///mlflow_cutoff_alerts.db"
MODEL_NAME    = "FacilityCutoffRiskClassifier"
MODEL_VERSION = "latest"

DRAINAGE_RISK_SCORES = {
    "SE": 1,
    "MW": 2,
    "I":  3,
    "P":  4,
    "VP": 5,
}
DRAINAGE_RISK_DEFAULT = 2

RAINFALL_FEATURES = [
    "rain_1d",
    "rain_3d_sum",
    "rain_7d_sum",
    "rain_14d_sum",
    "rain_percentile",
    "rain_p90_flag",
    "rain_p95_flag",
    "rain_anomaly",
    "rain_days_above_20mm",
]

SOIL_MOISTURE_FEATURES = [
    "sm_1d",
    "sm_7d_mean",
    "sm_percentile",
    "sm_p90_flag",
    "sm_p95_flag",
    "sm_14d_mean",
    "sm_change_3d",
    "sm_change_7d",
]

STATIC_FEATURES = [
    "elevation_m",
    "slope_deg",
    "drainage_risk",
    "TEXTURE_USDA",
]

ALL_FEATURES = RAINFALL_FEATURES + SOIL_MOISTURE_FEATURES + STATIC_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load static facility features
# ─────────────────────────────────────────────────────────────────────────────

def load_static_features(path):
    """Loads hfr_data.csv and prepares static features and the facilities lookup table."""
    df = pd.read_csv(path)

    df["facility_id"] = (
        df["lat"].round(5).astype(str)
        + "_"
        + df["lon"].round(5).astype(str)
    )

    df["drainage_risk"] = (
        df["DRAINAGE"]
        .map(DRAINAGE_RISK_SCORES)
        .fillna(DRAINAGE_RISK_DEFAULT)
        .astype(int)
    )

    df["TEXTURE_USDA"] = df["TEXTURE_USDA"].fillna(df["TEXTURE_USDA"].median())

    for column in ["elevation_m", "slope_deg"]:
        df[column] = df.groupby("region")[column].transform(
            lambda x: x.fillna(x.median())
        )

    static_df = df[
        ["facility_id", "name", "region", "facility_class", "lat", "lon"]
        + STATIC_FEATURES
    ].copy()

    # Facilities lookup table needed by rainfall.py and soilmoisture.py
    facilities = df[["facility_id", "lat", "lon"]]

    print(f"Loaded {len(static_df):,} facilities from {path}")
    return static_df, facilities


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Pull fresh GEE data
# ─────────────────────────────────────────────────────────────────────────────

def pull_fresh_features(facilities, run_date, rain_cache, sm_cache, ee_project):
    """
    Calls rainfall.py and soilmoisture.py to pull today's data from GEE.
    Uses the cached baseline parquets so the 10-year history is not re-downloaded.
    """
    # Write facilities to a temp CSV (both pipelines read from a file)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        facilities.to_csv(f, index=False)
        temp_path = f.name

    try:
        print("\nStep 1/2 — Pulling rainfall data (rainfall.py)...")
        import rainfall as rain_mod
        rain_df = rain_mod.run_pipeline(
            facilities_path     = temp_path,
            run_date            = run_date,
            baseline_cache_path = rain_cache,
            ee_project          = ee_project,
        )
        print(f"  Done — {len(rain_df):,} facility rows")

        print("\nStep 2/2 — Pulling soil moisture data (soilmoisture.py)...")
        import soilmoisture as sm_mod
        sm_df = sm_mod.run_pipeline(
            facilities_path     = temp_path,
            run_date            = run_date,
            baseline_cache_path = sm_cache,
            ee_project          = ee_project,
        )
        print(f"  Done — {len(sm_df):,} facility rows")

    finally:
        os.unlink(temp_path)

    return rain_df, sm_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Merge all sources
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_sources(static_df, rain_df, sm_df):
    """Joins rainfall, soil moisture, and static features into one table."""
    dynamic = rain_df.merge(sm_df, on=["facility_id", "date"], how="inner")

    merged = dynamic.merge(
        static_df[["facility_id"] + STATIC_FEATURES + ["name", "region", "facility_class", "lat", "lon"]],
        on="facility_id",
        how="left",
    )

    unmatched = merged[STATIC_FEATURES[0]].isna().sum()
    print(f"\nMerged {len(merged):,} rows  (unmatched static: {unmatched})")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Load model from MLflow
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Loads the latest trained model from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    print(f"\nLoading model '{MODEL_NAME}' from MLflow...")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    print("  Model loaded.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Score all facilities
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions(model, merged_df):
    """
    Runs the model on all facilities and adds prediction columns.

    Adds:
      predicted_alert   — the alert level (e.g. CUTOFF_HIGH)
      confidence        — probability of the top class
      prob_CUTOFF_HIGH
      prob_CUTOFF_WATCH
      prob_NO_ALERT
    """
    X = merged_df[ALL_FEATURES].values
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)

    output = merged_df.copy()
    output["predicted_alert"] = predictions
    output["confidence"]      = probabilities.max(axis=1).round(4)

    for i, class_name in enumerate(model.classes_):
        output[f"prob_{class_name}"] = probabilities[:, i].round(4)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Save and summarise
# ─────────────────────────────────────────────────────────────────────────────

def save_and_summarise(predictions_df, run_date, output_path):
    """Saves predictions to CSV and prints a summary."""
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    print(f"\nAlert summary for {run_date}:")
    for alert, count in predictions_df["predicted_alert"].value_counts().items():
        print(f"  {alert:<22} {count:>6}  ({count / len(predictions_df) * 100:.1f}%)")

    # Show the high risk facilities
    high_risk = predictions_df[predictions_df["predicted_alert"] == "CUTOFF_HIGH"]
    if len(high_risk) > 0:
        print(f"\nHigh risk facilities ({len(high_risk)}):")
        display_cols = ["facility_id", "name", "region", "rain_7d_sum",
                        "sm_percentile", "confidence"]
        print(high_risk[[c for c in display_cols if c in high_risk.columns]]
              .sort_values("confidence", ascending=False)
              .to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full prediction pipeline — pulls GEE data and scores all facilities"
    )
    parser.add_argument("--static",      required=True, help="Path to hfr_data.csv")
    parser.add_argument("--rain_cache",  required=True, help="Path to chirps_baseline.parquet")
    parser.add_argument("--sm_cache",    required=True, help="Path to era5_baseline.parquet")
    parser.add_argument("--ee_project",  required=True, help="Google Earth Engine project ID")
    parser.add_argument("--run_date",    default=None,  help="Date to predict for (YYYY-MM-DD, default: today)")
    parser.add_argument("--output",      default=None,  help="Output CSV path (default: predictions_YYYY-MM-DD.csv)")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.run_date) if args.run_date else date.today()
    output_path = args.output or f"predictions_{run_date}.csv"

    print(f"\n{'='*62}")
    print(f"  Prediction run for : {run_date}")
    print(f"  Output             : {output_path}")
    print(f"{'='*62}")

    # Load static features
    print("\n── STEP 1: LOADING STATIC FEATURES ───────────────────────")
    static_df, facilities = load_static_features(args.static)

    # Pull fresh GEE data
    print("\n── STEP 2: PULLING FRESH GEE DATA ────────────────────────")
    rain_df, sm_df = pull_fresh_features(
        facilities,
        run_date    = run_date,
        rain_cache  = args.rain_cache,
        sm_cache    = args.sm_cache,
        ee_project  = args.ee_project,
    )

    # Merge
    print("\n── STEP 3: MERGING ────────────────────────────────────────")
    merged = merge_all_sources(static_df, rain_df, sm_df)

    # Load model
    print("\n── STEP 4: LOADING MODEL ──────────────────────────────────")
    model = load_model()

    # Score
    print("\n── STEP 5: SCORING FACILITIES ─────────────────────────────")
    predictions = run_predictions(model, merged)

    # Save
    print("\n── STEP 6: SAVING RESULTS ─────────────────────────────────")
    save_and_summarise(predictions, run_date, output_path)


if __name__ == "__main__":
    main()