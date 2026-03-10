# api.py
#
# Web server that serves predictions from the cutoff risk model.
#
# Endpoints:
#   GET  /health          — check the server is running
#   POST /predict/run     — trigger a full pipeline run (pulls GEE data + scores all facilities)
#   GET  /predict/latest  — return the most recent saved predictions
#
# Usage:
#   uvicorn api:app --reload
#
# Then open http://127.0.0.1:8000/docs to test in your browser.


import os
import tempfile
import warnings
from datetime import date
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

MLFLOW_DB     = "sqlite:///mlflow_cutoff_alerts.db"
MODEL_NAME    = "FacilityCutoffRiskClassifier"
MODEL_VERSION = "latest"

STATIC_PATH  = "hfr_data.csv"
RAIN_CACHE   = "chirps_baseline.parquet"
SM_CACHE     = "era5_baseline.parquet"
EE_PROJECT   = "mhews-tz"

DRAINAGE_RISK_SCORES = {
    "SE": 1,
    "MW": 2,
    "I":  3,
    "P":  4,
    "VP": 5,
}
DRAINAGE_RISK_DEFAULT = 2

RAINFALL_FEATURES = [
    "rain_1d", "rain_3d_sum", "rain_7d_sum", "rain_14d_sum",
    "rain_percentile", "rain_p90_flag", "rain_p95_flag",
    "rain_anomaly", "rain_days_above_20mm",
]
SOIL_MOISTURE_FEATURES = [
    "sm_1d", "sm_7d_mean", "sm_percentile", "sm_p90_flag",
    "sm_p95_flag", "sm_14d_mean", "sm_change_3d", "sm_change_7d",
]
STATIC_FEATURES = ["elevation_m", "slope_deg", "drainage_risk", "TEXTURE_USDA"]
ALL_FEATURES    = RAINFALL_FEATURES + SOIL_MOISTURE_FEATURES + STATIC_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Load model once when the server starts
# ─────────────────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri(MLFLOW_DB)
print(f"Loading model '{MODEL_NAME}' from MLflow...")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
print("Model loaded successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline functions (same as predict.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_static_features(path):
    df = pd.read_csv(path)
    df["facility_id"] = (
        df["lat"].round(5).astype(str) + "_" + df["lon"].round(5).astype(str)
    )
    df["drainage_risk"] = (
        df["DRAINAGE"].map(DRAINAGE_RISK_SCORES).fillna(DRAINAGE_RISK_DEFAULT).astype(int)
    )
    df["TEXTURE_USDA"] = df["TEXTURE_USDA"].fillna(df["TEXTURE_USDA"].median())
    for col in ["elevation_m", "slope_deg"]:
        df[col] = df.groupby("region")[col].transform(lambda x: x.fillna(x.median()))

    static_df  = df[["facility_id", "name", "region", "facility_class", "lat", "lon"] + STATIC_FEATURES].copy()
    facilities = df[["facility_id", "lat", "lon"]]
    return static_df, facilities


def pull_fresh_features(facilities, run_date, rain_cache, sm_cache, ee_project):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        facilities.to_csv(f, index=False)
        temp_path = f.name
    try:
        import rainfall as rain_mod
        rain_df = rain_mod.run_pipeline(
            facilities_path     = temp_path,
            run_date            = run_date,
            baseline_cache_path = rain_cache,
            ee_project          = ee_project,
        )
        import soilmoisture as sm_mod
        sm_df = sm_mod.run_pipeline(
            facilities_path     = temp_path,
            run_date            = run_date,
            baseline_cache_path = sm_cache,
            ee_project          = ee_project,
        )
    finally:
        os.unlink(temp_path)
    return rain_df, sm_df


def merge_all_sources(static_df, rain_df, sm_df):
    dynamic = rain_df.merge(sm_df, on=["facility_id", "date"], how="inner")
    merged  = dynamic.merge(
        static_df[["facility_id"] + STATIC_FEATURES + ["name", "region", "facility_class", "lat", "lon"]],
        on="facility_id", how="left",
    )
    return merged


def score_facilities(merged_df):
    X             = merged_df[ALL_FEATURES].values
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)

    output = merged_df.copy()
    output["predicted_alert"] = predictions
    output["confidence"]      = probabilities.max(axis=1).round(4)
    for i, class_name in enumerate(model.classes_):
        output[f"prob_{class_name}"] = probabilities[:, i].round(4)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Request / response shapes
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Parameters for triggering a prediction run."""
    run_date:   str | None = None   # YYYY-MM-DD, defaults to today
    ee_project: str        = EE_PROJECT
    rain_cache: str        = RAIN_CACHE
    sm_cache:   str        = SM_CACHE
    static:     str        = STATIC_PATH


class RunStatus(BaseModel):
    """What the API returns after a run completes."""
    run_date:        str
    output_file:     str
    total_facilities: int
    alert_summary:   dict


# ─────────────────────────────────────────────────────────────────────────────
# Create the FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Health Facility Cutoff Risk API",
    description="Triggers the full prediction pipeline and returns cutoff risk alerts.",
    version="2.0.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check that the server and model are running."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict/run", response_model=RunStatus)
def run_pipeline(request: RunRequest):
    """
    Triggers the full prediction pipeline:
      1. Pulls fresh GEE data for the given date
      2. Merges with static facility features
      3. Scores all facilities with the trained model
      4. Saves results to a CSV and returns a summary

    This will take several minutes while GEE data is being pulled.
    """
    try:
        run_date    = date.fromisoformat(request.run_date) if request.run_date else date.today()
        output_file = f"predictions_{run_date}.csv"

        # Run the pipeline
        static_df, facilities = load_static_features(request.static)
        rain_df, sm_df        = pull_fresh_features(
            facilities,
            run_date    = run_date,
            rain_cache  = request.rain_cache,
            sm_cache    = request.sm_cache,
            ee_project  = request.ee_project,
        )
        merged      = merge_all_sources(static_df, rain_df, sm_df)
        predictions = score_facilities(merged)

        # Save to CSV
        predictions.to_csv(output_file, index=False)

        # Build alert summary
        alert_summary = predictions["predicted_alert"].value_counts().to_dict()

        return RunStatus(
            run_date         = str(run_date),
            output_file      = output_file,
            total_facilities = len(predictions),
            alert_summary    = alert_summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/latest")
def get_latest_predictions(alert: str | None = None):
    """
    Returns the most recently saved predictions CSV as JSON.

    Optionally filter by alert level:
      ?alert=CUTOFF_HIGH
      ?alert=CUTOFF_MODERATE
      ?alert=CUTOFF_WATCH
      ?alert=NO_ALERT
    """
    # Find the most recent predictions file
    prediction_files = sorted(Path(".").glob("predictions_*.csv"), reverse=True)

    if not prediction_files:
        raise HTTPException(
            status_code=404,
            detail="No predictions found. Run /predict/run first."
        )

    df = pd.read_csv(prediction_files[0])

    # Filter by alert level if requested
    if alert:
        df = df[df["predicted_alert"] == alert.upper()]

    # Return as a list of records
    return {
        "file":    prediction_files[0].name,
        "count":   len(df),
        "results": df.to_dict(orient="records"),
    }
