"""
predict.py  —  Flood EWS Prediction Engine + FastAPI

Uses Exp2_IF_AllFeatures (IsolationForest, dynamic + terrain features)
— the best performing model from train.py.

Modes
-----
  CLI   python predict.py --date 2024-03-15
  API   uvicorn predict:app --host 0.0.0.0 --port 8000
        then open  http://localhost:8000/docs

Endpoints
---------
  POST /predict/manual          enter feature values → get risk score + label
  GET  /predict                 score all facilities for a date (latest default)
  GET  /predict?date=YYYY-MM-DD score all facilities for a specific date
  GET  /risk/high               High Risk facilities only
  GET  /risk/summary            risk count breakdown
  GET  /facility/{id}           single facility detail
  GET  /health                  model status

Dependencies
------------
  pip install fastapi uvicorn joblib numpy pandas scikit-learn pyarrow
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Paths (resolve relative to this script's directory) ───────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = str(BASE_DIR / "models")
STATIC_FILE = str(BASE_DIR / "hfr_data.csv")
HAZARD_FILE = str(BASE_DIR / "daily_hazard.parquet")

# ── Production model — best experiment from train.py ─────────────────────────
PRODUCTION_MODEL = "Exp2_IF_AllFeatures.pkl"

# ── Feature columns the production model was trained on ───────────────────────
FEATURES = [
    "rain_1d", "rain_3d_sum", "rain_7d_sum", "rain_percentile", "rain_anomaly",
    "river_discharge_m3s", "discharge_percentile",
    "swi_raw", "swi_absorption_deficit", "swi_7d_mean", "swi_3d_trend", "swi_percentile",
    "runoff_1d", "runoff_3d_sum", "runoff_7d_sum", "runoff_percentile",
    "water_accumulation_index", "dist_to_river_km",
]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL STORE — loaded once, shared across all requests
# ══════════════════════════════════════════════════════════════════════════════
class ModelStore:
    pipeline  = None   # sklearn Pipeline: StandardScaler + IsolationForest
    static_df = None   # facility attribute table
    hazard_df = None   # daily hazard feature table

    @classmethod
    def load(cls,
             models_dir:  str = MODELS_DIR,
             static_file: str = STATIC_FILE,
             hazard_file: str = HAZARD_FILE) -> None:
        model_path  = os.path.join(models_dir, PRODUCTION_MODEL)
        cls.pipeline = joblib.load(model_path)
        log.info("Loaded model: %s", model_path)

        cls.static_df = pd.read_csv(static_file, encoding="latin1")
        # Add facility_id as row position if not already present
        # (train.py now saves it, but guard for older CSVs)
        if "facility_id" not in cls.static_df.columns:
            cls.static_df.insert(0, "facility_id", range(len(cls.static_df)))
        cls.static_df["facility_id"] = cls.static_df["facility_id"].astype(int)

        cls.hazard_df = pd.read_parquet(hazard_file)
        cls.hazard_df["date"] = pd.to_datetime(cls.hazard_df["date"])

        log.info(
            "Data loaded | facilities: %d | hazard dates: %d",
            len(cls.static_df),
            cls.hazard_df["date"].nunique(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCORING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def score_features(X: np.ndarray) -> np.ndarray:
    """
    Pass a feature matrix through the pipeline and return normalised [0,1] scores.
    Higher score = more anomalous = higher flood risk.
    """
    pipe = ModelStore.pipeline
    Xs   = pipe.named_steps["scaler"].transform(X)
    raw  = -pipe.named_steps["detector"].score_samples(Xs)
    # Normalise using training score distribution
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


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC INPUT SCHEMA — one row of raw hazard + terrain features
# Defaults represent neutral / no-event conditions so users only need to
# change the values they care about when testing in the Swagger UI.
# ══════════════════════════════════════════════════════════════════════════════
class FacilityFeatures(BaseModel):
    rain_1d:                  float = Field(0.0,  description="1-day rainfall (mm)")
    rain_3d_sum:              float = Field(0.0,  description="3-day cumulative rainfall (mm)")
    rain_7d_sum:              float = Field(0.0,  description="7-day cumulative rainfall (mm)")
    rain_percentile:          float = Field(0.5,  description="Rainfall percentile vs climatology (0–1)")
    rain_anomaly:             float = Field(0.0,  description="Rainfall anomaly vs climatology (mm)")
    river_discharge_m3s:      float = Field(0.0,  description="River discharge (m³/s)")
    discharge_percentile:     float = Field(0.5,  description="Discharge percentile (0–1)")
    swi_raw:                  float = Field(0.5,  description="Soil water index (0–1)")
    swi_absorption_deficit:   float = Field(0.5,  description="SWI absorption deficit (0–1)")
    swi_7d_mean:              float = Field(0.5,  description="7-day mean SWI (0–1)")
    swi_3d_trend:             float = Field(0.0,  description="3-day SWI trend")
    swi_percentile:           float = Field(0.5,  description="SWI percentile (0–1)")
    runoff_1d:                float = Field(0.0,  description="1-day surface runoff (mm)")
    runoff_3d_sum:            float = Field(0.0,  description="3-day cumulative runoff (mm)")
    runoff_7d_sum:            float = Field(0.0,  description="7-day cumulative runoff (mm)")
    runoff_percentile:        float = Field(0.5,  description="Runoff percentile (0–1)")
    water_accumulation_index: float = Field(0.5,  description="Terrain water accumulation index (0–1)")
    dist_to_river_km:         float = Field(5.0,  description="Distance to nearest river (km)")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE HANDLERS — plain functions, registered explicitly below
# ══════════════════════════════════════════════════════════════════════════════

def predict_manual(payload: FacilityFeatures):
    """
    Enter raw feature values and get back a flood risk score and label.
    All fields have defaults so you only need to fill in the values you want to test.
    """
    row   = [[getattr(payload, f) for f in FEATURES]]
    X     = np.array(row, dtype=float)
    score = float(score_features(X)[0])
    return {
        "final_score": round(score, 4),
        "risk_class":  classify(score),
        "model":       PRODUCTION_MODEL,
    }


def predict_date(
    date: Optional[str] = Query(
        None, description="YYYY-MM-DD — defaults to latest available date"
    )
):
    """Score all facilities for a given date."""
    try:
        results = run_predictions(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "date":         results["date"].iloc[0],
        "n_facilities": len(results),
        "predictions":  results.to_dict(orient="records"),
    }


def high_risk(date: Optional[str] = Query(None, description="YYYY-MM-DD")):
    """Return only High Risk facilities for a given date."""
    try:
        results = run_predictions(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    high = results[results["risk_class"] == "High Risk"]
    return {
        "date":        results["date"].iloc[0],
        "n_high_risk": len(high),
        "facilities":  high.to_dict(orient="records"),
    }


def risk_summary(date: Optional[str] = Query(None, description="YYYY-MM-DD")):
    """Risk count breakdown for a given date."""
    try:
        results = run_predictions(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    counts = results["risk_class"].value_counts().to_dict()
    return {
        "date":          results["date"].iloc[0],
        "total":         len(results),
        "high_risk":     counts.get("High Risk",     0),
        "moderate_risk": counts.get("Moderate Risk", 0),
        "no_risk":       counts.get("No Risk",       0),
        "mean_score":    round(float(results["final_score"].mean()), 4),
    }


def facility_detail(
    facility_id: int,
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """Full prediction detail for a single facility."""
    try:
        results = run_predictions(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    row = results[results["facility_id"] == facility_id]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Facility {facility_id} not found for this date.",
        )
    return row.to_dict(orient="records")[0]


def health():
    """Model and data readiness check."""
    m = ModelStore
    return {
        "status":      "ok" if m.pipeline is not None else "model not loaded",
        "model":       PRODUCTION_MODEL,
        "n_features":  len(FEATURES),
        "n_facilities": len(m.static_df) if m.static_df is not None else 0,
        "hazard_date_min": m.hazard_df["date"].min().date().isoformat() if m.hazard_df is not None else None,
        "hazard_date_max": m.hazard_df["date"].max().date().isoformat() if m.hazard_df is not None else None,
        "available_dates": m.hazard_df["date"].nunique() if m.hazard_df is not None else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION (used by date-based endpoints)
# ══════════════════════════════════════════════════════════════════════════════
def run_predictions(target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Score every facility for a given date from the hazard parquet.
    Returns a DataFrame sorted by final_score descending.
    """
    m  = ModelStore
    dt = pd.to_datetime(target_date) if target_date else m.hazard_df["date"].max()

    daily = m.hazard_df[m.hazard_df["date"] == dt].copy()
    if daily.empty:
        raise ValueError(f"No hazard data available for {dt.date()}")

    df          = daily.merge(m.static_df, on="facility_id", how="left")
    feats_avail = [f for f in FEATURES if f in df.columns]

    X               = df[feats_avail].fillna(0).values
    df["final_score"] = score_features(X)
    df["risk_class"]  = df["final_score"].apply(classify)
    df["date"]        = dt.date().isoformat()

    out_cols = [
        "facility_id", "name", "region", "facility_type", "lat", "lon",
        "date", "final_score", "risk_class",
    ]
    return (
        df[[c for c in out_cols if c in df.columns]]
        .sort_values("final_score", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(application: FastAPI):
    ModelStore.load()
    yield


app = FastAPI(
    title="Flood Early Warning System",
    description=(
        "Predicts flood-induced inaccessibility risk for Tanzanian primary "
        "healthcare facilities using IsolationForest trained on CHIRPS rainfall "
        "and GloFAS river-discharge features.\n\n"
        "**Quick start:** Use `POST /predict/manual` to enter feature values "
        "and get a risk score instantly."
    ),
    version="3.0",
    lifespan=lifespan,
)

# ── Explicit route registration (no decorators) ───────────────────────────────
app.add_api_route(
    "/predict/manual",
    predict_manual,
    methods=["POST"],
    summary="Enter feature values → get risk score",
    tags=["Manual Prediction"],
)
app.add_api_route(
    "/predict",
    predict_date,
    methods=["GET"],
    summary="Score all facilities for a date",
    tags=["Batch Prediction"],
)
app.add_api_route(
    "/risk/high",
    high_risk,
    methods=["GET"],
    summary="High Risk facilities only",
    tags=["Batch Prediction"],
)
app.add_api_route(
    "/risk/summary",
    risk_summary,
    methods=["GET"],
    summary="Risk count breakdown",
    tags=["Batch Prediction"],
)
app.add_api_route(
    "/facility/{facility_id}",
    facility_detail,
    methods=["GET"],
    summary="Single facility detail",
    tags=["Batch Prediction"],
)
app.add_api_route(
    "/health",
    health,
    methods=["GET"],
    summary="Model and data status",
    tags=["Health"],
)


# ══════════════════════════════════════════════════════════════════════════════
# CLI MODE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Flood EWS — CLI prediction")
    p.add_argument("--static",     default=STATIC_FILE,           help="Path to facility CSV")
    p.add_argument("--hazard",     default=HAZARD_FILE,           help="Path to daily hazard parquet")
    p.add_argument("--date",       default=None,                  help="YYYY-MM-DD (default: latest)")
    p.add_argument("--models_dir", default=MODELS_DIR,            help="Directory containing model artifacts")
    p.add_argument("--output",     default="predictions.parquet", help="Output parquet path")
    args = p.parse_args()

    ModelStore.load(args.models_dir, args.static, args.hazard)
    results = run_predictions(args.date)
    results.to_parquet(args.output, index=False, compression="snappy")

    log.info("Saved %d predictions -> %s", len(results), args.output)
    print("\nRisk breakdown:")
    print(results["risk_class"].value_counts().to_string())
    print("\nTop 10:")
    print(
        results[["name", "region", "facility_type", "final_score", "risk_class"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()