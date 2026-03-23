"""
Flood EWS — Prediction API
Run:  uvicorn predict:app --reload
Docs: http://localhost:8000/docs
"""

import logging
import math
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "Exp2_IF_AllFeatures.pkl"
STATIC_FILE = BASE_DIR / "hfr_data.csv"
HAZARD_FILE = BASE_DIR / "daily_hazard.parquet"

pipeline    = None
model_feats = None
static_df   = None
hazard_df   = None


def load_all():
    global pipeline, model_feats, static_df, hazard_df

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not STATIC_FILE.exists():
        raise FileNotFoundError(f"Static data not found: {STATIC_FILE}")
    if not HAZARD_FILE.exists():
        raise FileNotFoundError(f"Hazard data not found: {HAZARD_FILE}")

    pipeline = joblib.load(MODEL_PATH)
    scaler   = pipeline.named_steps["scaler"]
    model_feats = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") \
                  else [f"feature_{i}" for i in range(scaler.n_features_in_)]
    log.info("Loaded model | %d features: %s", len(model_feats), model_feats)

    static_df = pd.read_csv(STATIC_FILE, encoding="latin1")
    if "facility_id" not in static_df.columns:
        static_df.insert(0, "facility_id", range(len(static_df)))
    static_df["facility_id"] = static_df["facility_id"].astype(int)

    hazard_df = pd.read_parquet(HAZARD_FILE)
    hazard_df["date"] = pd.to_datetime(hazard_df["date"])
    log.info("Facilities: %d | Hazard dates: %d", len(static_df), hazard_df["date"].nunique())


def score(df: pd.DataFrame) -> np.ndarray:
    # reindex to exactly the model features, then fill ALL NaN (including
    # existing ones from the source data, not just absent columns)
    X = df.reindex(columns=model_feats, fill_value=0.0).astype(float).fillna(0.0)
    Xs  = pipeline.named_steps["scaler"].transform(X)
    raw = -pipeline.named_steps["detector"].score_samples(Xs)
    mn, mx = raw.min(), raw.max()
    if mx - mn < 1e-9:
        return np.full(len(raw), 0.5)
    return np.clip((raw - mn) / (mx - mn), 0, 1)


def label(s: float) -> str:
    if s >= 0.65: return "High Risk"
    if s >= 0.35: return "Moderate Risk"
    return "No Risk"


def _clean(v):
    """Replace NaN/inf with None for JSON compliance."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ── Pydantic schema — matches the 17 features the model was trained on ────────
class ManualInput(BaseModel):
    rain_1d:                  float = Field(0.0, description="1-day rainfall (mm)")
    rain_3d_sum:              float = Field(0.0, description="3-day cumulative rainfall (mm)")
    rain_7d_sum:              float = Field(0.0, description="7-day cumulative rainfall (mm)")
    rain_percentile:          float = Field(0.5, description="Rainfall percentile 0-1")
    rain_anomaly:             float = Field(0.0, description="Rainfall anomaly (mm)")
    river_discharge_m3s:      float = Field(0.0, description="River discharge (m3/s)")
    discharge_percentile:     float = Field(0.5, description="Discharge percentile 0-1")
    swi_1d:                   float = Field(0.5, description="1-day soil water index 0-1")
    swi_7d_mean:              float = Field(0.5, description="7-day mean SWI 0-1")
    swi_3d_trend:             float = Field(0.0, description="3-day SWI trend")
    swi_percentile:           float = Field(0.5, description="SWI percentile 0-1")
    runoff_1d:                float = Field(0.0, description="1-day runoff (mm)")
    runoff_3d_sum:            float = Field(0.0, description="3-day cumulative runoff (mm)")
    runoff_7d_sum:            float = Field(0.0, description="7-day cumulative runoff (mm)")
    runoff_percentile:        float = Field(0.5, description="Runoff percentile 0-1")
    water_accumulation_index: float = Field(0.5, description="Terrain water accumulation 0-1")
    dist_to_river_km:         float = Field(5.0, description="Distance to nearest river (km)")


# ── App with lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield

app = FastAPI(title="Flood EWS", version="4.2", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # Let FastAPI handle HTTPException normally (404, 422, etc.)
    if isinstance(exc, HTTPException):
        raise exc
    tb = traceback.format_exc()
    log.error("Unhandled exception:\n%s", tb)
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict/manual", summary="Score one facility from manual feature input")
def predict_manual(payload: ManualInput):
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    row  = pd.DataFrame([data])
    s    = float(score(row)[0])
    return {"final_score": round(s, 4), "risk_class": label(s)}


@app.get("/predict", summary="Score all facilities for a date")
def predict_all(date: Optional[str] = None):
    """Score every facility. Leave **date** blank for the latest available date (YYYY-MM-DD)."""
    dt    = pd.to_datetime(date) if date else hazard_df["date"].max()
    daily = hazard_df[hazard_df["date"] == dt]
    if daily.empty:
        raise HTTPException(status_code=404, detail=f"No hazard data for {dt.date()}")

    df = daily.merge(static_df, on="facility_id", how="left")
    df["final_score"] = score(df)
    df["risk_class"]  = df["final_score"].apply(label)
    df["date"]        = dt.date().isoformat()

    keep = ["facility_id", "name", "region", "facility_type", "date", "final_score", "risk_class"]
    out  = df[[c for c in keep if c in df.columns]].sort_values("final_score", ascending=False).copy()
    out["final_score"] = out["final_score"].round(4)

    return [{k: _clean(v) for k, v in row.items()} for row in out.to_dict("records")]


@app.get("/risk/summary", summary="Risk count breakdown for a date")
def risk_summary(date: Optional[str] = None):
    """High / Moderate / No Risk counts. Leave **date** blank for latest (YYYY-MM-DD)."""
    rows   = predict_all(date)
    counts = {"High Risk": 0, "Moderate Risk": 0, "No Risk": 0}
    for r in rows:
        counts[r["risk_class"]] += 1
    return {
        "date":          rows[0]["date"] if rows else None,
        "total":         len(rows),
        "high_risk":     counts["High Risk"],
        "moderate_risk": counts["Moderate Risk"],
        "no_risk":       counts["No Risk"],
        "mean_score":    round(sum(r["final_score"] for r in rows) / len(rows), 4) if rows else 0,
    }


@app.get("/health", summary="Model and data status")
def health():
    return {
        "model":          MODEL_PATH.name,
        "model_features": model_feats,
        "n_features":     len(model_feats) if model_feats else 0,
        "n_facilities":   len(static_df) if static_df is not None else 0,
        "date_min":       hazard_df["date"].min().date().isoformat() if hazard_df is not None else None,
        "date_max":       hazard_df["date"].max().date().isoformat() if hazard_df is not None else None,
    }
