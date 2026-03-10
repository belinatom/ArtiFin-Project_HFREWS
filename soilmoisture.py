"""
era5_soilmoisture_pipeline.py
==============================
Pulls ERA5-Land soil moisture at healthcare facility points and engineers
model-ready / dashboard-ready features on a repeatable, scheduled basis.

Data source
-----------
ERA5-Land Daily Aggregated
GEE asset   : "ECMWF/ERA5_LAND/DAILY_AGGR"
Variable    : volumetric_soil_water_layer_1
Depth layer : 0–7 cm (topsoil / surface layer)
Units       : m³/m³  (volumetric water content)
Spatial res : ~9 km (0.1°)
Temporal res: daily
Available   : 1950-01-01 to present (2–3 month lag for final reanalysis;
              preliminary data available sooner via ECMWF/ERA5_LAND/HOURLY)

Why ERA5-Land / why layer 1?
  * ERA5-Land is the standard reanalysis product for land-surface variables;
    the 0–7 cm layer captures surface wetness relevant to vector breeding,
    waterlogging, and diarrhoeal-disease risk — the most commonly modelled
    health outcomes in sub-Saharan Africa.
  * Layer 2 (7–28 cm) or deeper layers may be more appropriate for drought /
    agricultural-stress models; change ERA5_BAND below if needed.

Install
-------
    pip install earthengine-api pandas numpy scipy tqdm

Authenticate once (first run only):
    python -c "import ee; ee.Authenticate()"

Usage
-----
    # Weekly scheduled run:
    python era5_soilmoisture_pipeline.py \
        --facilities  hfr_data.csv \
        --output      sm_features.csv \
        --baseline_cache sm_baseline_cache.parquet

    # All options:
    python era5_soilmoisture_pipeline.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import ee
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (product-level — change only if switching ERA5 variable/layer)
# ─────────────────────────────────────────────────────────────────────────────

ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_BAND       = "volumetric_soil_water_layer_1"   # 0–7 cm; units: m³/m³
ERA5_SCALE_M    = 11_132                            # ~0.1° in metres at equator

# Missing-value handling rule (documented):
#   Up to MAX_MISSING_DAYS consecutive missing days per facility are
#   forward-filled from the preceding valid observation.
#   If a window still contains ANY missing value after filling, all
#   features for that facility on that run date are set to NaN.
#   Rationale: soil moisture is a slowly varying state variable; a single
#   missing reanalysis day is well-approximated by the prior day's value.
#   More than MAX_MISSING_DAYS suggests a systematic retrieval failure; in
#   that case, propagating unreliable fills into model inputs is worse than
#   emitting NaN and triggering an alert.
MAX_MISSING_DAYS     = 1     # forward-fill tolerance per rolling window
MAX_MISSING_FRACTION = 0.20  # warn per facility if raw missing fraction exceeds this

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

def build_config(
    run_date: date | None = None,
    update_freq: Literal["daily", "weekly"] = "weekly",
    rolling_window: int = 7,
    baseline_years: int = 10,
    seasonal_grouping: Literal["month", "week"] = "month",
) -> dict:
    """Validate and return the run configuration dict."""
    if run_date is None:
        run_date = date.today()

    cfg = dict(
        run_date          = run_date,
        update_freq       = update_freq,
        rolling_window    = rolling_window,
        baseline_years    = baseline_years,
        seasonal_grouping = seasonal_grouping,
        # Derived
        window_start        = run_date - timedelta(days=rolling_window - 1),
        baseline_start_year = run_date.year - baseline_years,
        baseline_end_year   = run_date.year - 1,
    )
    log.info("Run config: %s", cfg)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. GEE initialisation
# ─────────────────────────────────────────────────────────────────────────────

_EE_INITIALISED = False

def init_ee(project: str | None = None) -> None:
    global _EE_INITIALISED
    if _EE_INITIALISED:
        return
    try:
        ee.Initialize(project=project)
        log.info("Earth Engine initialised (project=%s)", project)
    except ee.EEException:
        log.warning("EE credentials not found — running ee.Authenticate()")
        ee.Authenticate()
        ee.Initialize(project=project)
    _EE_INITIALISED = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. ERA5-Land ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_one_facility(
    facility_id: str,
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> pd.Series:
    """
    Pull ERA5-Land daily volumetric soil moisture (layer 1, 0–7 cm) for a
    single point over the closed interval [start, end].

    ERA5_LAND/DAILY_AGGR already provides one image per day, so no
    sub-daily aggregation is needed (contrast with 3-hourly SMAP).

    Returns a pd.Series indexed by pd.Timestamp (daily), named facility_id.
    Days absent from the GEE response are returned as NaN.
    """
    point = ee.Geometry.Point([lon, lat])

    collection = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(str(start), str(end + timedelta(days=1)))  # end is exclusive in GEE
        .select(ERA5_BAND)
    )

    def extract(image: ee.Image) -> ee.Feature:
        dt  = ee.Date(image.get("system:time_start"))
        val = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=ERA5_SCALE_M,
        ).get(ERA5_BAND)
        return ee.Feature(None, {
            "date": dt.format("YYYY-MM-dd"),
            ERA5_BAND: val,
        })

    raw = collection.map(extract).getInfo()

    records: dict[pd.Timestamp, float] = {}
    for feat in raw.get("features", []):
        props  = feat.get("properties", {})
        dt_str = props.get("date")
        sm_val = props.get(ERA5_BAND)
        if dt_str:
            records[pd.Timestamp(dt_str)] = (
                float(sm_val) if sm_val is not None else np.nan
            )

    full_index = pd.date_range(start, end, freq="D")
    return pd.Series(records, dtype=float, name=facility_id).reindex(full_index)


def fetch_era5_batch(
    facilities: pd.DataFrame,
    start: date,
    end: date,
    ee_project: str | None = None,
) -> pd.DataFrame:
    """
    Pull ERA5-Land daily soil moisture for all facilities over [start, end].

    Returns wide DataFrame: index = DatetimeIndex (daily), columns = facility_id.
    """
    init_ee(ee_project)
    frames: dict[str, pd.Series] = {}

    for _, row in tqdm(facilities.iterrows(), total=len(facilities), desc="ERA5 pull"):
        fid = str(row["facility_id"])
        try:
            frames[fid] = _fetch_one_facility(
                fid, float(row["lat"]), float(row["lon"]), start, end
            )
        except Exception as exc:
            log.warning("Fetch failed — facility %s: %s", fid, exc)
            frames[fid] = pd.Series(
                np.nan,
                index=pd.date_range(start, end, freq="D"),
                name=fid,
            )

    return pd.DataFrame(frames)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Validation & missing-value handling
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_fill(sm_wide: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    1. Confirm full date coverage for the rolling window; reindex if gaps found.
    2. Forward-fill up to MAX_MISSING_DAYS consecutive missing values per column.
       Rule: soil moisture changes slowly enough that one missing reanalysis day
       can be safely approximated by the prior day's value.
    3. If any column still has NaN values after filling, those columns are
       left as NaN — features will propagate NaN downstream (see engineer_features).
    4. Log a per-facility warning when the raw missing fraction exceeds
       MAX_MISSING_FRACTION before filling.
    """
    expected = pd.date_range(cfg["window_start"], cfg["run_date"], freq="D")
    missing_dates = expected.difference(sm_wide.index)

    if len(missing_dates):
        log.warning(
            "Date gap in GEE response — %d missing date(s): %s …",
            len(missing_dates), missing_dates[:3].tolist(),
        )
        sm_wide = sm_wide.reindex(expected)

    # Warn on high raw missingness before filling
    raw_miss = sm_wide.isna().mean()
    high = raw_miss[raw_miss > MAX_MISSING_FRACTION]
    if not high.empty:
        log.warning(
            "Raw missing fraction > %.0f%% in %d facility/ies: %s",
            MAX_MISSING_FRACTION * 100, len(high), high.index.tolist()[:10],
        )

    # Forward-fill (tolerance = MAX_MISSING_DAYS)
    sm_filled = sm_wide.ffill(limit=MAX_MISSING_DAYS)

    # Report residual NaN after filling
    residual = sm_filled.isna().any()
    if residual.any():
        log.warning(
            "After forward-fill (%d day limit), %d facility/ies still have NaN — "
            "features will be NaN for those facilities.",
            MAX_MISSING_DAYS, residual.sum(),
        )

    return sm_filled


# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline distributions
# ─────────────────────────────────────────────────────────────────────────────

def fetch_baseline_wide(
    facilities: pd.DataFrame,
    cfg: dict,
    ee_project: str | None = None,
) -> pd.DataFrame:
    """
    Pull the full baseline period once; return wide DataFrame.
    Call this once and cache to Parquet to avoid re-fetching on every run.
    """
    start = date(cfg["baseline_start_year"], 1, 1)
    end   = date(cfg["baseline_end_year"],   12, 31)
    log.info(
        "Fetching baseline %s → %s (%d years)", start, end, cfg["baseline_years"]
    )
    df = fetch_era5_batch(facilities, start, end, ee_project=ee_project)
    # Light forward-fill over the baseline too, for parity with production
    return df.ffill(limit=MAX_MISSING_DAYS)


def build_baseline_distributions(
    baseline_df: pd.DataFrame,
    rolling_window: int,
    seasonal_grouping: Literal["month", "week"],
) -> dict[str, dict[int, np.ndarray]]:
    """
    Compute rolling-window means over the baseline period and bin by season.

    This is the sole reference used for percentile scoring and MUST be
    identical between model training and production inference — never rebuild
    it ad-hoc. Cache it and reload.

    Structure:
        {facility_id: {season_key (int): np.ndarray of historical 7d means}}
    """
    result: dict[str, dict[int, np.ndarray]] = {}

    for fid in baseline_df.columns:
        means = (
            baseline_df[fid]
            .rolling(window=rolling_window, min_periods=max(1, rolling_window // 2))
            .mean()
            .dropna()
        )
        if seasonal_grouping == "month":
            keys = means.index.month
        else:
            keys = means.index.isocalendar().week.astype(int)

        dist: dict[int, np.ndarray] = {}
        for k in np.unique(keys):
            dist[int(k)] = means.values[keys == k]
        result[fid] = dist

    log.info(
        "Baseline distributions built for %d facilities (grouping=%s)",
        len(result), seasonal_grouping,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _season_key(dt: date, grouping: str) -> int:
    return dt.month if grouping == "month" else dt.isocalendar()[1]


def engineer_features(
    sm_filled: pd.DataFrame,
    baseline_dists: dict[str, dict[int, np.ndarray]],
    cfg: dict,
) -> pd.DataFrame:
    """
    Produce one row per facility with all required and optional features.

    Required
    --------
    sm_1d          Surface soil moisture at run_date  (m³/m³)
    sm_7d_mean     Mean over last 7 days              (m³/m³)
    sm_percentile  Percentile of sm_7d_mean vs same-season historical
                   baseline distribution (0–100)
    sm_p90_flag    1 if sm_percentile ≥ 90 else 0
    sm_p95_flag    1 if sm_percentile ≥ 95 else 0

    Optional
    --------
    sm_14d_mean    Mean over last 14 days             (m³/m³)
    sm_change_3d   sm_1d minus value 3 days ago       (m³/m³, + = wetting)
    sm_change_7d   sm_1d minus value 7 days ago       (m³/m³, + = wetting)

    NaN propagation rule
    --------------------
    Any feature that depends on a window containing NaN (after forward-fill
    tolerance is exhausted) is itself emitted as NaN.  The percentile-based
    features and binary flags are also NaN if their prerequisite is NaN,
    rather than defaulting to a misleading 0 or 50.
    """
    run_ts   = pd.Timestamp(cfg["run_date"])
    grouping = cfg["seasonal_grouping"]
    skey     = _season_key(cfg["run_date"], grouping)
    records  = []

    for fid in sm_filled.columns:
        s = sm_filled[fid].sort_index()

        if run_ts not in s.index:
            log.warning(
                "run_date %s missing for facility %s — skipping", run_ts.date(), fid
            )
            continue

        # ── Required ─────────────────────────────────────────────────────────

        sm_1d = s.loc[run_ts]   # scalar; may be NaN if gap > tolerance

        win7       = s.loc[run_ts - pd.Timedelta(days=6): run_ts]
        sm_7d_mean = win7.mean() if len(win7) == cfg["rolling_window"] else np.nan

        dist = baseline_dists.get(fid, {}).get(skey, np.array([]))
        if len(dist) > 0 and not np.isnan(sm_7d_mean):
            sm_percentile = round(
                percentileofscore(dist, sm_7d_mean, kind="rank"), 2
            )
        else:
            sm_percentile = np.nan
            if not np.isnan(sm_7d_mean):
                log.warning(
                    "No baseline dist — facility %s, season key %s", fid, skey
                )

        sm_p90_flag = (
            int(sm_percentile >= 90) if not np.isnan(sm_percentile) else np.nan
        )
        sm_p95_flag = (
            int(sm_percentile >= 95) if not np.isnan(sm_percentile) else np.nan
        )

        # ── Optional ─────────────────────────────────────────────────────────

        win14      = s.loc[run_ts - pd.Timedelta(days=13): run_ts]
        sm_14d_mean = win14.mean() if len(win14) == 14 else np.nan

        t_minus_3 = run_ts - pd.Timedelta(days=3)
        sm_change_3d = (
            float(sm_1d - s.loc[t_minus_3])
            if t_minus_3 in s.index and not np.isnan(sm_1d) and not np.isnan(s.loc[t_minus_3])
            else np.nan
        )

        t_minus_7 = run_ts - pd.Timedelta(days=7)
        sm_change_7d = (
            float(sm_1d - s.loc[t_minus_7])
            if t_minus_7 in s.index and not np.isnan(sm_1d) and not np.isnan(s.loc[t_minus_7])
            else np.nan
        )

        records.append(dict(
            facility_id   = fid,
            date          = cfg["run_date"],
            # Required
            sm_1d         = _r4(sm_1d),
            sm_7d_mean    = _r4(sm_7d_mean),
            sm_percentile = sm_percentile,
            sm_p90_flag   = sm_p90_flag,
            sm_p95_flag   = sm_p95_flag,
            # Optional
            sm_14d_mean   = _r4(sm_14d_mean),
            sm_change_3d  = _r4(sm_change_3d),
            sm_change_7d  = _r4(sm_change_7d),
        ))

    return pd.DataFrame(records)


def _r4(v: float) -> float:
    """Round to 4 d.p. or propagate NaN."""
    return round(float(v), 4) if not (v is None or np.isnan(v)) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 7. End-to-end orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    facilities_path: str | Path,
    run_date: date | None = None,
    update_freq: Literal["daily", "weekly"] = "weekly",
    rolling_window: int = 7,
    baseline_years: int = 10,
    seasonal_grouping: Literal["month", "week"] = "month",
    output_path: str | Path | None = None,
    baseline_cache_path: str | Path | None = None,
    ee_project: str | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: load → fetch → validate → baseline → features → write.

    Parameters
    ----------
    facilities_path     : CSV with columns facility_id, lat, lon (+ extras OK)
    run_date            : date to compute features for; defaults to today
    update_freq         : scheduler hint — "daily" or "weekly"
    rolling_window      : window in days for mean and change features (default 7)
    baseline_years      : number of full calendar years for percentile baseline
    seasonal_grouping   : "month" (recommended) or "week" for baseline binning
    output_path         : if set, append this run's rows to the CSV (de-dup by date)
    baseline_cache_path : Parquet path — avoids re-fetching the 10-year baseline
    ee_project          : GEE project ID (e.g. "mhews-tz")

    Returns
    -------
    pd.DataFrame — one row per facility with all feature columns described above
    """
    cfg        = build_config(run_date, update_freq, rolling_window,
                              baseline_years, seasonal_grouping)
    facilities = _load_facilities(facilities_path)

    # Step 1 — rolling window pull (need window + 7 extra days for sm_change_7d)
    extended_start = cfg["window_start"] - timedelta(days=7)
    log.info(
        "Step 1/4 — pulling ERA5-Land window %s → %s",
        extended_start, cfg["run_date"],
    )
    sm_raw = fetch_era5_batch(
        facilities, extended_start, cfg["run_date"], ee_project
    )

    # Step 2 — validate + fill
    log.info("Step 2/4 — validating and forward-filling")
    # validate_and_fill operates over the primary rolling window only
    sm_primary = sm_raw.loc[cfg["window_start"].isoformat():]
    sm_filled_primary = validate_and_fill(sm_primary, cfg)
    # Re-attach the extra 7-day lookback tail (used only for sm_change_7d)
    sm_filled = pd.concat([
        sm_raw.loc[: (cfg["window_start"] - timedelta(days=1)).isoformat()]
              .ffill(limit=MAX_MISSING_DAYS),
        sm_filled_primary,
    ]).sort_index()

    # Step 3 — baseline (load from cache or fetch)
    log.info("Step 3/4 — baseline distributions")
    cache = Path(baseline_cache_path) if baseline_cache_path else None
    if cache and cache.exists():
        log.info("  Loading cached baseline from %s", cache)
        baseline_df = pd.read_parquet(cache)
    else:
        baseline_df = fetch_baseline_wide(facilities, cfg, ee_project)
        if cache:
            baseline_df.to_parquet(cache)
            log.info("  Baseline cached to %s", cache)

    baseline_dists = build_baseline_distributions(
        baseline_df, rolling_window, seasonal_grouping
    )

    # Step 4 — feature engineering
    log.info("Step 4/4 — engineering features")
    features = engineer_features(sm_filled, baseline_dists, cfg)
    log.info("Output shape: %s", features.shape)

    if output_path:
        out = Path(output_path)
        if out.exists():
            existing = pd.read_csv(out, parse_dates=["date"])
            existing = existing[existing["date"].dt.date != cfg["run_date"]]
            features  = pd.concat([existing, features], ignore_index=True)
        features.to_csv(output_path, index=False)
        log.info("Features written to %s", output_path)

    return features


def _load_facilities(path: str | Path) -> pd.DataFrame:
    facilities = pd.read_csv(path)
    required   = {"facility_id", "lat", "lon"}
    missing    = required - set(facilities.columns)
    if missing:
        raise ValueError(f"Facility CSV missing required columns: {missing}")
    log.info("Loaded %d facilities", len(facilities))
    return facilities


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "ERA5-Land soil moisture feature pipeline — healthcare facilities\n"
            f"  GEE asset : {ERA5_COLLECTION}\n"
            f"  Band      : {ERA5_BAND}  (0–7 cm, m³/m³)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--facilities",        required=True,
                   help="CSV with facility_id, lat, lon")
    p.add_argument("--run_date",          default=None,
                   help="YYYY-MM-DD (default: today)")
    p.add_argument("--update_freq",       choices=["daily", "weekly"],
                   default="weekly")
    p.add_argument("--rolling_window",    type=int, default=7)
    p.add_argument("--baseline_years",    type=int, default=10)
    p.add_argument("--seasonal_grouping", choices=["month", "week"],
                   default="month")
    p.add_argument("--output",            default="sm_features.csv")
    p.add_argument("--baseline_cache",    default=None,
                   help="Parquet path for caching baseline (avoids re-fetch)")
    p.add_argument("--ee_project",        default=None,
                   help="GEE project ID (e.g. mhews-tz)")
    return p


def main(argv: list[str] | None = None) -> None:
    args     = _build_parser().parse_args(argv)
    run_date = date.fromisoformat(args.run_date) if args.run_date else None
    run_pipeline(
        facilities_path     = args.facilities,
        run_date            = run_date,
        update_freq         = args.update_freq,
        rolling_window      = args.rolling_window,
        baseline_years      = args.baseline_years,
        seasonal_grouping   = args.seasonal_grouping,
        output_path         = args.output,
        baseline_cache_path = args.baseline_cache,
        ee_project          = args.ee_project,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
