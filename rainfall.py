import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from tqdm import tqdm


# The name of the rainfall dataset in Google Earth Engine
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"

# The specific data column that holds the rainfall values
CHIRPS_BAND = "precipitation"

# The resolution of CHIRPS data in metres (~5.O km per pixel)
CHIRPS_SCALE_M = 5000

# If a facility has no rainfall data for a day, we fill it with 0 mm
MISSING_VALUE_FILL = 0.0



# Setting up logging to see what the script is doing as it runs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Step 1

def build_config(run_date=None, rolling_window=7, baseline_years=10):
    """
    Creates a dictionary of settings for this pipeline run.

    run_date       : The date we want features for (defaults to today)
    rolling_window : How many days to sum rainfall over (default: 7)
    baseline_years : How many past years to use as a comparison baseline
    """
    if run_date is None:
        run_date = date.today()

    # The rolling window starts this many days before run_date
    window_start = run_date - timedelta(days=rolling_window - 1)

    config = {
        "run_date":           run_date,
        "rolling_window":     rolling_window,
        "baseline_years":     baseline_years,
        "window_start":       window_start,
        # Baseline covers the 10 years before the current year
        "baseline_start_year": run_date.year - baseline_years,
        "baseline_end_year":   run_date.year - 1,
    }

    log.info("Run config: %s", config)
    return config

# Step 2 — Connecting to Google Earth Engine
# ─────────────────────────────────────────────────────────────────────────────

# This variable tracks whether we've already connected (so we don't do it twice)
already_connected_to_ee = False

def connect_to_earth_engine(project=None):
    """
    Connects to Google Earth Engine.
    If credentials are missing, it asks for log in first.
    """
    global already_connected_to_ee

    if already_connected_to_ee:
        return  # Already connected, nothing to do

    try:
        ee.Initialize(project=project)
        log.info("Connected to Earth Engine (project=%s)", project)
    except ee.EEException:
        log.warning("EE credentials missing — running ee.Authenticate()")
        ee.Authenticate()
        ee.Initialize(project=project)

    already_connected_to_ee = True


# Step 3 — Download rainfall data from GEE

def download_rainfall_for_one_facility(facility_id, lat, lon, start_date, end_date):
    """
    Downloads daily rainfall (mm) for a single facility between start_date and end_date.

    Returns a pandas Series where:
      - The index is a list of dates (one per day)
      - The values are rainfall in mm (0 if data was missing)
    """
    # Create a GEE point geometry from the facility's coordinates
    point = ee.Geometry.Point([lon, lat])

    # Filter the CHIRPS image collection to our date range
    chirps_images = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate(str(start_date), str(end_date + timedelta(days=1)))
        .select(CHIRPS_BAND)
    )

    # For each daily image, extract the rainfall value at this point
    def extract_value_from_image(image):
        sample = image.sample(
            region=point, scale=CHIRPS_SCALE_M, dropNulls=False
        ).first()
        return sample.set("system:time_start", image.get("system:time_start"))

    raw_data = chirps_images.map(extract_value_from_image).getInfo()

    # Parse the GEE response into a simple date → rainfall dictionary
    daily_rainfall = {}
    for feature in raw_data.get("features", []):
        properties = feature.get("properties", {})
        timestamp_ms = properties.get("system:time_start")
        rainfall_mm  = properties.get(CHIRPS_BAND)  # None if the pixel was masked

        if timestamp_ms is not None:
            day = pd.Timestamp(timestamp_ms, unit="ms")
            daily_rainfall[day] = float(rainfall_mm) if rainfall_mm is not None else np.nan

    # Make sure every day in the range has an entry (fill gaps with 0)
    all_days = pd.date_range(start_date, end_date, freq="D")
    rainfall_series = pd.Series(daily_rainfall, dtype=float, name=facility_id)
    rainfall_series = rainfall_series.reindex(all_days)  # fill missing days with NaN

    return rainfall_series


def download_rainfall_all_facilities(facilities_df, start_date, end_date, ee_project=None):
    """
    Downloads rainfall for every facility in the facilities table.

    Returns a wide DataFrame:
      - Rows    = dates
      - Columns = facility IDs
      - Values  = rainfall in mm
    """
    connect_to_earth_engine(ee_project)

    all_series = {}

    for _, row in tqdm(facilities_df.iterrows(), total=len(facilities_df), desc="CHIRPS pull"):
        facility_id = str(row["facility_id"])
        try:
            all_series[facility_id] = download_rainfall_for_one_facility(
                facility_id,
                float(row["lat"]),
                float(row["lon"]),
                start_date,
                end_date,
            )
        except Exception as error:
            log.warning("Download failed for facility %s: %s", facility_id, error)
            # Fill the entire period with NaN if the download failed
            all_series[facility_id] = pd.Series(
                np.nan,
                index=pd.date_range(start_date, end_date, freq="D"),
                name=facility_id,
            )

    return pd.DataFrame(all_series)



# Step 4 — Checking the data and filling the gaps


def check_and_fill_missing_data(rainfall_df, config):
    """
    Checks that we have data for every day in the rolling window.
    Any missing values are filled with 0 mm.
    """
    expected_dates = pd.date_range(config["window_start"], config["run_date"], freq="D")
    missing_dates  = expected_dates.difference(rainfall_df.index)

    if len(missing_dates) > 0:
        log.warning(
            "%d date(s) missing from GEE response — filling with zeros: %s",
            len(missing_dates), missing_dates[:3].tolist()
        )
        rainfall_df = rainfall_df.reindex(expected_dates)

    return rainfall_df.fillna(0)

# Step 5 — Download or load the historical baseline

def get_baseline_data(facilities_df, config, cache_path=None, ee_project=None):
    """
    Gets 10 years of historical rainfall data (the "baseline").
    If a cache file exists, loads from that instead of downloading again.
    """
    cache = Path(cache_path) if cache_path else None

    if cache and cache.exists():
        log.info("Loading cached baseline from %s", cache)
        baseline_df = pd.read_parquet(cache)
    else:
        start = date(config["baseline_start_year"], 1, 1)
        end   = date(config["baseline_end_year"],   12, 31)
        log.info("Downloading baseline data: %s → %s (%d years)", start, end, config["baseline_years"])

        baseline_df = download_rainfall_all_facilities(facilities_df, start, end, ee_project)
        baseline_df = baseline_df.fillna(MISSING_VALUE_FILL)

        if cache:
            baseline_df.to_parquet(cache)
            log.info("Baseline saved to %s", cache)

    return baseline_df



# Step 6 — Building historical distributions for percentile scoring
# ─────────────────────────────────────────────────────────────────────────────
def build_historical_distributions(baseline_df, rolling_window):
    """
    For each facility, computes 7-day rolling rainfall totals over the entire
    baseline period, then groups them by calendar month.

    This gives us a distribution of "what is a normal 7-day total in March?"
    which we can use to score the current week's rainfall as a percentile.

    Returns a nested dictionary:
      { facility_id: { month_number: array_of_historical_7day_totals } }
    """
    distributions = {}

    for facility_id in baseline_df.columns:
        # Calculate rolling 7-day sums for this facility over all baseline years
        rolling_sums = (
            baseline_df[facility_id]
            .rolling(window=rolling_window, min_periods=rolling_window)
            .sum()
            .dropna()
        )

        # Group the sums by month (1=Jan, 2=Feb, ..., 12=Dec)
        month_of_each_sum = rolling_sums.index.month
        monthly_distributions = {}

        for month in range(1, 13):
            values_this_month = rolling_sums.values[month_of_each_sum == month]
            monthly_distributions[month] = values_this_month

        distributions[facility_id] = monthly_distributions

    log.info("Historical distributions built for %d facilities", len(distributions))
    return distributions



# Step 7 — Calculating features for each facility

def calculate_features(rainfall_filled, historical_distributions, config):
    """
    For each facility, calculates a set of rainfall features for the run_date.

    Features calculated:
      rain_1d              Rainfall on run_date (mm)
      rain_3d_sum          Total rainfall over last 3 days (mm)
      rain_7d_sum          Total rainfall over last 7 days (mm)
      rain_percentile      How unusual is this week's rain vs historical? (0–100)
      rain_p90_flag        1 if rain_percentile >= 90 (very wet week)
      rain_p95_flag        1 if rain_percentile >= 95 (extremely wet week)
      rain_14d_sum         Total rainfall over last 14 days (mm)
      rain_anomaly         rain_7d_sum minus the typical value for this month (mm)
      rain_days_above_20mm Number of days in the last 7 days with > 20 mm rainfall
    """
    run_date_ts = pd.Timestamp(config["run_date"])
    current_month = config["run_date"].month
    rows = []

    for facility_id in rainfall_filled.columns:
        # Get the rainfall time series for this facility, sorted by date
        ts = rainfall_filled[facility_id].sort_index()

        # Skip if run_date isn't in the data
        if run_date_ts not in ts.index:
            log.warning("run_date missing for facility %s — skipping", facility_id)
            continue

        # ── Calculate each feature ───────────────────────────────────────────

        # Today's rainfall
        rain_today = float(ts.loc[run_date_ts])

        # Last 3 days (including today)
        last_3_days = ts.loc[run_date_ts - pd.Timedelta(days=2) : run_date_ts]
        rain_3d = float(last_3_days.sum()) if len(last_3_days) == 3 else np.nan

        # Last 7 days (including today)
        last_7_days = ts.loc[run_date_ts - pd.Timedelta(days=6) : run_date_ts]
        rain_7d = float(last_7_days.sum()) if len(last_7_days) == 7 else np.nan

        # Percentile: where does this 7-day total sit in historical values for this month?
        historical_values = historical_distributions.get(facility_id, {}).get(current_month, np.array([]))

        if len(historical_values) > 0 and not np.isnan(rain_7d):
            # percentileofscore: 80 means this week is wetter than 80% of historical weeks
            rain_percentile = round(percentileofscore(historical_values, rain_7d, kind="rank"), 2)
        else:
            rain_percentile = np.nan
            if not np.isnan(rain_7d):
                log.warning("No historical data for facility %s, month %s", facility_id, current_month)

        # Flag extremely wet weeks
        rain_p90_flag = int(rain_percentile >= 90) if not np.isnan(rain_percentile) else np.nan
        rain_p95_flag = int(rain_percentile >= 95) if not np.isnan(rain_percentile) else np.nan

        # Last 14 days
        last_14_days = ts.loc[run_date_ts - pd.Timedelta(days=13) : run_date_ts]
        rain_14d = float(last_14_days.sum()) if len(last_14_days) == 14 else np.nan

        # Anomaly: how much more (or less) rain than usual this month?
        if len(historical_values) > 0 and not np.isnan(rain_7d):
            typical_value = float(np.median(historical_values))
            rain_anomaly  = round(rain_7d - typical_value, 2)
        else:
            rain_anomaly = np.nan

        # Count days with heavy rainfall (> 20 mm in a single day)
        rain_heavy_days = int((last_7_days > 20).sum()) if len(last_7_days) == 7 else np.nan

        # ── Collect all features into a row ─────────────────────────────────
        rows.append({
            "facility_id":          facility_id,
            "date":                 config["run_date"],
            "rain_1d":              rain_today,
            "rain_3d_sum":          rain_3d,
            "rain_7d_sum":          rain_7d,
            "rain_percentile":      rain_percentile,
            "rain_p90_flag":        rain_p90_flag,
            "rain_p95_flag":        rain_p95_flag,
            "rain_14d_sum":         rain_14d,
            "rain_anomaly":         rain_anomaly,
            "rain_days_above_20mm": rain_heavy_days,
        })

    return pd.DataFrame(rows)



# Step 8 — Loading the facilities CSV


def load_facilities(csv_path):
    """
    Loads the facilities CSV file.
    It must have columns: facility_id, lat, lon
    """
    facilities = pd.read_csv(csv_path)

    required_columns = {"facility_id", "lat", "lon"}
    missing_columns  = required_columns - set(facilities.columns)

    if missing_columns:
        raise ValueError(f"Facilities CSV is missing these required columns: {missing_columns}")

    log.info("Loaded %d facilities from %s", len(facilities), csv_path)
    return facilities


# Step 9 — Run the full pipeline

def run_pipeline(
    facilities_path,
    run_date=None,
    rolling_window=7,
    baseline_years=10,
    output_path=None,
    baseline_cache_path=None,
    ee_project=None,
):
    """
    Runs the full pipeline from start to finish:
      1. Load facilities
      2. Download recent rainfall (rolling window)
      3. Check and fill missing data
      4. Get historical baseline data (download or load from cache)
      5. Build historical distributions by month
      6. Calculate features
      7. Save to CSV
    """

    # Build config
    config = build_config(run_date, rolling_window, baseline_years)

    # Load facilities
    facilities = load_facilities(facilities_path)

    # Step 1 — Download recent rainfall
    log.info("Step 1/4 — Downloading rolling window (%s → %s)", config["window_start"], config["run_date"])
    raw_rainfall = download_rainfall_all_facilities(
        facilities, config["window_start"], config["run_date"], ee_project
    )

    # Step 2 — Clean the data
    log.info("Step 2/4 — Checking and filling missing data")
    clean_rainfall = check_and_fill_missing_data(raw_rainfall, config)

    # Step 3 — Get baseline data
    log.info("Step 3/4 — Getting historical baseline data")
    baseline_data = get_baseline_data(facilities, config, baseline_cache_path, ee_project)
    historical_distributions = build_historical_distributions(baseline_data, rolling_window)

    # Step 4 — Calculate features
    log.info("Step 4/4 — Calculating rainfall features")
    features = calculate_features(clean_rainfall, historical_distributions, config)
    log.info("Done! Output shape: %s", features.shape)

    # Save to CSV (append to existing file if it exists)
    if output_path:
        output_file = Path(output_path)
        if output_file.exists():
            existing = pd.read_csv(output_file, parse_dates=["date"])
            # Remove any rows for the current run_date before appending
            existing = existing[existing["date"].dt.date != config["run_date"]]
            features  = pd.concat([existing, features], ignore_index=True)

        features.to_csv(output_path, index=False)
        log.info("Features saved to %s", output_path)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Download CHIRPS rainfall and calculate features for healthcare facilities"
    )
    parser.add_argument("--facilities",      required=True,  help="CSV file with facility_id, lat, lon")
    parser.add_argument("--run_date",        default=None,   help="Date to calculate features for (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--rolling_window",  type=int, default=7, help="Number of days to sum rainfall over (default: 7)")
    parser.add_argument("--baseline_years",  type=int, default=10, help="Number of past years to use as baseline (default: 10)")
    parser.add_argument("--output",          default="rainfall_features.csv", help="Output CSV filename")
    parser.add_argument("--baseline_cache",  default=None,   help="Parquet file to cache the baseline (avoids re-downloading)")
    parser.add_argument("--ee_project",      default=None,   help="Google Earth Engine project ID")
    return parser


def main():
    parser = build_argument_parser()
    args   = parser.parse_args()

    # Convert run_date string to a date object if provided
    run_date = date.fromisoformat(args.run_date) if args.run_date else None

    run_pipeline(
        facilities_path     = args.facilities,
        run_date            = run_date,
        rolling_window      = args.rolling_window,
        baseline_years      = args.baseline_years,
        output_path         = args.output,
        baseline_cache_path = args.baseline_cache,
        ee_project          = args.ee_project,
    )


if __name__ == "__main__":
    main()