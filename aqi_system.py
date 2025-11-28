import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
import requests
import joblib


# CONFIG (ENV-based, no hard-coded paths)
import os

API_KEY = os.environ.get("WAQI_API_KEY", "")     
PROJECT_BASE = os.environ.get("PROJECT_BASE", ".")   

# Paths relative to project root
STATIONS_PATH = os.path.join(PROJECT_BASE, "Final_Station.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "CatBoost_Optimized.pkl"))
FEATURE_PATH = os.environ.get("FEATURE_PATH", os.path.join("models", "feature_cols_new.pkl"))
HISTORICAL_UPLOAD_PATH = os.path.join(PROJECT_BASE, "data", "Preprocessed_Data_Final.xls")
HISTORY_FILE = os.path.join(PROJECT_BASE, "hourly_history.csv")
FORECAST_DIR = os.path.join(PROJECT_BASE, "forecast")

SLEEP_BETWEEN = 0.5

# Ensure forecast folder exists
os.makedirs(FORECAST_DIR, exist_ok=True)



# Helpers 
def load_stations(path=STATIONS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stations file not found: {path}")
    df = pd.read_csv(path)
    required = {"StationId", "StationName", "City", "waqi_id", "StationId_enc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stations CSV missing columns: {missing}")
    return df


def fetch_live_waqi(waqi_id):
    """
    Fetch WAQI feed for a station and return a normalized live dict.
    This will try to extract all commonly available iaqi fields and map them to the
    feature names your model expects:
      - PM2.5, NO2, CO, SO2, O3
      - Temperature, RelativeHumidity, WindSpeed
      - Pressure (p), DewPoint (dew), WindDirection (wd)
    Missing values -> 0 (you can change fallback to np.nan if preferred).
    Returns (live_dict, status_str)
    """
    url = f"https://api.waqi.info/feed/@{waqi_id}/?token={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"error: {e}"

    if data.get("status") != "ok":
        return None, f"waqi_status_{data.get('status')}"

    iaqi = data["data"].get("iaqi", {}) or {}

    # helper to safely get nested numeric v value (and coerce to float)
    def g(key):
        try:
            v = iaqi.get(key, {})
            return float(v.get("v")) if isinstance(v, dict) and "v" in v else float(v) if isinstance(v, (int, float)) else 0.0
        except Exception:
            return 0.0

    # WAQI sometimes uses slightly different keys; check common variants
    # Primary keys we expect: 'pm25','no2','co','so2','o3','t','h','w','p','dew','wd'
    live = {
        "PM2.5": g("pm25"),
        "NO2": g("no2"),
        "CO": g("co"),
        "SO2": g("so2"),
        "O3": g("o3"),
        "Temperature": g("t"),            # WAQI 't' is temp
        "RelativeHumidity": g("h"),       # WAQI 'h' is humidity
        "WindSpeed": g("w"),              # WAQI 'w' is wind speed
        # additional fields (map to your feature_list names)
        "Pressure": g("p") or g("pressure"),
        "DewPoint": g("dew") or g("dewpoint"),
        "WindDirection": g("wd") or g("winddir") or g("wind_direction"),
    }

    # Some feeds embed values under different keys (rare) — attempt a second pass
    # if any of the additional keys are zero, try scanning iaqi keys for close matches.
    if live["Pressure"] == 0.0 or live["DewPoint"] == 0.0 or live["WindDirection"] == 0.0:
        for k, v in iaqi.items():
            if not isinstance(v, dict):
                continue
            key_lower = k.lower()
            val = v.get("v", None)
            if val is None:
                continue
            try:
                val = float(val)
            except Exception:
                continue
            if live["Pressure"] == 0.0 and ("p" in key_lower or "press" in key_lower):
                live["Pressure"] = val
            if live["DewPoint"] == 0.0 and ("dew" in key_lower):
                live["DewPoint"] = val
            if live["WindDirection"] == 0.0 and ("wd" in key_lower or "dir" in key_lower):
                live["WindDirection"] = val

    return live, "ok"



def ensure_feature_list(features_path=FEATURE_PATH, fallback_hist=HISTORICAL_UPLOAD_PATH):
    """
    Ensure JSON feature list exists. If not, try to build from historical file columns.
    Feature list = all columns in historical df minus ["AQI","StationId","Datetime","Unnamed: 0"].
    """
    if os.path.exists(features_path):
        feature_list = joblib.load(FEATURE_PATH)
        return feature_list

    # Try to build from historical upload
    if os.path.exists(fallback_hist):
        print("nowcast feature JSON not found — building from historical file:", fallback_hist)
        try:
            hist = pd.read_csv(fallback_hist)
        except Exception:
            hist = pd.read_excel(fallback_hist)
        cols = [c for c in hist.columns if c not in ["AQI", "StationId", "Datetime", "Unnamed: 0"]]
        # keep only numeric/time/encoded features (exclude object columns if any)
        numeric_cols = [c for c in cols if hist[c].dtype in [np.float64, np.int64, np.int32, np.float32]]
        # but also keep engineered time columns if present
        # final list: numeric_cols + any known time/one-hot columns that exist
        extras = [c for c in cols if c not in numeric_cols]
        features = numeric_cols + extras
        with open(features_path, "w") as f:
            json.dump(features, f)
        print(f"Feature JSON created at {features_path} with {len(features)} features.")
        return features

    raise FileNotFoundError(
        f"Feature JSON not found ({features_path}) and historical file not available ({fallback_hist})"
    )


def build_feature_row(live_dict, station_enc, feature_list, dt=None, history_path=HISTORY_FILE):
    """
    Build a single-row feature DataFrame matching feature_list.
    Includes:
      - Raw WAQI pollutants/weather (PM2.5, NO2, CO, SO2, O3, Temp, RH, WindSpeed, Pressure, DewPoint, WindDirection)
      - Time features (Hour_sin, Hour_cos, Month_sin, Month_cos, DOW one-hot, Season one-hot)
      - Engineered features (Pollution_Load, PM_Ratio, Temp*Humidity, Wind_Inv)
      - Lags: PM25_lag1, lag3, lag6, lag24; NO2_lag1, O3_lag1
      - Rolling means: roll3, roll6, roll12, roll24
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime

    if dt is None:
        dt = datetime.now()

    # Base row 
    row = pd.DataFrame([[0] * len(feature_list)], columns=feature_list)

    # Fill direct live features 
    for col, val in live_dict.items():
        if col in row.columns:
            row[col] = float(val)

    # Load station history (needed for lags/rollings)
    try:
        hist = pd.read_csv(history_path)
        hist["Datetime"] = pd.to_datetime(hist["Datetime"])
        hist = hist[hist["StationId_enc"] == station_enc] if "StationId_enc" in hist.columns else hist
        hist = hist.sort_values("Datetime")
    except:
        hist = pd.DataFrame()

    # Helper to get latest value of a column
    def get_latest(col):
        try:
            return float(hist[col].iloc[-1])
        except:
            return float(live_dict.get(col, 0))

    # PM2.5 Lag Features 
    def get_lag(series, lag_hours):
        try:
            return float(series.iloc[-lag_hours])
        except:
            return float(get_latest("PM2.5"))

    if not hist.empty and "PM2.5" in hist.columns:
        pm25_series = hist["PM2.5"]

        # PM25 Lags
        if "PM25_lag1" in row.columns:
            row["PM25_lag1"] = get_lag(pm25_series, 1)
        if "PM25_lag3" in row.columns:
            row["PM25_lag3"] = get_lag(pm25_series, 3)
        if "PM25_lag6" in row.columns:
            row["PM25_lag6"] = get_lag(pm25_series, 6)
        if "PM25_lag24" in row.columns:
            row["PM25_lag24"] = get_lag(pm25_series, 24)

        # Rolling means
        def roll(series, w):
            if len(series) >= w:
                return float(series.tail(w).mean())
            else:
                return float(series.mean())

        if "PM25_roll3" in row.columns:
            row["PM25_roll3"] = roll(pm25_series, 3)
        if "PM25_roll6" in row.columns:
            row["PM25_roll6"] = roll(pm25_series, 6)
        if "PM25_roll12" in row.columns:
            row["PM25_roll12"] = roll(pm25_series, 12)
        if "PM25_roll24" in row.columns:
            row["PM25_roll24"] = roll(pm25_series, 24)

    # NO2 & O3 Lag 1 
    if not hist.empty:
        if "NO2_lag1" in row.columns:
            try:
                row["NO2_lag1"] = float(hist["NO2"].iloc[-1])
            except:
                row["NO2_lag1"] = float(live_dict.get("NO2", 0))

        if "O3_lag1" in row.columns:
            try:
                row["O3_lag1"] = float(hist["O3"].iloc[-1])
            except:
                row["O3_lag1"] = float(live_dict.get("O3", 0))

    # Time based features 
    # Cyclic hour and month
    if "Hour_sin" in row.columns:
        row["Hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
    if "Hour_cos" in row.columns:
        row["Hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
    if "Month_sin" in row.columns:
        row["Month_sin"] = np.sin(2 * np.pi * dt.month / 12)
    if "Month_cos" in row.columns:
        row["Month_cos"] = np.cos(2 * np.pi * dt.month / 12)

    # Hour, DayOfWeek, Month, Quarter, Year, DayOfYear, IsWeekend
    if "Hour" in row.columns: row["Hour"] = dt.hour
    if "DayOfWeek" in row.columns: row["DayOfWeek"] = dt.weekday()
    if "IsWeekend" in row.columns: row["IsWeekend"] = int(dt.weekday() >= 5)
    if "Month" in row.columns: row["Month"] = dt.month
    if "Quarter" in row.columns: row["Quarter"] = (dt.month - 1) // 3 + 1
    if "DayOfYear" in row.columns: row["DayOfYear"] = dt.timetuple().tm_yday
    if "Year" in row.columns: row["Year"] = dt.year

    # Day-of-week one-hot
    dow = dt.weekday()
    dow_map = ["Day_Monday", "Day_Tuesday", "Day_Wednesday", "Day_Thursday",
               "Day_Friday", "Day_Saturday", "Day_Sunday"]
    for i, c in enumerate(dow_map):
        if c in row.columns: row[c] = 1 if i == dow else 0

    # Season one-hot
    if "Season_Winter" in row.columns:
        row["Season_Winter"] = 1 if dt.month in [12, 1, 2] else 0
    if "Season_Summer" in row.columns:
        row["Season_Summer"] = 1 if dt.month in [4, 5, 6] else 0
    if "Season_PostMonsoon" in row.columns:
        row["Season_PostMonsoon"] = 1 if dt.month in [10, 11] else 0

    # Engineered interactions
    pm25 = float(live_dict.get("PM2.5", 0))
    no2 = float(live_dict.get("NO2", 0))
    o3 = float(live_dict.get("O3", 0))
    temp = float(live_dict.get("Temperature", 0))
    rh = float(live_dict.get("RelativeHumidity", 0))
    ws = float(live_dict.get("WindSpeed", 0))

    if "Pollution_Load" in row.columns:
        row["Pollution_Load"] = pm25 + no2 + live_dict.get("CO", 0) + live_dict.get("SO2", 0) + o3

    if "PM_Ratio" in row.columns:
        row["PM_Ratio"] = pm25 / (no2 + o3 + 1e-6)

    if "Temp_Humidity_Interaction" in row.columns:
        row["Temp_Humidity_Interaction"] = temp * rh

    if "Wind_Inv" in row.columns:
        row["Wind_Inv"] = 1.0 / (ws + 1e-6)

    # Station Encoding 
    if "StationId_enc" in row.columns:
        row["StationId_enc"] = station_enc

    return row


def save_hourly_history(entry, history_path=HISTORY_FILE):
    """
    Append a new hourly entry to history CSV.
    Ensures:
      - Consistent columns
      - New columns auto-added if needed
      - Sorted by datetime
      - No duplicate timestamps per station
    """
    import pandas as pd
    import numpy as np
    import os

    df_entry = pd.DataFrame([entry])

    # If file does not exist → create it with this row
    if not os.path.exists(history_path):
        df_entry.to_csv(history_path, index=False)
        return

    # Load existing history
    try:
        hist = pd.read_csv(history_path)
    except Exception:
        hist = pd.DataFrame()

    # Ensure Datetime is datetime
    try:
        df_entry["Datetime"] = pd.to_datetime(df_entry["Datetime"])
        if "Datetime" in hist.columns:
            hist["Datetime"] = pd.to_datetime(hist["Datetime"])
    except:
        pass

    # Ensure all columns exist in both 
    all_cols = sorted(list(set(hist.columns).union(set(df_entry.columns))))

    for col in all_cols:
        if col not in hist.columns:
            hist[col] = np.nan
        if col not in df_entry.columns:
            df_entry[col] = np.nan

    hist = hist[all_cols]
    df_entry = df_entry[all_cols]

    # Remove duplicates for SAME station + datetime 
    if "StationId" in all_cols and "Datetime" in all_cols:
        before_len = len(hist)
        hist = hist[~(
            (hist["StationId"] == df_entry["StationId"].iloc[0]) &
            (hist["Datetime"] == df_entry["Datetime"].iloc[0])
        )]
        # (optional) print(f"Removed duplicates: {before_len-len(hist)}")

    # Append new entry 
    hist = pd.concat([hist, df_entry], ignore_index=True)

    # Sort by datetime (important for lags/rolling) 
    if "Datetime" in hist.columns:
        hist = hist.sort_values("Datetime")

    # Save 
    hist.to_csv(history_path, index=False)



def forecast_24h_recursive(aqi_now, live_dict, model, feature_list, station_enc=0, seed_dt=None):
    """
    24-hour forecast using static live pollutant/weather features.
    Why static?
      - Your model uses PM2.5 lag/rolling from HISTORY, NOT future predictions.
      - For forecasting, WAQI future values are unknown.
    So:
      - We only vary datetime-dependent features (hour, DOW, month, season, etc.)
      - Pollutants/weather remain last-known values (live_dict)
      - Lags/rollings remain based on HISTORY (computed inside build_feature_row)
    """
    from datetime import datetime, timedelta
    import pandas as pd

    if seed_dt is None:
        seed_dt = datetime.now()

    out = []

    for h in range(1, 25):
        forecast_dt = seed_dt + timedelta(hours=h)

        # Build feature row for this future hour
        row = build_feature_row(
            live_dict=live_dict,
            station_enc=station_enc,
            feature_list=feature_list,
            dt=forecast_dt
        )

        # Predict
        try:
            pred = float(model.predict(row)[0])
        except Exception as e:
            pred = float("nan")

        out.append({"Datetime": forecast_dt, "AQI": pred})

    return pd.DataFrame(out)



# Main pipeline 
def run_pipeline():
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
    import time

    # 1) Load stations
    stations = load_stations(STATIONS_PATH)

    # 2) Load feature list (47 features)
    features = ensure_feature_list(FEATURE_PATH, HISTORICAL_UPLOAD_PATH)

    # 3) Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    all_now = []

    # 4) Iterate through each station
    for idx, s in stations.iterrows():
        st_id = s["StationId"]
        waqi_id = s["waqi_id"]
        st_enc = s["StationId_enc"]

        # Fetch live WAQI data 
        live, status = fetch_live_waqi(waqi_id)
        if live is None:
            print(f"[{st_id}] WAQI fetch failed: {status}")
            time.sleep(SLEEP_BETWEEN)
            continue

        # Build feature row for NOW 
        row = build_feature_row(
            live_dict=live,
            station_enc=st_enc,
            feature_list=features,
            dt=datetime.now()
        )

        # Predict current AQI
        try:
            aqi_now = float(model.predict(row)[0])
        except Exception as e:
            print(f"[{st_id}] Model prediction failed: {e}")
            time.sleep(SLEEP_BETWEEN)
            continue

        # Save hourly history entry 
        hist_entry = {
            "StationId": st_id,
            "StationName": s.get("StationName", ""),
            "City": s.get("City", ""),
            "StationId_enc": st_enc,
            "Datetime": datetime.now().isoformat(),

            # model target
            "AQI": aqi_now,

            # pollutants
            "PM2.5": live.get("PM2.5"),
            "NO2": live.get("NO2"),
            "CO": live.get("CO"),
            "SO2": live.get("SO2"),
            "O3": live.get("O3"),

            # weather
            "Temperature": live.get("Temperature"),
            "RelativeHumidity": live.get("RelativeHumidity"),
            "WindSpeed": live.get("WindSpeed"),
            "Pressure": live.get("Pressure"),
            "DewPoint": live.get("DewPoint"),
            "WindDirection": live.get("WindDirection")
        }

        save_hourly_history(hist_entry, HISTORY_FILE)

        # 24-hour forecast 
        forecast_df = forecast_24h_recursive(
            aqi_now=aqi_now,
            live_dict=live,
            model=model,
            feature_list=features,
            station_enc=st_enc,
            seed_dt=datetime.now()
        )

        # Save forecast output
        out_csv = os.path.join(FORECAST_DIR, f"forecast_24h_{st_id}.csv")
        forecast_df.to_csv(out_csv, index=False)
        print(f"[{st_id}] AQI Now: {aqi_now:.1f} | Forecast saved → {out_csv}")

        # Add to nowcast snapshot list
        all_now.append({
            "StationId": st_id,
            "StationName": s.get("StationName", ""),
            "City": s.get("City", ""),
            "AQI_Now": aqi_now,
            "Datetime": datetime.now().isoformat()
        })

        time.sleep(SLEEP_BETWEEN)

    # Save snapshot of all predictions 
    if all_now:
        df_now = pd.DataFrame(all_now)
        snap_dir = os.path.join(PROJECT_BASE, "nowcast_snapshot")
        os.makedirs(snap_dir, exist_ok=True)

        snapshot_path = os.path.join(
            snap_dir,
            f"nowcast_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_now.to_csv(snapshot_path, index=False)
        print("Snapshot saved →", snapshot_path)


if __name__ == "__main__":
    print("Starting AQI pipeline: Live Nowcast + 24h Forecast")
    run_pipeline()
    print("Pipeline complete. History updated at:", HISTORY_FILE)
