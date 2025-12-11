import os
import time
import json
from datetime imp)ort datetime, timedelta

import pandas as pd
import numpy as np
import requests
import joblib

# gspread + oauth
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    gspread = None
    ServiceAccountCredentials = None

# CONFIG (env)
API_KEY = os.environ.get("WAQI_API_KEY", "")
PROJECT_BASE = os.environ.get("PROJECT_BASE", ".")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "CatBoost_Optimized.pkl"))
FEATURE_PATH = os.environ.get("FEATURE_PATH", os.path.join("models", "feature_cols_new.pkl"))
HISTORICAL_UPLOAD_PATH = os.path.join(PROJECT_BASE, "data", "Preprocessed_Data_Final.xls")
HISTORY_FILE = os.path.join(PROJECT_BASE, "hourly_history.csv")
FORECAST_DIR = os.path.join(PROJECT_BASE, "forecast")
SLEEP_BETWEEN = float(os.environ.get("SLEEP_BETWEEN", 0.5))

# Path where the service account JSON secret is mounted (Render secret)
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON_PATH", "/etc/secrets/SERVICE_ACCOUNT_JSON")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")

os.makedirs(FORECAST_DIR, exist_ok=True)


#### HELPERS ####

def load_stations(path=os.path.join(PROJECT_BASE, "Final_Station.csv")):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stations file not found: {path}")
    df = pd.read_csv(path)
    required = {"StationId", "StationName", "City", "waqi_id", "StationId_enc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stations CSV missing columns: {missing}")
    return df


def fetch_live_waqi(waqi_id):
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

    def g(key):
        try:
            v = iaqi.get(key, {})
            return float(v.get("v")) if isinstance(v, dict) and "v" in v else float(v) if isinstance(v, (int, float)) else 0.0
        except Exception:
            return 0.0

    live = {
        "PM2.5": g("pm25"),
        "NO2": g("no2"),
        "CO": g("co"),
        "SO2": g("so2"),
        "O3": g("o3"),
        "Temperature": g("t"),
        "RelativeHumidity": g("h"),
        "WindSpeed": g("w"),
        "Pressure": g("p") or g("pressure"),
        "DewPoint": g("dew") or g("dewpoint"),
        "WindDirection": g("wd") or g("winddir") or g("wind_direction"),
    }

    # fallback scan
    if live["Pressure"] == 0.0 or live["DewPoint"] == 0.0 or live["WindDirection"] == 0.0:
        for k, v in iaqi.items():
            if not isinstance(v, dict): continue
            key_lower = k.lower()
            val = v.get("v", None)
            if val is None: continue
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
    if os.path.exists(features_path):
        # support both joblib and json (older versions)
        try:
            return joblib.load(features_path)
        except Exception:
            with open(features_path, "r") as f:
                return json.load(f)
    if os.path.exists(fallback_hist):
        try:
            hist = pd.read_csv(fallback_hist)
        except Exception:
            hist = pd.read_excel(fallback_hist)
        cols = [c for c in hist.columns if c not in ["AQI", "StationId", "Datetime", "Unnamed: 0"]]
        numeric_cols = [c for c in cols if hist[c].dtype in [np.float64, np.int64, np.int32, np.float32]]
        extras = [c for c in cols if c not in numeric_cols]
        features = numeric_cols + extras
        # save as json for future runs
        with open(features_path, "w") as f:
            json.dump(features, f)
        return features
    raise FileNotFoundError("Feature JSON not found and historical file not available.")


def build_feature_row(live_dict, station_enc, feature_list, dt=None, history_path=HISTORY_FILE):
    import pandas as pd, numpy as np
    from datetime import datetime
    if dt is None:
        dt = datetime.now()
    row = pd.DataFrame([[0] * len(feature_list)], columns=feature_list)
    for col, val in live_dict.items():
        if col in row.columns:
            row[col] = float(val)
    # load history (for lags)
    try:
        hist = pd.read_csv(history_path)
        hist["Datetime"] = pd.to_datetime(hist["Datetime"])
        hist = hist[hist["StationId_enc"] == station_enc] if "StationId_enc" in hist.columns else hist
        hist = hist.sort_values("Datetime")
    except:
        hist = pd.DataFrame()

    def get_latest(col):
        try:
            return float(hist[col].iloc[-1])
        except:
            return float(live_dict.get(col, 0))

    def get_lag(series, lag_hours):
        try:
            return float(series.iloc[-lag_hours])
        except:
            return float(get_latest("PM2.5"))

    if not hist.empty and "PM2.5" in hist.columns:
        pm25_series = hist["PM2.5"]
        if "PM25_lag1" in row.columns: row["PM25_lag1"] = get_lag(pm25_series, 1)
        if "PM25_lag3" in row.columns: row["PM25_lag3"] = get_lag(pm25_series, 3)
        if "PM25_lag6" in row.columns: row["PM25_lag6"] = get_lag(pm25_series, 6)
        if "PM25_lag24" in row.columns: row["PM25_lag24"] = get_lag(pm25_series, 24)
        def roll(series, w):
            if len(series) >= w: return float(series.tail(w).mean())
            return float(series.mean())
        if "PM25_roll3" in row.columns: row["PM25_roll3"] = roll(pm25_series, 3)
        if "PM25_roll6" in row.columns: row["PM25_roll6"] = roll(pm25_series, 6)
        if "PM25_roll12" in row.columns: row["PM25_roll12"] = roll(pm25_series, 12)
        if "PM25_roll24" in row.columns: row["PM25_roll24"] = roll(pm25_series, 24)

    if not hist.empty:
        if "NO2_lag1" in row.columns:
            try: row["NO2_lag1"] = float(hist["NO2"].iloc[-1])
            except: row["NO2_lag1"] = float(live_dict.get("NO2", 0))
        if "O3_lag1" in row.columns:
            try: row["O3_lag1"] = float(hist["O3"].iloc[-1])
            except: row["O3_lag1"] = float(live_dict.get("O3", 0))

    # time features
    import numpy as np
    if "Hour_sin" in row.columns: row["Hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
    if "Hour_cos" in row.columns: row["Hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
    if "Month_sin" in row.columns: row["Month_sin"] = np.sin(2 * np.pi * dt.month / 12)
    if "Month_cos" in row.columns: row["Month_cos"] = np.cos(2 * np.pi * dt.month / 12)
    if "Hour" in row.columns: row["Hour"] = dt.hour
    if "DayOfWeek" in row.columns: row["DayOfWeek"] = dt.weekday()
    if "IsWeekend" in row.columns: row["IsWeekend"] = int(dt.weekday() >= 5)
    if "Month" in row.columns: row["Month"] = dt.month
    if "Quarter" in row.columns: row["Quarter"] = (dt.month - 1) // 3 + 1
    if "DayOfYear" in row.columns: row["DayOfYear"] = dt.timetuple().tm_yday
    if "Year" in row.columns: row["Year"] = dt.year

    dow = dt.weekday()
    dow_map = ["Day_Monday", "Day_Tuesday", "Day_Wednesday", "Day_Thursday",
               "Day_Friday", "Day_Saturday", "Day_Sunday"]
    for i, c in enumerate(dow_map):
        if c in row.columns: row[c] = 1 if i == dow else 0

    if "Season_Winter" in row.columns: row["Season_Winter"] = 1 if dt.month in [12,1,2] else 0
    if "Season_Summer" in row.columns: row["Season_Summer"] = 1 if dt.month in [4,5,6] else 0
    if "Season_PostMonsoon" in row.columns: row["Season_PostMonsoon"] = 1 if dt.month in [10,11] else 0

    pm25 = float(live_dict.get("PM2.5", 0))
    no2 = float(live_dict.get("NO2", 0))
    o3 = float(live_dict.get("O3", 0))
    temp = float(live_dict.get("Temperature", 0))
    rh = float(live_dict.get("RelativeHumidity", 0))
    ws = float(live_dict.get("WindSpeed", 0))

    if "Pollution_Load" in row.columns: row["Pollution_Load"] = pm25 + no2 + live_dict.get("CO", 0) + live_dict.get("SO2", 0) + o3
    if "PM_Ratio" in row.columns: row["PM_Ratio"] = pm25 / (no2 + o3 + 1e-6)
    if "Temp_Humidity_Interaction" in row.columns: row["Temp_Humidity_Interaction"] = temp * rh
    if "Wind_Inv" in row.columns: row["Wind_Inv"] = 1.0 / (ws + 1e-6)
    if "StationId_enc" in row.columns: row["StationId_enc"] = station_enc

    return row


def save_hourly_history(entry, history_path=HISTORY_FILE):
    df_entry = pd.DataFrame([entry])
    if not os.path.exists(history_path):
        df_entry.to_csv(history_path, index=False)
        return
    try:
        hist = pd.read_csv(history_path)
    except Exception:
        hist = pd.DataFrame()
    try:
        df_entry["Datetime"] = pd.to_datetime(df_entry["Datetime"])
        if "Datetime" in hist.columns:
            hist["Datetime"] = pd.to_datetime(hist["Datetime"])
    except:
        pass
    all_cols = sorted(list(set(hist.columns).union(set(df_entry.columns))))
    for col in all_cols:
        if col not in hist.columns: hist[col] = np.nan
        if col not in df_entry.columns: df_entry[col] = np.nan
    hist = hist[all_cols]
    df_entry = df_entry[all_cols]
    if "StationId" in all_cols and "Datetime" in all_cols:
        hist = hist[~(
            (hist["StationId"] == df_entry["StationId"].iloc[0]) &
            (hist["Datetime"] == df_entry["Datetime"].iloc[0])
        )]
    hist = pd.concat([hist, df_entry], ignore_index=True)
    if "Datetime" in hist.columns:
        hist = hist.sort_values("Datetime")
    hist.to_csv(history_path, index=False)


def forecast_24h_recursive(aqi_now, live_dict, model, feature_list, station_enc=0, seed_dt=None):
    if seed_dt is None:
        seed_dt = datetime.now()
    out = []
    for h in range(1, 25):
        forecast_dt = seed_dt + timedelta(hours=h)
        row = build_feature_row(live_dict=live_dict, station_enc=station_enc, feature_list=feature_list, dt=forecast_dt)
        try:
            pred = float(model.predict(row)[0])
        except Exception:
            pred = float("nan")
        out.append({"Datetime": forecast_dt.isoformat(), "AQI": pred})
    return pd.DataFrame(out)


#### GOOGLE SHEETS APPEND-ONLY BACKUP (worker) ####

def _get_gsheet_client():
    if gspread is None or ServiceAccountCredentials is None:
        return None
    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        return None
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_JSON, scope)
    client = gspread.authorize(creds)
    return client


def append_rows_to_sheet(sheet, rows, header=None):
    """
    Append rows (list of lists) to a worksheet.
    If worksheet is empty, optionally write header first.
    """
    existing = sheet.get_all_values()
    if len(existing) == 0 and header:
        # add header
        sheet.append_row(header, value_input_option="USER_ENTERED")
    if rows:
        # append multiple rows in a batch
        # gspread supports append_rows
        try:
            sheet.append_rows(rows, value_input_option="USER_ENTERED")
        except Exception:
            # fallback to single append if batch fails
            for r in rows:
                sheet.append_row(r, value_input_option="USER_ENTERED")


def backup_last_hour_rows_and_forecasts():
    """
    Append latest-hour rows (all stations) to HourlyHistory sheet,
    and append all per-station forecast rows to Forecasts sheet.
    Append-only; never clears existing content.
    """
    client = _get_gsheet_client()
    if client is None or not GOOGLE_SHEET_ID:
        print("Google Sheets client/ID missing — skipping backup.")
        return

    try:
        book = client.open_by_key(GOOGLE_SHEET_ID)
    except Exception as e:
        print("Failed opening Google Sheet:", e)
        return

    # ensure worksheets exist (create if missing)
    try:
        wh_hist = book.worksheet("HourlyHistory")
    except Exception:
        wh_hist = book.add_worksheet(title="HourlyHistory", rows="1000", cols="30")
    try:
        wh_fore = book.worksheet("Forecasts")
    except Exception:
        wh_fore = book.add_worksheet(title="Forecasts", rows="10000", cols="6")

    # read local history
    if not os.path.exists(HISTORY_FILE):
        print("Local history missing, skipping Google backup.")
        return
    df = pd.read_csv(HISTORY_FILE)
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])

    if df.empty:
        print("History empty, nothing to append.")
        return

    # Latest timestamp rows (the single last hourly run)
    latest_ts = df["Datetime"].max()
    new_rows = df[df["Datetime"] == latest_ts].copy()
    if new_rows.empty:
        print("No new rows for latest timestamp to append.")
    else:
        # prepare header and rows
        header = new_rows.columns.tolist()
        rows = new_rows.fillna("").values.tolist()
        append_rows_to_sheet(wh_hist, rows, header=header)
        print(f"Appended {len(rows)} rows to HourlyHistory")

    # Append forecasts: for each forecast CSV in FORECAST_DIR create rows: StationId, Datetime, AQI
    forecast_files = sorted([f for f in os.listdir(FORECAST_DIR) if f.startswith("forecast_24h_") and f.endswith(".csv")])
    forecast_rows = []
    for f in forecast_files:
        st_id = f.replace("forecast_24h_", "").replace(".csv", "")
        try:
            fdf = pd.read_csv(os.path.join(FORECAST_DIR, f))
            # ensure Datetime is string/ISO
            if "Datetime" in fdf.columns:
                try:
                    fdf["Datetime"] = pd.to_datetime(fdf["Datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
                except:
                    fdf["Datetime"] = fdf["Datetime"].astype(str)
            for _, r in fdf.iterrows():
                forecast_rows.append([st_id, r.get("Datetime", ""), r.get("AQI", "")])
        except Exception:
            continue

    if forecast_rows:
        # header
        header_f = ["StationId", "Datetime", "AQI"]
        append_rows_to_sheet(wh_fore, forecast_rows, header=header_f)
        print(f"Appended {len(forecast_rows)} forecast rows to Forecasts")


#### MAIN PIPELINE ####

def run_pipeline():
    stations = load_stations()
    features = ensure_feature_list()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    all_now = []

    for idx, s in stations.iterrows():
        st_id = s["StationId"]
        waqi_id = s["waqi_id"]
        st_enc = s["StationId_enc"]

        live, status = fetch_live_waqi(waqi_id)
        if live is None:
            print(f"[{st_id}] WAQI fetch failed: {status}")
            time.sleep(SLEEP_BETWEEN)
            continue

        row = build_feature_row(live_dict=live, station_enc=st_enc, feature_list=features, dt=datetime.now())
        try:
            aqi_now = float(model.predict(row)[0])
        except Exception as e:
            print(f"[{st_id}] Model prediction failed: {e}")
            time.sleep(SLEEP_BETWEEN)
            continue

        hist_entry = {
            "StationId": st_id,
            "StationName": s.get("StationName", ""),
            "City": s.get("City", ""),
            "StationId_enc": st_enc,
            "Datetime": datetime.now().isoformat(),
            "AQI": aqi_now,
            "PM2.5": live.get("PM2.5"),
            "NO2": live.get("NO2"),
            "CO": live.get("CO"),
            "SO2": live.get("SO2"),
            "O3": live.get("O3"),
            "Temperature": live.get("Temperature"),
            "RelativeHumidity": live.get("RelativeHumidity"),
            "WindSpeed": live.get("WindSpeed"),
            "Pressure": live.get("Pressure"),
            "DewPoint": live.get("DewPoint"),
            "WindDirection": live.get("WindDirection")
        }

        save_hourly_history(hist_entry, HISTORY_FILE)

        forecast_df = forecast_24h_recursive(aqi_now=aqi_now, live_dict=live, model=model, feature_list=features, station_enc=st_enc, seed_dt=datetime.now())
        out_csv = os.path.join(FORECAST_DIR, f"forecast_24h_{st_id}.csv")
        forecast_df.to_csv(out_csv, index=False)
        print(f"[{st_id}] AQI Now: {aqi_now:.1f} | Forecast saved → {out_csv}")

        all_now.append({"StationId": st_id, "StationName": s.get("StationName", ""), "City": s.get("City", ""), "AQI_Now": aqi_now, "Datetime": datetime.now().isoformat()})

        time.sleep(SLEEP_BETWEEN)

    # snapshot
    if all_now:
        df_now = pd.DataFrame(all_now)
        snap_dir = os.path.join(PROJECT_BASE, "nowcast_snapshot")
        os.makedirs(snap_dir, exist_ok=True)
        snapshot_path = os.path.join(snap_dir, f"nowcast_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_now.to_csv(snapshot_path, index=False)
        print("Snapshot saved →", snapshot_path)

    # backup (append-only)
    try:
        backup_last_hour_rows_and_forecasts()
    except Exception as e:
        print("Backup Error:", e)

    print("Pipeline complete. History updated at:", HISTORY_FILE)


if __name__ == "__main__":
    print("Starting AQI pipeline: Live Nowcast + 24h Forecast")
    run_pipeline()
    print("Pipeline complete. History updated at:", HISTORY_FILE)
