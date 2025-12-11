import os
import pandas as pd
import streamlit as st
from datetime import datetime

# optionally using gspread if secret present
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    gspread = None
    ServiceAccountCredentials = None

st.set_page_config(layout="wide", page_title="AQI Dashboard")

# PATHS / CONFIG
BASE = "."
STATIONS_PATH = os.path.join(BASE, "Final_Station.csv")
HISTORY_PATH = os.path.join(BASE, "hourly_history.csv")
FORECAST_DIR = os.path.join(BASE, "forecast")

SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON_PATH", "/etc/secrets/SERVICE_ACCOUNT_JSON")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")

# helper: read HourlyHistory and Forecasts from Google Sheets if possible
def read_sheets():
    if gspread is None or ServiceAccountCredentials is None:
        return None, None, "gspread-missing"
    if not GOOGLE_SHEET_ID or not os.path.exists(SERVICE_ACCOUNT_JSON):
        return None, None, "no-creds"
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_JSON, scope)
        client = gspread.authorize(creds)
        book = client.open_by_key(GOOGLE_SHEET_ID)
    except Exception as e:
        return None, None, f"open_failed: {e}"

    # HourlyHistory
    try:
        ws = book.worksheet("HourlyHistory")
        values = ws.get_all_records()
        hist_df = pd.DataFrame(values)
        if not hist_df.empty and "Datetime" in hist_df.columns:
            hist_df["Datetime"] = pd.to_datetime(hist_df["Datetime"], errors="coerce")
        else:
            # no Datetime column => try to find and coerce if possible
            pass
    except Exception:
        hist_df = pd.DataFrame()

    # Forecasts
    try:
        wf = book.worksheet("Forecasts")
        values = wf.get_all_records()
        fore_df = pd.DataFrame(values)
        if not fore_df.empty and "Datetime" in fore_df.columns:
            fore_df["Datetime"] = pd.to_datetime(fore_df["Datetime"], errors="coerce")
    except Exception:
        fore_df = pd.DataFrame()

    return hist_df, fore_df, None

@st.cache_data(ttl=60)
def load_stations():
    if not os.path.exists(STATIONS_PATH):
        return pd.DataFrame()
    return pd.read_csv(STATIONS_PATH)

@st.cache_data(ttl=60)
def load_data():
    # try Google Sheets first
    hist_df, fore_df, err = read_sheets()
    if err is None:
        return hist_df, fore_df, "sheets"
    # else try local files
    hist_local = pd.DataFrame()
    fore_local = pd.DataFrame()
    if os.path.exists(HISTORY_PATH):
        try:
            hist_local = pd.read_csv(HISTORY_PATH, parse_dates=["Datetime"])
        except Exception:
            hist_local = pd.read_csv(HISTORY_PATH)
    # load forecasts from folder (latest per station)
    try:
        files = [f for f in os.listdir(FORECAST_DIR) if f.startswith("forecast_24h_") and f.endswith(".csv")]
        rows = []
        for f in files:
            sid = f.replace("forecast_24h_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(FORECAST_DIR, f))
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
            df["StationId"] = sid
            rows.append(df)
        if rows:
            fore_local = pd.concat(rows, ignore_index=True)
    except Exception:
        fore_local = pd.DataFrame()
    return hist_local, fore_local, "local"

# load everything
stations = load_stations()
history, forecasts, source = load_data()

# SAFE GUARD:
if (history is None) or history.empty or ("StationId" not in history.columns):
    st.warning("hourly_history.csv / HourlyHistory sheet not ready yet. Waiting for the cron job to create data. (Tried reading from Google Sheets first.)")
    st.stop()

if stations.empty:
    st.error("Stations CSV not found in this folder (Final_Station.csv).")
    st.stop()

# UI
st.markdown("<h1 style='text-align:center;'>AQI — Live Nowcast & 24-Hour Forecast</h1>", unsafe_allow_html=True)
st.caption(f"Data source: {source} — Last load: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

cities = sorted(stations["City"].unique())
city = st.selectbox("Select City", cities)

stations_filtered = stations[stations["City"] == city]
station_label = stations_filtered["StationId"].astype(str) + " — " + stations_filtered["StationName"]
station_map = dict(zip(station_label, stations_filtered["StationId"]))

station_choice = st.selectbox("Select Station", station_label)
station_id = station_map[station_choice]

latest = history[history["StationId"] == station_id].sort_values("Datetime")
if latest.empty:
    st.warning("No history available yet for this station.")
    st.stop()
last_row = latest.iloc[-1]

# layout
c1, c2, c3 = st.columns([0.8, 1.2, 1.2])

with c1:
    aqi = float(last_row["AQI"])
    def aqi_category(aqi):
        if aqi <= 50: return "Good", "#009966"
        if aqi <= 100: return "Moderate", "#ffde33"
        if aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff9933"
        if aqi <= 200: return "Unhealthy", "#cc0033"
        if aqi <= 300: return "Very Unhealthy", "#660099"
        return "Hazardous", "#7e0023"
    cat, color = aqi_category(aqi)
    st.markdown(f"<div style='padding:18px;border-radius:12px;background-color:{color};text-align:center;'><h2 style='color:white;margin:0;'>AQI: {int(aqi)}</h2><h4 style='color:white;margin:0;'>{cat}</h4></div>", unsafe_allow_html=True)

    import plotly.graph_objects as go
    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=aqi, title={"text": "AQI Level"}, gauge={"axis": {"range":[0,500]}, "bar": {"color": "black"}}))
    st.plotly_chart(fig_gauge, use_container_width=True)

with c2:
    st.markdown("### Pollutants / Weather")
    cols = ["PM2.5","NO2","CO","SO2","O3","Temperature","RelativeHumidity","WindSpeed","Pressure","DewPoint","WindDirection"]
    df_pol = pd.DataFrame({"Metric": cols, "Value": [last_row.get(c, None) for c in cols]})
    st.table(df_pol.style.set_properties(**{"text-align":"center"}).set_table_styles([dict(selector="th", props=[("text-align","center")])]))

with c3:
    st.markdown("### Latest Snapshot")
    snapshot_df = pd.DataFrame({"Date": [pd.to_datetime(last_row["Datetime"]).date()], "Time":[pd.to_datetime(last_row["Datetime"]).strftime("%H:%M:%S")], "AQI":[last_row["AQI"]]})
    st.table(snapshot_df.style.set_properties(**{"text-align":"center"}).set_table_styles([dict(selector="th", props=[("text-align","center")])]))

st.markdown("## 24-Hour Forecast")
# try to find forecasts for station either from forecasts dataframe (sheet/local)
if forecasts is None or forecasts.empty:
    st.info("No 24-hour forecast file found for this station yet.")
else:
    # prefer forecasts that have StationId column
    if "StationId" in forecasts.columns:
        fdf = forecasts[forecasts["StationId"] == station_id].copy()
    else:
        # if forecasts from local folder had StationId in rows earlier we handled that
        fdf = forecasts.copy()
        if "StationId" in fdf.columns:
            fdf = fdf[fdf["StationId"] == station_id]
    if fdf.empty:
        st.info("No 24-hour forecast file found for this station yet.")
    else:
        if "Datetime" in fdf.columns:
            fdf["Date"] = pd.to_datetime(fdf["Datetime"]).dt.date
            fdf["Time"] = pd.to_datetime(fdf["Datetime"]).dt.strftime("%H:%M:%S")
        import plotly.express as px
        fig = px.line(fdf, x="Datetime", y="AQI", markers=True, title=f"24-Hour AQI Forecast — {station_id}")
        fig.update_xaxes(tickformat="%d %b %Y\n%H:%M")
        st.plotly_chart(fig, use_container_width=True)
        fore_table = fdf[["Date","Time","AQI"]].copy()
        st.write("### Forecast Table")
        st.dataframe(fore_table, use_container_width=True)

st.markdown("## Historical AQI (Last 48 Hours)")
hist = latest.tail(48).copy()
if hist.empty:
    st.info("Not enough historical data yet.")
else:
    import plotly.express as px
    fig2 = px.line(hist, x="Datetime", y="AQI", markers=True, title=f"Last 48 Hours — {station_id}")
    fig2.update_xaxes(tickformat="%d %b %Y\n%H:%M")
    st.plotly_chart(fig2, use_container_width=True)
