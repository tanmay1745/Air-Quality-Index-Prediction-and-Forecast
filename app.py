import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import scheduler

st.set_page_config(layout="wide", page_title="AQI Dashboard")

# CONFIG
BASE = "."
STATIONS_PATH = rf"{BASE}/Final_Station.csv"
HISTORY_PATH = rf"{BASE}/mnt/aqi/hourly_history.csv"
FORECAST_DIR = rf"{BASE}/mnt/aqi/forecast"



# LOAD DATA 
@st.cache_data(ttl=10)
def load_stations():
    if not os.path.exists(STATIONS_PATH):
        return pd.DataFrame()
    return pd.read_csv(STATIONS_PATH)

@st.cache_data(ttl=10)
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_PATH, parse_dates=["Datetime"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=10)
def load_forecast(station_id):
    f_path = f"{FORECAST_DIR}/forecast_24h_{station_id}.csv"
    if not os.path.exists(f_path):
        return pd.DataFrame()
    return pd.read_csv(f_path, parse_dates=["Datetime"])


# DATA 
stations = load_stations()
history = load_history()

# --- SAFE GUARD: if history is empty OR missing StationId, show message and stop ---
if history.empty or ("StationId" not in history.columns):
    st.warning("hourly_history.csv not ready yet. Waiting for the cron job to create a valid file with 'StationId' column.")
    st.stop()

if stations.empty:
    st.error("Stations CSV not found in this folder.")
    st.stop()

# TITLE 
st.markdown("<h1 style='text-align:center;'>AQI — Live Nowcast & 24-Hour Forecast</h1>", unsafe_allow_html=True)
st.caption(f"Last updated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

# SELECT CITY & STATION 
cities = sorted(stations["City"].unique())
city = st.selectbox("Select City", cities)

stations_filtered = stations[stations["City"] == city]
station_label = stations_filtered["StationId"].astype(str) + " — " + stations_filtered["StationName"]
station_map = dict(zip(station_label, stations_filtered["StationId"]))

station_choice = st.selectbox("Select Station", station_label)
station_id = station_map[station_choice]

# CURRENT AQI 
st.markdown("## Current AQI (Nowcast)")

latest = history[history["StationId"] == station_id].sort_values("Datetime")

if latest.empty:
    st.warning("No history available yet for this station.")
    st.stop()

last_row = latest.iloc[-1]

# FIXED 3-COLUMN LAYOUT 
c1, c2, c3 = st.columns([0.8, 1.2, 1.2])

# AQI BADGE + GAUGE
with c1:
    aqi = last_row["AQI"]

    def aqi_category(aqi):
        if aqi <= 50: return "Good", "#009966"
        elif aqi <= 100: return "Moderate", "#ffde33"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff9933"
        elif aqi <= 200: return "Unhealthy", "#cc0033"
        elif aqi <= 300: return "Very Unhealthy", "#660099"
        else: return "Hazardous", "#7e0023"

    cat, color = aqi_category(aqi)

    st.markdown(
        f"""
        <div style="
            padding:18px;
            border-radius:12px;
            background-color:{color};
            text-align:center;
        ">
            <h2 style="color:white; margin:0;">AQI: {int(aqi)}</h2>
            <h4 style="color:white; margin:0;">{cat}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={"text": "AQI Level"},
        gauge={
            "axis": {"range": [0, 500]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [0, 50], "color": "#009966"},
                {"range": [51, 100], "color": "#ffde33"},
                {"range": [101, 150], "color": "#ff9933"},
                {"range": [151, 200], "color": "#cc0033"},
                {"range": [201, 300], "color": "#660099"},
                {"range": [301, 500], "color": "#7e0023"},
            ],
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

# POLLUTANTS TABLE 
with c2:
    st.markdown("### Pollutants / Weather")

    cols = [
        "PM2.5","NO2","CO","SO2","O3",
        "Temperature","RelativeHumidity","WindSpeed",
        "Pressure","DewPoint","WindDirection"
    ]

    df_pol = pd.DataFrame({
        "Metric": cols,
        "Value": [last_row.get(c, None) for c in cols]
    })

    st.table(
        df_pol.style.set_properties(**{"text-align": "center"})
        .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
    )


# LATEST SNAPSHOT 
with c3:
    st.markdown("### Latest Snapshot")
    snapshot_df = pd.DataFrame({
        "Date": [last_row["Datetime"].date()],
        "Time": [last_row["Datetime"].strftime("%H:%M:%S")],
        "AQI": [last_row["AQI"]]
    })

    st.table(
        snapshot_df.style.set_properties(**{"text-align": "center"})
        .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
    )


# 24-HOUR FORECAST 
st.markdown("## 24-Hour Forecast")

forecast_df = load_forecast(station_id)

if forecast_df.empty:
    st.info("No 24-hour forecast file found for this station yet.")
else:
    forecast_df["Date"] = forecast_df["Datetime"].dt.date
    forecast_df["Time"] = forecast_df["Datetime"].dt.strftime("%H:%M:%S")

    fig = px.line(
        forecast_df,
        x="Datetime",
        y="AQI",
        markers=True,
        title=f"24-Hour AQI Forecast — {station_id}"
    )

    fig.update_xaxes(tickformat="%d %b %Y\n%H:%M")

    fig.update_layout(
        xaxis_title="Datetime",
        yaxis_title="AQI",
        modebar=dict(remove=[
            'zoom2d','pan','select','lasso2d',
            'zoomIn2d','zoomOut2d','autoScale2d',
            'resetScale2d','toImage'
        ])
    )

    st.plotly_chart(fig, use_container_width=True)

    fore_table = forecast_df[["Date", "Time", "AQI"]].copy()
    fore_table = fore_table.style.set_properties(**{
        "text-align": "center"
    }).set_table_styles([
        dict(selector="th", props=[("text-align", "center")])
    ])

    st.write("### Forecast Table")
    st.dataframe(fore_table, use_container_width=True)


# HISTORICAL 48 HOURS
st.markdown("## Historical AQI (Last 48 Hours)")

hist = latest.tail(48).copy()
if hist.empty:
    st.info("Not enough historical data yet.")
else:
    fig2 = px.line(
        hist,
        x="Datetime",
        y="AQI",
        markers=True,
        title=f"Last 48 Hours — {station_id}"
    )
    fig2.update_xaxes(tickformat="%d %b %Y\n%H:%M")

    fig2.update_layout(
        modebar=dict(remove=[
            'zoom2d','pan','select','lasso2d',
            'zoomIn2d','zoomOut2d','autoScale2d',
            'resetScale2d','toImage'
        ])
    )

    st.plotly_chart(fig2, use_container_width=True)
