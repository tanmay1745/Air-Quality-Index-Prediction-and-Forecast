# 🌫️ Air Quality Index (AQI) Prediction & 24-Hour Forecast  
An end-to-end production-ready system that predicts **real-time AQI** using live pollutant & weather data from the WAQI API and generates a **24-hour AQI forecast** for 96 stations across India.  
The entire system runs automatically every hour and updates the dashboard + data storage with zero manual intervention.

---

## 🚀 Live Dashboard  
🔗 **Streamlit App:**  
👉 https://air-quality-index-prediction-and-forecast.onrender.com  

View real-time AQI, forecasts, pollutants, and historical trends for all stations.

---

## 🔍 Project Overview  
This end-to-end pipeline:

1. Fetches live pollutant + weather data from WAQI for 96 stations  
2. Builds feature vectors (lags, rolling means, time features, etc.)  
3. Predicts **current AQI (nowcast)** using a trained CatBoost model  
4. Generates a **24-hour recursive forecast** for each station  
5. Stores:  
   - hourly AQI & features  
   - forecast results  
   - snapshot of all stations  
6. Uploads the complete hourly dataset to **Google Sheets for backup**  
7. Displays everything through a clean Streamlit dashboard  

All backend processing runs **automatically every 1 hour** through a scheduler.

---

## 🧠 Key Features  

### 📡 Live AQI Prediction  
- Fetches real-time PM2.5, NO₂, CO, SO₂, O₃, temperature, humidity, wind speed, pressure, and dew point  
- Cleans & standardizes WAQI API output  
- Builds engineered features & time-based signals  
- Predicts current AQI using a CatBoost regression model  

### 📈 24-Hour Forecast  
- Uses recursive forecasting  
- Includes historical lags & rolling averages  
- Forecasts 1–24 hours into the future  

### 📊 Streamlit Dashboard  
- Station & city selector  
- Live AQI category card  
- Pollutant table  
- 24-hour forecast graph  
- Last 48-hour history graph  
- Latest snapshot summary  

### 💾 Automated Data Storage  
Every hour, the pipeline writes:

- **hourly_history.csv** — full dataset (AQI + pollutants + weather)  
- **forecast/** — 24-hour forecast CSV per station  
- **nowcast_snapshot/** — AQI snapshot for all 96 stations  

### ☁️ Cloud Backup (Google Sheets)  
Since Render’s filesystem is ephemeral, the system uploads:

- **complete hourly_history.csv** including pollutants + weather to Google Sheets after every cycle to ensure safe long-term storage.

---

## 🛠 Tech Stack  

### Backend  
- Python 
- Scikit-Learn 
- CatBoost  
- Pandas / NumPy  
- Requests (WAQI API)

### Frontend  
- Streamlit  
- Plotly  

### Deployment  
- Render Web Service  
- Custom scheduler for hourly execution  

### External APIs  
- WAQI API  
- Google Sheets API  

---

## 🎯 Outputs  

- Real-time AQI prediction (96 stations)  
- 24-hour AQI forecast (96 stations) 
- Live pollutant and weather readings  
- Hourly historical AQI dataset  
- Snapshot of all stations  
- Automatic Google Sheets backup  
- Interactive Streamlit dashboard  

---

## 📌 Future Enhancements  

- Add new features from accumulated hourly dataset  
- Retrain model weekly for better accuracy   
- Trend analytics dashboard (multi-city/time-series)  
- Alerts for hazardous AQI levels   


