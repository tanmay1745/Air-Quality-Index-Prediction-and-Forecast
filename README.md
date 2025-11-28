# 🌫️ Air Quality Index (AQI) Prediction & 24-Hour Forecast  
An end-to-end system that predicts **real-time AQI** using live pollutant + weather data from the WAQI API and forecasts **AQI for the next 24 hours** with a Streamlit interface.

---

### 🔍 Project Overview  
This project collects real-time environmental data, predicts the current AQI using a trained machine learning model, and forecasts AQI levels for the next 24 hours.  
The application also stores hourly AQI and pollutant data for future analysis, enabling continuous model improvement and addition of new features.

---

### 🧠 Features  
- Fetches **live pollutant & weather data** from the **WAQI API**  
- Predicts **current AQI** using ML model 
- Generates **24-hour AQI forecast**  
- Streamlit interface for easy interaction  
- Automatically saves hourly data for long-term analysis  
- Future enhancement planned:  
  - Add 2–3 new features using accumulated historical data  
  - Improve forecasting accuracy  
  - Add visual trends dashboard  

---

### 🛠 Tech Stack  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Requests (API calls)  
- Matplotlib / Seaborn  
- WAQI API  

---

### 🎯 Output
- Real-time AQI prediction
- 24-hour forecast curve
- Live pollutant levels
- Auto-updated historical AQI dataset (hourly)
- Clean and interactive Streamlit UI

---

### 📌 Future Improvements
- Add new features from accumulated hourly data
- Retrain model weekly for better accuracy
- Deploy forecasting dashboard with trend visualization
- Integrate alerts for hazardous AQI levels