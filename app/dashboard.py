# app/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import joblib
from dotenv import load_dotenv
import os
import json
import mlflow
import matplotlib.pyplot as plt

# === Load Environment Variables ===
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# === Streamlit Page Config ===
st.set_page_config(
    page_title="🌫️ AQI Prediction Dashboard",
    page_icon="🌤️",
    layout="wide",
)

# === Modern CSS Styling ===
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            color: #fff;
        }
        .main-title {
            font-size: 2.6rem;
            font-weight: bold;
            color: #b3e0f2;
            text-shadow: 1px 1px 10px rgba(255,255,255,0.4);
            margin-bottom: 20px;
        }
        .metric-card {
            background: rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 2px 16px rgba(0,0,0,0.3);
        }
        .alert {
            padding: 14px;
            font-size: 1.2rem;
            border-radius: 12px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 16px;
        }
        .alert-good {
            background: linear-gradient(90deg, #43cea2, #185a9d);
            color: #fff;
        }
        .alert-moderate {
            background: linear-gradient(90deg, #f7971e, #ffd200);
            color: #222;
        }
        .alert-bad {
            background: linear-gradient(90deg, #ff5858, #f09819);
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# === Helper Functions ===
def get_city_coordinates(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    resp = requests.get(url).json()
    if len(resp) == 0:
        return None, None
    return resp[0]["lat"], resp[0]["lon"]

def fetch_live_data(city):
    lat, lon = get_city_coordinates(city)
    if lat is None:
        return None

    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"

    aqi_data = requests.get(aqi_url).json()
    weather_data = requests.get(weather_url).json()

    if "list" not in aqi_data:
        return None

    pollutants = aqi_data["list"][0]["components"]
    aqi = aqi_data["list"][0]["main"]["aqi"]
    weather = weather_data["main"]

    df = pd.DataFrame({
        "datetime": [datetime.utcnow()],
        "temp": [weather["temp"]],
        "humidity": [weather["humidity"]],
        "pressure": [weather["pressure"]],
        **pollutants,
        "aqi": [aqi]
    })
    return df

def fetch_historical_aqi(city):
    lat, lon = get_city_coordinates(city)
    if lat is None:
        return None
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={lat}&lon={lon}&start={int(start_date.timestamp())}&end={int(end_date.timestamp())}&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url).json()
    if "list" not in resp:
        return None
    records = [{"datetime": datetime.utcfromtimestamp(e["dt"]), "aqi": e["main"]["aqi"]} for e in resp["list"]]
    return pd.DataFrame(records)

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "alert-good"
    elif aqi <= 100:
        return "Moderate", "alert-moderate"
    else:
        return "Poor", "alert-bad"

# === MLflow Integration ===
def fetch_latest_mlflow_metrics():
    """Fetch metrics from latest MLflow run"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("AQI Prediction Project")
        runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if len(runs) == 0:
            return None
        latest_run = runs[0].data.metrics
        return {
            "RMSE": latest_run.get("RMSE", None),
            "MAE": latest_run.get("MAE", None),
            "R2": latest_run.get("R2", None),
        }
    except Exception as e:
        print(f"⚠️ MLflow connection issue: {e}")
        return None

# === Sidebar Navigation ===
st.sidebar.header("🌍 Navigation")
sidebar_tab = st.sidebar.radio("Choose a section", ["Live AQI", "Trends", "3-Day Forecast", "Model Metrics"])
city = st.sidebar.text_input("Enter City", "New York")

# Load trained model
try:
    model = joblib.load("models/aqi_predictor_mlflow.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found! Please run your training pipeline first.")
    model = None

# === Tabs ===
if sidebar_tab == "Live AQI":
    st.markdown(f"<div class='main-title'>🌤️ Live Air Quality in {city}</div>", unsafe_allow_html=True)
    df_live = fetch_live_data(city)
    if df_live is not None:
        aqi_value = df_live["aqi"].values[0]
        category, color_class = get_aqi_category(aqi_value)
        st.markdown(f"<div class='alert {color_class}'>Air Quality: <b>{category}</b></div>", unsafe_allow_html=True)

        cols = st.columns(4)
        metrics = [("Temp (°C)", "temp"), ("Humidity (%)", "humidity"),
                   ("Pressure (hPa)", "pressure"), ("AQI", "aqi")]
        for i, (label, key) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"<div class='metric-card'><h4>{label}</h4><p>{round(df_live[key].values[0],2)}</p></div>", unsafe_allow_html=True)

        st.markdown("<h4>Pollutant Levels (μg/m³)</h4>", unsafe_allow_html=True)
        pollutant_keys = [k for k in df_live.columns if k not in ["datetime", "temp", "humidity", "pressure", "aqi"]]
        st.bar_chart(df_live[pollutant_keys].iloc[0])
    else:
        st.error("Failed to fetch AQI data.")

elif sidebar_tab == "Trends":
    st.markdown(f"<div class='main-title'>📈 7-Day AQI Trend for {city}</div>", unsafe_allow_html=True)
    df_hist = fetch_historical_aqi(city)
    if df_hist is not None and not df_hist.empty:
        st.line_chart(df_hist.set_index("datetime")["aqi"])
    else:
        st.warning("No historical AQI data available.")

elif sidebar_tab == "3-Day Forecast":
    st.markdown(f"<div class='main-title'>📅 3-Day AQI Forecast for {city}</div>", unsafe_allow_html=True)
    forecast_path = "models/aqi_3day_forecast.csv"

    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)

        # --- Normalize and clean column names ---
        df.columns = [c.lower().strip() for c in df.columns]
        if "ds" in df.columns and "yhat" in df.columns:
            df.rename(columns={
                "ds": "Date",
                "yhat": "Predicted_AQI",
                "yhat_lower": "Lower_Bound",
                "yhat_upper": "Upper_Bound"
            }, inplace=True)
        elif "datetime" in df.columns and "aqi" in df.columns:
            df.rename(columns={"datetime": "Date", "aqi": "Predicted_AQI"}, inplace=True)

        if "Date" not in df.columns or "Predicted_AQI" not in df.columns:
            st.error("⚠️ Forecast file missing required columns.")
        else:
            # Convert date
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

            # --- Add AQI category labels ---
            def categorize_aqi(aqi):
                if aqi <= 50:
                    return "Good ✅"
                elif aqi <= 100:
                    return "Moderate 🟡"
                elif aqi <= 150:
                    return "Poor 🟠"
                elif aqi <= 200:
                    return "Very Poor 🔴"
                else:
                    return "Hazardous ☠️"

            df["Quality"] = df["Predicted_AQI"].apply(categorize_aqi)

            # --- Display in table and chart ---
            st.markdown("### 🌫️ Forecasted AQI & Quality (Next 3 Days)")
            st.dataframe(df[["Date", "Predicted_AQI", "Quality"]].tail(3), use_container_width=True)

            st.line_chart(df.set_index("Date")["Predicted_AQI"])
            st.caption("Forecast includes qualitative air quality assessment for each day.")
    else:
        st.warning("⚠️ No forecast file found. Run your training pipeline to generate `aqi_3day_forecast.csv`.")


# -------------------------------------------------------------------------
elif sidebar_tab == "Model Metrics":
    st.markdown("<div class='main-title'>🧠 Model Performance (from MLflow)</div>", unsafe_allow_html=True)

    metrics = fetch_latest_mlflow_metrics()
    if metrics:
        cols = st.columns(3)
        cols[0].metric("RMSE", f"{metrics['RMSE']:.3f}")
        cols[1].metric("MAE", f"{metrics['MAE']:.3f}")
        cols[2].metric("R²", f"{metrics['R2']:.3f}")
    else:
        st.info("No metrics found in MLflow. Showing local metrics instead.")
        if os.path.exists("models/training_metrics.json"):
            with open("models/training_metrics.json") as f:
                m = json.load(f)
            st.json(m)
        else:
            st.warning("No training metrics file found.")

    # === Feature Importance ===
    if model is not None:
        st.markdown("### 🔍 Feature Importance Explanation")
        st.markdown("""
        <div style='font-size:1.05rem; line-height:1.7;'>
        The model estimates <b>how much each feature contributes</b> to predicting the Air Quality Index (AQI).<br><br>
        <ul>
        <li><b>Pollutant features</b> (CO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃) usually have the strongest influence — they directly reflect air composition.</li>
        <li><b>Weather features</b> (temperature, humidity, pressure, wind speed) impact pollutant dispersion and accumulation.</li>
        <li><b>Time features</b> (hour, day, month, weekday) capture daily and seasonal patterns in pollution levels.</li>
        <li><b>AQI change rate</b> shows how rapidly conditions are worsening or improving.</li>
        </ul>
        The chart below shows which features the Random Forest model found most important.
        </div>
        """, unsafe_allow_html=True)

        try:
            feature_names = [
                "co", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
                "temperature", "humidity", "pressure", "wind_speed",
                "hour", "day", "month", "weekday", "aqi_change_rate"
            ]
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(fi_df["Feature"], fi_df["Importance"], color="#6dd5ed")
            ax.set_xlabel("Importance Score")
            ax.set_title("Feature Importance (Random Forest)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"⚠️ Could not plot feature importance: {e}")
