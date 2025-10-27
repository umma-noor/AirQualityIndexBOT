import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from prophet import Prophet
import json

# ---------------------------
# 1️⃣ Load API Keys
# ---------------------------
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---------------------------
# 2️⃣ Utility: Get Coordinates of City
# ---------------------------
def get_city_coordinates(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url).json()
    if not response:
        raise ValueError(f"❌ Could not find coordinates for city: {city}")
    lat, lon = response[0]["lat"], response[0]["lon"]
    print(f"📍 {city} located at (lat={lat}, lon={lon})")
    return lat, lon

# ---------------------------
# 3️⃣ Fetch Historical Air Pollution Data
# ---------------------------
def fetch_aqi_data(city="New York", days=60):
    print(f"🌫️ Fetching AQI history for {city} ({days} days)...")

    lat, lon = get_city_coordinates(city)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_API_KEY}"
    )
    response = requests.get(url).json()

    if "list" not in response:
        raise ValueError("❌ No AQI data found. Check API key or city.")

    records = []
    for entry in response["list"]:
        ts = datetime.utcfromtimestamp(entry["dt"])
        main = entry["main"]
        comps = entry["components"]
        records.append({
            "datetime": ts,
            "aqi": main["aqi"],
            "co": comps.get("co"),
            "no2": comps.get("no2"),
            "o3": comps.get("o3"),
            "so2": comps.get("so2"),
            "pm2_5": comps.get("pm2_5"),
            "pm10": comps.get("pm10"),
            "nh3": comps.get("nh3")
        })

    df = pd.DataFrame(records)
    print(f"✅ Fetched {len(df)} AQI records.")
    return df

# ---------------------------
# 4️⃣ Fetch Weather Data
# ---------------------------
def fetch_weather_data(city="New York", days=60):
    print(f"🌤️ Fetching weather data for {city} ({days} days)...")

    lat, lon = get_city_coordinates(city)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    url_template = "https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
    records = []

    for i in range(days):
        date = start_date + timedelta(days=i)
        url = url_template.format(lat=lat, lon=lon, key=OPENWEATHER_API_KEY)
        try:
            response = requests.get(url, timeout=10).json()
            if response.get("cod") == 200:
                main = response["main"]
                wind = response.get("wind", {})
                records.append({
                    "datetime": date,
                    "temperature": main.get("temp"),
                    "humidity": main.get("humidity"),
                    "pressure": main.get("pressure"),
                    "wind_speed": wind.get("speed")
                })
        except Exception as e:
            print(f"⚠️ Error fetching weather for {date.date()}: {e}")

        if i % 10 == 0:
            print(f"☀️ Processed {i} days...")

    df = pd.DataFrame(records)
    print(f"✅ Weather data fetched: {len(df)} records")
    return df

# ---------------------------
# 5️⃣ Merge & Feature Engineering
# ---------------------------
def create_features(aqi_df, weather_df):
    print("🧩 Creating features...")
    aqi_df["datetime"] = pd.to_datetime(aqi_df["datetime"])
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    df = pd.merge_asof(aqi_df.sort_values("datetime"),
                       weather_df.sort_values("datetime"),
                       on="datetime", direction="nearest")

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

    df = df.dropna()
    print("✅ Feature dataset ready. Shape:", df.shape)
    return df

# ---------------------------
# 6️⃣ Train, Evaluate & Log with MLflow
# ---------------------------
def train_and_log_mlflow(df, city, days):
    print("🤖 Training model with MLflow logging...")
    features = [
        "co", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "temperature", "humidity", "pressure", "wind_speed",
        "hour", "day", "month", "weekday", "aqi_change_rate"
    ]
    target = "aqi"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.set_experiment("AQI Prediction Project")

    with mlflow.start_run(run_name=f"{city}_last{days}days_RF"):
        mlflow.log_param("city", city)
        mlflow.log_param("days_of_data", days)
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 12)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, artifact_path="trained_model")

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    os.makedirs("models", exist_ok=True)
    with open("models/training_metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"\n📊 Model Evaluation:\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}")
    return model

# ---------------------------
# 7️⃣ Forecast Future AQI (Prophet)
# ---------------------------
def forecast_aqi(df, days_ahead=3):
    df_prophet = df[["datetime", "aqi"]].rename(columns={"datetime": "ds", "aqi": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days_ahead)
    forecast.columns = ["Date", "Predicted_AQI", "Lower", "Upper"]
    forecast["Category"] = forecast["Predicted_AQI"].apply(categorize_aqi)
    forecast.to_csv("models/aqi_3day_forecast.csv", index=False)
    print("📈 3-Day AQI Forecast saved at models/aqi_3day_forecast.csv")
    return forecast

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive Groups)"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

# ---------------------------
# 8️⃣ Save Model
# ---------------------------
def save_model(model):
    os.makedirs("models", exist_ok=True)
    model_path = "models/aqi_predictor_mlflow.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved at: {model_path}")

# ---------------------------
# 9️⃣ Save to Hopsworks Feature Store
# ---------------------------
def save_to_feature_store(df):
    import hopsworks
    print("📡 Connecting to Hopsworks...")

    df["datetime_str"] = df["datetime"].astype(str)

    # Login and get feature store
    project = hopsworks.login()
    fs = project.get_feature_store()  # uses default connected FS

    # Create or reuse feature group
    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Air quality and weather features for AQI prediction",
        primary_key=["datetime_str"],
        online_enabled=True
    )

    # Insert data
    feature_group.insert(df)
    print("✅ Features successfully inserted into Hopsworks Feature Store!")

# ---------------------------
# 🔚 Main Execution
# ---------------------------
if __name__ == "__main__":
    city = "New York"
    days = 360

    print("🚀 Training Pipeline Started...")
    aqi_df = fetch_aqi_data(city, days)
    weather_df = fetch_weather_data(city, days)
    df = create_features(aqi_df, weather_df)

    save_to_feature_store(df)       # ✅ Hopsworks FS
    model = train_and_log_mlflow(df, city, days)  # MLflow logging
    save_model(model)               # Local model
    forecast_aqi(df, days_ahead=3)  # 3-day Prophet forecast

    print("\n✅ Training Pipeline Completed Successfully.")
