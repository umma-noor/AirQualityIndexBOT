import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_API_KEY = os.getenv("AQICN_API_KEY")


# ---------------------------
# 1️⃣ Fetch Weather Data
# ---------------------------
def get_weather_data(city: str):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
        }
    else:
        print("Error fetching weather:", data)
        return None


# ---------------------------
# 2️⃣ Fetch AQI Data
# ---------------------------
def get_aqi_data(city: str):
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data.get("status") == "ok":
        aqi = data["data"]["aqi"]
        dominentpol = data["data"]["dominentpol"]
        return {"aqi": aqi, "dominent_pol": dominentpol}
    else:
        print("Error fetching AQI:", data)
        return None


# ---------------------------
# 3️⃣ Combine & Add Features
# ---------------------------
def collect_features(city: str):
    weather = get_weather_data(city)
    aqi = get_aqi_data(city)

    if weather and aqi:
        # Merge weather + AQI data
        features = {**weather, **aqi, "city": city}

        # Add time-based features
        now = datetime.now()
        features["hour"] = now.hour
        features["day"] = now.day
        features["month"] = now.month
        features["weekday"] = now.weekday()  # 0 = Monday

        # Add timestamp
        features["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Add derived feature: AQI change rate (if previous data exists)
        csv_path = "data/aqi_features.csv"
        if os.path.exists(csv_path):
            prev_df = pd.read_csv(csv_path)
            if not prev_df.empty:
                last_aqi = prev_df["aqi"].iloc[-1]
                df["aqi_change_rate"] = df["aqi"] - last_aqi
            else:
                df["aqi_change_rate"] = 0
        else:
            df["aqi_change_rate"] = 0

        return df
    else:
        return None


# ---------------------------
# 4️⃣ Save Data
# ---------------------------
if __name__ == "__main__":
    city = "Lahore"  # You can change or loop through multiple cities
    df = collect_features(city)

    if df is not None:
        print("\n✅ Latest Features:\n", df)
        os.makedirs("data", exist_ok=True)
        csv_path = "data/aqi_features.csv"

        # Append new data to existing CSV (keep history)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        print(f"\n📁 Saved features to {csv_path}")
    else:
        print("\n❌ Failed to collect data.")
