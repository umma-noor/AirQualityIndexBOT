import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ---------------------------
# 1️⃣ Load API Key
# ---------------------------
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---------------------------
# 2️⃣ Get City Coordinates
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
# 3️⃣ Fetch AQI Data
# ---------------------------
def fetch_aqi_data(city="New York", days=60):
    print(f"🌫️ Fetching AQI history for {city} ({days} days)...")
    lat, lon = get_city_coordinates(city)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={lat}&lon={lon}&start={int(start_date.timestamp())}&end={int(end_date.timestamp())}&appid={OPENWEATHER_API_KEY}"
    )
    response = requests.get(url).json()
    if "list" not in response:
        raise ValueError("❌ No AQI data found. Check API key or city.")

    records = []
    for entry in response["list"]:
        ts = datetime.utcfromtimestamp(entry["dt"])
        comps = entry["components"]
        records.append({
            "datetime": ts,
            "aqi": entry["main"]["aqi"],
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
# 4️⃣ Fetch Weather Data (with progress)
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
# 5️⃣ Merge and Feature Creation
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
# 6️⃣ Main Execution
# ---------------------------
if __name__ == "__main__":
    city = "New York"
    days = 360
    print("🚀 Feature Extraction Started...")
    aqi_df = fetch_aqi_data(city, days)
    weather_df = fetch_weather_data(city, days)
    df = create_features(aqi_df, weather_df)

    os.makedirs("models", exist_ok=True)
    df.to_csv("models/feature_data.csv", index=False)
    print("💾 Features saved to models/feature_data.csv")
