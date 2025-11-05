import os
import requests
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timezone

project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])   #connect to Hopsworks
fs = project.get_feature_store()

print("Fetching live AQI data for Karachi...")   #fetch live data from OpenWeather API

API_KEY = os.environ["OPENWEATHER_API_KEY"]
lat, lon = 24.8607, 67.0011   #Karachi coordinates

url_air = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"   #AQI endpoint
air_data = requests.get(url_air).json()

url_weather = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"   #weather endpoint
weather_data = requests.get(url_weather).json()

pollution = air_data["list"][0]["components"]
aqi_index = air_data["list"][0]["main"]["aqi"]
timestamp = datetime.now(timezone.utc)

weather = weather_data["main"]
wind = weather_data["wind"]["speed"] if "wind" in weather_data else np.nan

df = pd.DataFrame([{
    "datetime": timestamp,
    "aqi_index": aqi_index,
    "co": pollution["co"],
    "no": pollution["no"],
    "no2": pollution["no2"],
    "o3": pollution["o3"],
    "so2": pollution["so2"],
    "pm2_5": pollution["pm2_5"],
    "pm10": pollution["pm10"],
    "nh3": pollution["nh3"],
    "temp": weather["temp"],
    "feels_like": weather["feels_like"],
    "pressure": weather["pressure"],
    "humidity": weather["humidity"],
    "wind_speed": wind,
    "hour": timestamp.hour,
    "day": timestamp.day,
    "month": timestamp.month
}])

df = df.ffill().bfill()   #handling missing values

fg = fs.get_or_create_feature_group(
    name="karachi_aqi_pollution_history",
    version=3,
    description="Hourly Karachi AQI and weather data",
    primary_key=["timestamp_utc"],
    event_time="timestamp_utc",
    online_enabled=True
)

fg.insert(df)
print("New data inserted into Hopsworks Feature Store!")
