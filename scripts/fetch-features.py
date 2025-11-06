import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import hopsworks
import os

LAT, LON = 24.8607, 67.0011

end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=90)

# Air quality data
aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aqi_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
               "sulphur_dioxide", "ozone"],
    "timezone": "auto"
}
aqi_data = requests.get(aqi_url, params=aqi_params).json()
df_aqi = pd.DataFrame(aqi_data["hourly"])
df_aqi["timestamp_utc"] = pd.to_datetime(df_aqi["time"])
df_aqi.drop(columns=["time"], inplace=True)

# Weather data
weather_url = "https://api.open-meteo.com/v1/forecast"
weather_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "hourly": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m"],
    "timezone": "auto"
}
weather_data = requests.get(weather_url, params=weather_params).json()
df_weather = pd.DataFrame(weather_data["hourly"])
df_weather["timestamp_utc"] = pd.to_datetime(df_weather["time"])
df_weather.drop(columns=["time"], inplace=True)

# Merge data
df = pd.merge(df_aqi, df_weather, on="timestamp_utc", how="inner")

df = df.rename(columns={
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure_hpa",
    "wind_speed_10m": "wind_speed"
})

# Derived & time-based features
df["aqi_index"] = df[["pm2_5", "pm10", "no2", "so2", "o3", "co"]].mean(axis=1)
df["hour"] = df["timestamp_utc"].dt.hour.astype(int)
df["day"] = df["timestamp_utc"].dt.day.astype(int)
df["month"] = df["timestamp_utc"].dt.month.astype(int)
df["dayofweek"] = df["timestamp_utc"].dt.dayofweek.astype(int)
df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)
df["next_day_aqi"] = df["aqi_index"].shift(-24).bfill()

df = df[[
    "timestamp_utc", "aqi_index", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "temp_c", "humidity", "pressure_hpa", "wind_speed",
    "hour", "day", "month", "dayofweek", "aqi_change_rate", "next_day_aqi"
]]

api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = "AQI_Predictor_KARACHI"

project = hopsworks.login(api_key_value=api_key, project=project_name)
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="karachi_aqi_pollution_history",
    version=4,
    primary_key=["timestamp_utc"],
    event_time="timestamp_utc",
    description="3 months of hourly AQI + weather data from Open-Meteo"
)

fg.insert(df)
