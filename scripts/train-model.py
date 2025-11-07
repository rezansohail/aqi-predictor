import os
import pandas as pd
import numpy as np
import hopsworks
import hsfs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Connect to Hopsworks
api_key = os.environ["HOPSWORKS_API_KEY"]
project = hopsworks.login(api_key_value=api_key, project="AQI_Predictor_KARACHI")
fs = project.get_feature_store()

# Fetch feature data
fg = fs.get_feature_group("karachi_aqi_pollution_history", version=4)
df = fg.read().sort_values(by="timestamp_utc").dropna()

# Derived features
df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)
df["target_next_hour"] = df["aqi_index"].shift(-1).fillna(df["aqi_index"].iloc[-1])

# Feature selection
features = [
    "aqi_index", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "temp_c", "humidity", "pressure_hpa", "wind_speed",
    "hour", "day", "month", "dayofweek", "aqi_change_rate"
]
X = df[features]
y = df["target_next_hour"]

# Time-based split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train the best-performing model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"RandomForest — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# Save and register the trained model
model_path = "random_forest_aqi_model.pkl"
joblib.dump(model, model_path)

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="aqi_random_forest",
    metrics={"RMSE": rmse, "MAE": mae, "R2": r2},
    description="Daily retrained AQI prediction model using Random Forest (Open-Meteo data)"
)
model_meta.save(model_path)
print("Model saved & registered successfully in Hopsworks!")
