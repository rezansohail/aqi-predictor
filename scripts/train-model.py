import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Logging in to Hopsworks
api_key = os.environ["HOPSWORKS_API_KEY"]
project = hopsworks.login(api_key_value=api_key, project="AQI_Predictor_KARACHI")
fs = project.get_feature_store()

# Read Feature Group
fg = fs.get_feature_group("karachi_aqi_pollution_history", version=4)
df = fg.read().sort_values(by="timestamp_utc").dropna()

# Derived features
df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)
df["target_next_3_days"] = df["aqi_index"].shift(-72).fillna(df["aqi_index"].iloc[-1])

# Define features and target
features = [
    "aqi_index", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "temp_c", "humidity", "pressure_hpa", "wind_speed",
    "hour", "day", "month", "dayofweek", "aqi_change_rate"
]
X = df[features]
y = df["target_next_3_days"]

# Time-based split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, preds, squared=False)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Linear Regression Model — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# Save model
model_path = "linear_regression_3day_aqi_model.pkl"
joblib.dump(model, model_path)

# Register model
mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="aqi_linear_regression_3day",
    metrics={"RMSE": rmse, "MAE": mae, "R2": r2},
    description="AQI prediction model for next 3 days using Linear Regression"
)
model_meta.save(model_path)

print("Model saved and registered successfully!")
