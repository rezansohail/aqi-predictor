import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

api_key = os.environ["HOPSWORKS_API_KEY"]
project = hopsworks.login(api_key_value=api_key, project="AQI_Predictor_KARACHI")
fs = project.get_feature_store()

fg = fs.get_feature_group("karachi_aqi_pollution_history", version=4)
df = fg.read().sort_values(by="timestamp_utc").dropna()

df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)
df["target_next_3_days"] = df["aqi_index"].shift(-72).fillna(df["aqi_index"].iloc[-1])

features = [
    "aqi_index", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "temp_c", "humidity", "pressure_hpa", "wind_speed",
    "hour", "day", "month", "dayofweek", "aqi_change_rate"
]
X = df[features]
y = df["target_next_3_days"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=200, random_state=103)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Random Forest Model — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

model_path = "random_forest_3day_aqi_model.pkl"
joblib.dump(model, model_path)

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="aqi_random_forest_3day",
    metrics={"RMSE": rmse, "MAE": mae, "R2": r2},
    description="AQI prediction model for next 3 days using Random Forest"
)
model_meta.save(model_path)

print("Model saved and registered successfully!")
