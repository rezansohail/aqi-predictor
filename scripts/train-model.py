import os
import pandas as pd
import numpy as np
import hopsworks
import hsfs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

hsfs.engine.set_default("python")  # prevent Flight socket timeout

api_key = os.environ["HOPSWORKS_API_KEY"]
project = hopsworks.login(api_key_value=api_key, project="AQI_Predictor_KARACHI")
fs = project.get_feature_store()

fg = fs.get_feature_group("karachi_aqi_pollution_history", version=4)
df = fg.read().sort_values(by="timestamp_utc").dropna()

# Derived features
df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)
df["target_next_hour"] = df["aqi_index"].shift(-1).fillna(df["aqi_index"].iloc[-1])

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

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append((name, rmse, mae, r2))
    print(f"{name} — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# Best model
results.sort(key=lambda x: x[1])
best_name, best_rmse, best_mae, best_r2 = results[0]
best_model = models[best_name]

print("\nBest Model:", best_name)
model_path = f"{best_name}_aqi_model.pkl"
joblib.dump(best_model, model_path)

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name=f"aqi_{best_name.lower()}",
    metrics={"RMSE": best_rmse, "MAE": best_mae, "R2": best_r2},
    description="Daily retrained AQI model (Open-Meteo)"
)
model_meta.save(model_path)
print("Model saved & registered successfully!")
