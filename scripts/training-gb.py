import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

api_key = os.environ["HOPSWORKS_API_KEY"]
project = hopsworks.login(api_key_value=api_key, project="AQI_Predictor_KARACHI")
fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_aqi_pollution_history", version=4)
df = fg.read().sort_values("timestamp_utc").dropna()
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

cutoff = df["timestamp_utc"].max() - pd.DateOffset(months=3)
df = df[df["timestamp_utc"] >= cutoff].reset_index(drop=True)

df["hour"] = df["timestamp_utc"].dt.hour
df["day"] = df["timestamp_utc"].dt.day
df["month"] = df["timestamp_utc"].dt.month
df["dayofweek"] = df["timestamp_utc"].dt.dayofweek
df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)

features = [
    "aqi_index","co","no2","o3","so2","pm2_5","pm10",
    "temp_c","humidity","pressure_hpa","wind_speed",
    "hour","day","month","dayofweek","aqi_change_rate"
]
X = df[features]
y = df["aqi_index"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = GradientBoostingRegressor(n_estimators=200, random_state=103)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"Gradient Boosting — MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

model_path = "gradient_boosting_3day_aqi_model.pkl"
joblib.dump(model, model_path)

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="aqi_gradient_boosting_3day",
    metrics={"MAE": mae, "RMSE": rmse, "R2": r2},
    description="AQI prediction model for next 3 days using Gradient Boosting"
)
model_meta.save(model_path)
print("Gradient Boosting model saved and registered successfully!")
