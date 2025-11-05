import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

print("Loading feature data from Hopsworks...")
fg = fs.get_feature_group("karachi_aqi_pollution_history", version=1)
df = fg.read().sort_values(by="datetime")

df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)
df["target_next_hour"] = df["aqi_index"].shift(-1).fillna(df["aqi_index"].iloc[-1])
df = df.dropna()

features = [
    "aqi_index", "co", "no2", "pm2_5", "pm10", "humidity", "temp", "pressure", "aqi_change_rate", "hour", "day", "month"
]
X = df[features]
y = df["target_next_hour"]

print(f"Using {len(df)} samples for training.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

#training multiple models
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

#selecting best model
results.sort(key=lambda x: x[1])  #sort by RMSE
best_model_name, best_rmse, best_mae, best_r2 = results[0]
print("\nBest Model Summary:")
print(f"Model: {best_model_name}")
print(f"RMSE: {best_rmse:.3f}, MAE: {best_mae:.3f}, R²: {best_r2:.3f}")

best_model = models[best_model_name]

model_path = f"{best_model_name}_karachi_aqi.pkl"
joblib.dump(best_model, model_path)


mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name=f"karachi_aqi_{best_model_name.lower()}",
    metrics={"RMSE": best_rmse, "MAE": best_mae, "R2": best_r2},
    description="AQI prediction model retrained daily"
)
model_meta.save(model_path)
print("Model registered in Hopsworks!")
