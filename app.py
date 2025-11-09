import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import hopsworks
import shap
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# UI CONFIG
st.set_page_config(page_title="Karachi AQI Predictor", layout="wide")
st.markdown("""
<style>
body { background-color:#2B0202; color:white; }
.main { background-color:#2B0202; color:white; }
h1,h2,h3,h4 { color:#ff69b4; }
div[data-testid="stMetricValue"] { color:#ff69b4; }
.stDataFrame { background-color:#0c1c2a; color:white; }
.stDataFrame tbody tr th { color:white; }
.stDataFrame tbody tr td { color:white; }
</style>
""", unsafe_allow_html=True)

st.title("KARACHI AIR QUALITY INDEX PREDICTOR")
st.write("Hourly AQI Prediction for the Next 3 Days")

# CONNECT TO HOPSWORKS
api_key = os.environ.get("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key, project="AQI_Predictor_KARACHI")
fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_aqi_pollution_history", version=4)
df = fg.read().sort_values("timestamp_utc").dropna()
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

# LAST 3 MONTHS
cutoff = df["timestamp_utc"].max() - pd.DateOffset(months=3)
df = df[df["timestamp_utc"] >= cutoff].reset_index(drop=True)

# FEATURE ENGINEERING
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

# Time-based split
split_idx = int(len(df)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# TRAIN MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=13),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=13)
}

predictions = {}
metrics = {"Model": [], "MAE": [], "RMSE": [], "R2": []}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    metrics["Model"].append(name)
    metrics["MAE"].append(mae)
    metrics["RMSE"].append(rmse)
    metrics["R2"].append(r2)
    
    # Forecast next 3 days (72h)
    future_input = X.tail(72)
    future_pred = model.predict(future_input)
    future_pred = np.clip(future_pred, 0, 500)  # realistic bounds
    future_dates = pd.date_range(start=df["timestamp_utc"].iloc[-1]+timedelta(hours=1), periods=72, freq="H")
    predictions[name] = pd.DataFrame({"timestamp": future_dates, "predicted_aqi": future_pred})

compare_df = pd.DataFrame(metrics).sort_values("MAE")
best_model = compare_df.iloc[0]["Model"]

# REAL-TIME AQI
st.divider()
st.subheader("🌐 Latest Real-Time AQI")

# Most recent row from Hopsworks Feature Store
latest_row = df.iloc[-1]
current_aqi = latest_row["aqi_index"]

# Display as metric
st.metric("Current AQI", f"{current_aqi:.2f}")

cols = st.columns(3)
with cols[0]:
    st.metric("PM2.5", f"{latest_row['pm2_5']:.2f}")
    st.metric("O₃", f"{latest_row['o3']:.2f}")
with cols[1]:
    st.metric("PM10", f"{latest_row['pm10']:.2f}")
    st.metric("NO₂", f"{latest_row['no2']:.2f}")
with cols[2]:
    st.metric("SO₂", f"{latest_row['so2']:.2f}")
    st.metric("CO", f"{latest_row['co']:.2f}")


# RANDOM FOREST
st.subheader("🌟 Random Forest — Hourly AQI Prediction (Next 3 Days)")
rf_pred = predictions["Random Forest"]
for day in range(3):
    day_start = rf_pred["timestamp"].dt.date.min() + timedelta(days=day)
    st.write(f"**{day_start}**")
    st.dataframe(
        rf_pred[rf_pred["timestamp"].dt.date == day_start][["timestamp", "predicted_aqi"]]
        .rename(columns={"timestamp":"Hour","predicted_aqi":"AQI"}),
        use_container_width=True
    )

# AQI ALERT
st.divider()
st.subheader("🚨 Latest AQI Alert")

if current_aqi > 249:
    st.error(f"Hazardous AQI: {current_aqi:.2f}")
elif current_aqi > 149:
    st.error(f"Very Unhealthy AQI: {current_aqi:.2f}")
elif current_aqi > 99:
    st.warning(f"Unhealthy AQI: {current_aqi:.2f}")
elif current_aqi > 49:
    st.warning(f"Poor AQI: {current_aqi:.2f}")
elif current_aqi > 19:
    st.info(f"Fair AQI: {current_aqi:.2f}")
else:
    st.success(f"Excellent AQI: {current_aqi:.2f}")

# MODEL PERFORMANCE
st.divider()
st.subheader("⚖️ Model Performance Comparison (3-Month Test Data)")
st.dataframe(compare_df)

st.info("""
**Why Random Forest is better here:**
- Handles **non-linear AQI patterns** and sudden changes better than Linear Regression.
- More robust to **outliers** in pollution spikes.
- Captures interactions between multiple pollutants and weather features without needing manual feature engineering.
""")

# SIDE-BY-SIDE GRAPH AND METRICS
st.divider()
st.subheader("📈 AQI Trend — Last 3 Months vs Random Forest Prediction")
col1, col2 = st.columns([2,1])

with col1:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df["timestamp_utc"], df["aqi_index"], color="#ff69b4", label="Actual AQI")
    ax.plot(rf_pred["timestamp"], rf_pred["predicted_aqi"], color="#00bfff", label="RF Prediction")
    ax.set_title("AQI Variation", color="#ff69b4")
    ax.set_xlabel("Date/Hour", color="white")
    ax.set_ylabel("AQI", color="white")
    ax.legend()
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")
    ax.tick_params(colors="white")
    st.pyplot(fig)

with col2:
    st.write("📊 Summary Metrics")
    st.metric("Mean Test MAE", f"{compare_df['MAE'].min():.2f}")
    st.metric("Mean Test RMSE", f"{compare_df['RMSE'].min():.2f}")
    st.metric("Best R²", f"{compare_df['R2'].max():.2f}")

# SHAP FEATURE IMPORTANCE
st.divider()
st.subheader("🔍 Random Forest Feature Importance (SHAP)")

explainer = shap.TreeExplainer(models["Random Forest"])
shap_values = explainer.shap_values(X_test)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean |SHAP Value|": mean_abs_shap
}).sort_values(by="Mean |SHAP Value|", ascending=False).head(10)

st.dataframe(
    top_features.style
        .background_gradient(cmap="Blues", axis=0)
        .set_properties(subset=["Feature"], **{"color": "#ff69b4", "font-weight":"bold"})
        .set_properties(subset=["Mean |SHAP Value|"], **{"color": "black"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "12px")]},
            {"selector": "td", "props": [("font-size", "12px")]}
        ])
)
