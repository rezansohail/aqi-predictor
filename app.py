import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Karachi AQI — 3 Day Forecast", layout="wide")

if "HOPSWORKS_API_KEY" in st.secrets:
    HOPSWORKS_API_KEY = st.secrets["HOPSWORKS_API_KEY"]
else:
    import os
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

PROJECT_NAME = "AQI_Predictor_KARACHI"
FEATURE_GROUP_NAME = "karachi_aqi_pollution_history"
FEATURE_GROUP_VERSION = 4

st.title("Karachi AQI — 3-Day Forecast")

if HOPSWORKS_API_KEY is None:
    st.error("HOPSWORKS_API_KEY not found. Add it to Streamlit secrets or set environment variable and restart.")
    st.stop()

# Connect to Hopsworks and load features
@st.cache_data(ttl=600)
def load_features():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read().sort_values(by="timestamp_utc").reset_index(drop=True)
    return df, project

with st.spinner("Loading data from Hopsworks..."):
    df, project = load_features()

st.write(f"Data loaded — records: {len(df)} (last timestamp: {df['timestamp_utc'].iloc[-1]})")

df = df.copy()
df["aqi_change_rate"] = df["aqi_index"].diff().fillna(0)

# target is AQI 72 hours ahead (3 days)
TARGET_NAME = "target_next_3_days"
df[TARGET_NAME] = df["aqi_index"].shift(-72)

features = [
    "aqi_index", "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "temp_c", "humidity", "pressure_hpa", "wind_speed",
    "hour", "day", "month", "dayofweek", "aqi_change_rate"
]

df_features = df.dropna(subset=features + ["aqi_index"])
df_model = df_features.dropna(subset=[TARGET_NAME]).reset_index(drop=True)

if len(df_model) < 200:
    st.warning("Not enough historical labelled rows found for robust training. Need at least 200 labelled rows.")
    # continue anyway

split_idx = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train = train_df[features]
y_train = train_df[TARGET_NAME]
X_test = test_df[features]
y_test = test_df[TARGET_NAME]

# Train Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on test set
preds_test = model.predict(X_test)

mse = mean_squared_error(y_test, preds_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds_test)
r2 = r2_score(y_test, preds_test)
# MAPE (avoid division by zero)
mape = (np.abs((y_test - preds_test) / np.where(y_test == 0, 1e-6, y_test))).mean() * 100
accuracy_pct = max(0.0, 100.0 - mape)  # a simple interpretation

# Retrain on all available labelled data (so model uses max history)
final_model = LinearRegression()
final_model.fit(df_model[features], df_model[TARGET_NAME])

# Use the last 72 rows of the current feature DataFrame
X_recent = df_features[features].tail(72).copy().reset_index(drop=True)

# Predict next 72 hours
pred_next72 = final_model.predict(X_recent)

# Map predictions to future timestamps:
last_timestamp = df["timestamp_utc"].iloc[-1]
future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=72, freq="H")

pred_df = pd.DataFrame({
    "timestamp": future_timestamps,
    "predicted_aqi": pred_next72
})

# Display metrics
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    st.subheader("Model evaluation (test set)")
    st.write(f"RMSE: {rmse:.3f}")
    st.write(f"MAE: {mae:.3f}")
    st.write(f"R²: {r2:.3f}")
with col2:
    st.subheader("Accuracy (derived from MAPE)")
    st.write(f"MAPE: {mape:.2f}%")
    st.write(f"Accuracy: {accuracy_pct:.2f}%")
with col3:
    st.subheader("Data / Model info")
    st.write(f"Training rows: {len(train_df)}")
    st.write(f"Test rows: {len(test_df)}")
    st.write(f"Total labelled rows used: {len(df_model)}")

st.markdown("---")

# Plot
st.subheader("Predicted AQI — Next 3 Days (72 hours)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(pred_df["timestamp"], pred_df["predicted_aqi"], label="Predicted AQI (3 days)", linewidth=2)
ax1.set_xlabel("Timestamp (UTC)")
ax1.set_ylabel("AQI Index")
ax1.legend()
st.pyplot(fig1)

# Plot: actual vs predicted
test_feature_times = test_df["timestamp_utc"].reset_index(drop=True)
test_target_times = test_feature_times + pd.Timedelta(hours=72)

st.subheader("Actual vs Predicted (Test Set)")
comp_df = pd.DataFrame({
    "timestamp": test_target_times,
    "actual": y_test.values,
    "predicted": preds_test
}).reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(comp_df["timestamp"], comp_df["actual"], label="Actual (target time)", linewidth=2)
ax2.plot(comp_df["timestamp"], comp_df["predicted"], label="Predicted", linestyle="dashed")
ax2.set_xlabel("Timestamp (UTC)")
ax2.set_ylabel("AQI Index")
ax2.legend()
st.pyplot(fig2)

st.subheader("Predicted values (next 72 hours)")
st.dataframe(pred_df)

st.subheader("Recent AQI (historical)")
hist_df = df.set_index("timestamp_utc")["aqi_index"].tail(168)  # last 7 days
fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.plot(hist_df.index, hist_df.values, label="AQI (historical)")
ax3.set_xlabel("Timestamp (UTC)")
ax3.set_ylabel("AQI Index")
ax3.legend()
st.pyplot(fig3)

# Optional: allow download of predicted CSV
csv = pred_df.to_csv(index=False)
st.download_button("Download predictions CSV", csv, "predictions_next_3_days.csv", "text/csv")

# End
st.write("Note: Model is trained on the past available labelled data (target = AQI 72h ahead). The evaluation metrics above are calculated on a time-based test split and reflect how well the model predicts AQI 72 hours ahead historically.")
