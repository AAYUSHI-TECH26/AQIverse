
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("city_name.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["City"] == "Delhi"].sort_values("Date").reset_index(drop=True)
print("Shape:", df.shape)
print(df.head())
print(df.isnull().sum())


df = df.dropna(subset=["AQI"]).drop_duplicates()          # target missing -> remove
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
df[pollutants] = df[pollutants].interpolate().bfill()     # time-based gap filling
df = df.set_index("Date")


df["AQI"].plot(figsize=(12, 4), title="Delhi AQI Trend")
plt.ylabel("AQI"); plt.show()


df.groupby(df.index.month)["AQI"].mean().plot(kind="bar", title="Average AQI by Month")
plt.xlabel("Month"); plt.ylabel("AQI"); plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df[pollutants + ["AQI"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pollutant Correlation"); plt.show()


df["AQI_lag1"] = df["AQI"].shift(1)                       # kal ka AQI
df["AQI_lag7"] = df["AQI"].shift(7)                       # 7 din pehle ka AQI
df["AQI_roll7"] = df["AQI"].shift(1).rolling(7).mean()    # pichhle 7 din ka average
df["PM25_lag1"] = df["PM2.5"].shift(1)
df["PM10_lag1"] = df["PM10"].shift(1)
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek

features = ["AQI_lag1", "AQI_lag7", "AQI_roll7", "PM25_lag1", "PM10_lag1", "month", "dayofweek"]

df = df.dropna(subset=features)

X, y = df[features], df["AQI"]

    


split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "Linear Regression (baseline)": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = pred
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")


plt.figure(figsize=(12, 4))
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, results["Random Forest"], label="Predicted (RF)")
plt.title("Actual vs Predicted AQI"); plt.legend(); plt.show()


rf = models["Random Forest"]
pd.Series(rf.feature_importances_, index=features).sort_values().plot(
    kind="barh", title="Feature Importance")
plt.show()


def health_risk(aqi):
    if aqi <= 100:  return "Low"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "High"
    return "Severe"

risk_df = pd.DataFrame({"Predicted_AQI": results["Random Forest"]}, index=y_test.index)
risk_df["Health_Risk"] = risk_df["Predicted_AQI"].apply(health_risk)
print(risk_df["Health_Risk"].value_counts())
print(risk_df.tail())