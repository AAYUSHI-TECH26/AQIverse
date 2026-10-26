"""
Delhi AQI Prediction - Review 1 (EDA + Preprocessing + baseline modelling)
Dataset: city_name.csv (daily air-quality data for Indian cities)
Run:     python aqi_project.py   (saves all graphs in ./images)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("images", exist_ok=True)
sns.set_style("whitegrid")

def save(name):
    plt.tight_layout()
    plt.savefig(f"images/{name}.png", dpi=120)
    plt.close()

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("city_name.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["City"] == "Delhi"].sort_values("Date").reset_index(drop=True)

print("Shape:", df.shape)
print(df.head())
df.info()
print(df.describe().T)

pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

# ============================================================
# 2. EDA (raw data)
# ============================================================
# 2.1 Missing values
missing_pct = df.isnull().mean().mul(100).sort_values(ascending=False)
print("\nMissing % per column:\n", missing_pct)
missing_pct[missing_pct > 0].plot(kind="bar", figsize=(10, 4), color="tomato",
                                  title="Missing Values (%) per Column")
plt.ylabel("% missing")
save("01_missing_values")

# 2.2 Cleaning basics: target missing -> drop, duplicates -> drop
before = len(df)
df = df.dropna(subset=["AQI"]).drop_duplicates()
print(f"\nRows removed (missing AQI / duplicates): {before - len(df)}")
df = df.set_index("Date")

# 2.3 AQI distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["AQI"], bins=40, kde=True, ax=ax[0], color="steelblue")
ax[0].set_title("AQI Distribution")
sns.boxplot(x=df["AQI"], ax=ax[1], color="lightblue")
ax[1].set_title("AQI Boxplot")
save("02_aqi_distribution")

# 2.4 AQI trend over time
df["AQI"].plot(figsize=(12, 4), title="Delhi AQI Trend", color="darkred")
plt.ylabel("AQI")
save("03_aqi_trend")

# 2.5 Seasonality: month and year
df.groupby(df.index.month)["AQI"].mean().plot(kind="bar", figsize=(8, 4),
                                              title="Average AQI by Month", color="orange")
plt.xlabel("Month"); plt.ylabel("AQI")
save("04_aqi_by_month")

df.groupby(df.index.year)["AQI"].mean().plot(kind="bar", figsize=(8, 4),
                                             title="Average AQI by Year", color="seagreen")
plt.xlabel("Year"); plt.ylabel("AQI")
save("05_aqi_by_year")

# 2.6 Pollutant boxplots BEFORE outlier handling
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[pollutants], palette="Set2")
plt.title("Pollutants - Boxplot BEFORE Outlier Treatment")
save("06_boxplot_before")

# ============================================================
# 3. PREPROCESSING
# ============================================================
# 3.1 Outlier detection (IQR rule, k = 3 -> only extreme values)
def iqr_bounds(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outlier_summary = {}
for col in pollutants:
    lo, hi = iqr_bounds(df[col])
    mask = (df[col] < lo) | (df[col] > hi)
    outlier_summary[col] = int(mask.sum())
    df.loc[mask, col] = np.nan          # 3.2 removal: outlier value removed (set to NaN)

print("\nOutliers removed per pollutant:\n", pd.Series(outlier_summary))
pd.Series(outlier_summary).plot(kind="bar", figsize=(8, 4), color="crimson",
                                title="Outliers Detected (IQR, k=3) per Pollutant")
save("07_outlier_counts")

# 3.3 Missing value imputation (time-series aware, also fills removed outliers)
df[pollutants] = df[pollutants].interpolate(method="time").bfill().ffill()
print("Remaining missing values:", int(df[pollutants].isnull().sum().sum()))

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[pollutants], palette="Set2")
plt.title("Pollutants - Boxplot AFTER Outlier Treatment")
save("08_boxplot_after")

# 3.4 Correlation heatmap (cleaned data)
plt.figure(figsize=(9, 7))
sns.heatmap(df[pollutants + ["AQI"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pollutant Correlation")
save("09_correlation")

# 3.5 Feature engineering
df["AQI_lag1"] = df["AQI"].shift(1)                      # yesterday's AQI
df["AQI_lag7"] = df["AQI"].shift(7)                      # AQI 7 days ago
df["AQI_roll7"] = df["AQI"].shift(1).rolling(7).mean()   # mean of previous 7 days
df["PM25_lag1"] = df["PM2.5"].shift(1)
df["PM10_lag1"] = df["PM10"].shift(1)
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek

# 3.6 Encoding (categorical -> numeric)
def get_season(m):
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5, 6): return "Summer"
    if m in (7, 8, 9): return "Monsoon"
    return "Post-Monsoon"

df["season"] = df["month"].map(get_season)
sns.boxplot(x="season", y="AQI", data=df,
            order=["Winter", "Summer", "Monsoon", "Post-Monsoon"], palette="Set3")
plt.title("AQI by Season")
save("10_aqi_by_season")

df = pd.get_dummies(df, columns=["season"], drop_first=True, dtype=int)  # one-hot encoding
season_cols = [c for c in df.columns if c.startswith("season_")]

num_features = ["AQI_lag1", "AQI_lag7", "AQI_roll7", "PM25_lag1", "PM10_lag1", "month", "dayofweek"]
features = num_features + season_cols
df = df.dropna(subset=features)
df.to_csv("delhi_cleaned.csv")

X, y = df[features], df["AQI"]

# 3.7 Time-based train/test split (no shuffling -> no future leakage)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# 3.8 Scaling (StandardScaler fitted ONLY on train data, inside a Pipeline)
scaler_demo = StandardScaler().fit(X_train[num_features])
scaled = pd.DataFrame(scaler_demo.transform(X_train[num_features]), columns=num_features)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
X_train[num_features].boxplot(ax=ax[0]); ax[0].set_title("Before Scaling")
scaled.boxplot(ax=ax[1]);                ax[1].set_title("After StandardScaler")
for a in ax: a.tick_params(axis="x", rotation=45)
save("11_scaling_before_after")

# ============================================================
# 4. MODELLING (baseline)
# ============================================================
models = {
    "Linear Regression (baseline)": Pipeline([("scaler", StandardScaler()),
                                              ("model", LinearRegression())]),
    "Random Forest": Pipeline([("scaler", StandardScaler()),
                               ("model", RandomForestRegressor(n_estimators=200, random_state=42))]),
}

results, rows = {}, []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = pred
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    rows.append([name, mae, rmse, r2])
    print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).to_csv("model_results.csv", index=False)

plt.figure(figsize=(12, 4))
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, results["Random Forest"], label="Predicted (RF)")
plt.title("Actual vs Predicted AQI"); plt.legend()
save("12_actual_vs_predicted")

rf = models["Random Forest"].named_steps["model"]
pd.Series(rf.feature_importances_, index=features).sort_values().plot(
    kind="barh", figsize=(8, 5), title="Feature Importance (Random Forest)")
save("13_feature_importance")

# ============================================================
# 5. HEALTH RISK CATEGORY
# ============================================================
def health_risk(aqi):
    if aqi <= 100: return "Low"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "High"
    return "Severe"

risk_df = pd.DataFrame({"Predicted_AQI": results["Random Forest"]}, index=y_test.index)
risk_df["Health_Risk"] = risk_df["Predicted_AQI"].apply(health_risk)
print(risk_df["Health_Risk"].value_counts())
print(risk_df.tail())