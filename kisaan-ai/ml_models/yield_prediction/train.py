"""
Crop Yield Prediction Model Training
Predicts crop yield in hg/ha given country, crop, year, rainfall,
pesticide usage and average temperature.

Dataset columns:
  Area, Item, Year, hg/ha_yield,
  average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp

Models compared: Random Forest, XGBoost, Gradient Boosting
Best model saved to artifacts/yield_model.pkl
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI   = "http://localhost:5000"
EXPERIMENT   = "yield_prediction"
DATA_PATH    = "data_pipeline/datasets/yield_df.csv"
ARTIFACTS    = "ml_models/yield_prediction/artifacts"


# ── Data loading ────────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)

    # Drop index column if present
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]

    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical columns
    area_enc  = LabelEncoder()
    item_enc  = LabelEncoder()
    df["Area_enc"] = area_enc.fit_transform(df["Area"])
    df["Item_enc"] = item_enc.fit_transform(df["Item"])

    feature_cols = [
        "Area_enc", "Item_enc", "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes", "avg_temp",
    ]
    target_col = "hg/ha_yield"

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, area_enc, item_enc, scaler, feature_cols, df


# ── Metrics helper ───────────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ── Random Forest ────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="rf_yield"):
        params = {"n_estimators": 200, "max_depth": 20,
                  "random_state": 42, "n_jobs": -1}
        mlflow.log_params(params)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae, rmse, r2 = eval_metrics(y_test, preds)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        logger.info(f"Random Forest — MAE: {mae:.0f} | RMSE: {rmse:.0f} | R²: {r2:.4f}")
        return model, r2


# ── XGBoost ──────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="xgb_yield"):
        params = {
            "n_estimators": 300, "max_depth": 7,
            "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": 42,
        }
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params, verbosity=0)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)

        mae, rmse, r2 = eval_metrics(y_test, preds)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        logger.info(f"XGBoost     — MAE: {mae:.0f} | RMSE: {rmse:.0f} | R²: {r2:.4f}")
        return model, r2


# ── Gradient Boosting ────────────────────────────────────────────
def train_gradient_boosting(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="gb_yield"):
        params = {
            "n_estimators": 200, "max_depth": 6,
            "learning_rate": 0.1, "subsample": 0.8,
            "random_state": 42,
        }
        mlflow.log_params(params)

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae, rmse, r2 = eval_metrics(y_test, preds)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        logger.info(f"Grad Boost  — MAE: {mae:.0f} | RMSE: {rmse:.0f} | R²: {r2:.4f}")
        return model, r2


# ── Main ─────────────────────────────────────────────────────────
def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    X, y, area_enc, item_enc, scaler, feature_cols, df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Yield range: {y.min():.0f} – {y.max():.0f} hg/ha")

    results = {}

    logger.info("\n--- Training Random Forest ---")
    rf_model, rf_r2 = train_random_forest(X_train, X_test, y_train, y_test)
    results["random_forest"] = (rf_model, rf_r2)

    logger.info("\n--- Training XGBoost ---")
    xgb_model, xgb_r2 = train_xgboost(X_train, X_test, y_train, y_test)
    results["xgboost"] = (xgb_model, xgb_r2)

    logger.info("\n--- Training Gradient Boosting ---")
    gb_model, gb_r2 = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results["gradient_boosting"] = (gb_model, gb_r2)

    # Pick best by R²
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]

    logger.info(f"\n{'='*55}")
    logger.info("Model Comparison (R² score):")
    for name, (_, r2) in results.items():
        marker = " ← BEST" if name == best_name else ""
        logger.info(f"  {name:22s}: {r2:.4f}{marker}")
    logger.info(f"{'='*55}")

    # Save best model + encoders
    joblib.dump(best_model, f"{ARTIFACTS}/yield_model.pkl")
    joblib.dump(area_enc,   f"{ARTIFACTS}/area_encoder.pkl")
    joblib.dump(item_enc,   f"{ARTIFACTS}/item_encoder.pkl")
    joblib.dump(scaler,     f"{ARTIFACTS}/yield_scaler.pkl")

    # Save metadata
    metadata = {
        "best_model": best_name,
        "r2": results[best_name][1],
        "all_scores": {k: v[1] for k, v in results.items()},
        "feature_cols": feature_cols,
        "target": "hg/ha_yield",
        "areas": list(area_enc.classes_),
        "crops": list(item_enc.classes_),
        "yield_min": float(y.min()),
        "yield_max": float(y.max()),
        "yield_mean": float(y.mean()),
    }
    with open(f"{ARTIFACTS}/yield_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nArtifacts saved to {ARTIFACTS}/")
    logger.info(f"Supported areas : {len(area_enc.classes_)}")
    logger.info(f"Supported crops : {len(item_enc.classes_)}")
    logger.info(f"  Sample crops  : {list(item_enc.classes_[:8])}")
    logger.info("\n✅ Yield model training complete!")
    logger.info("   Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()
