"""
Crop Recommendation Model Training
Compares Random Forest, XGBoost, and Neural Network.
Logs all experiments to MLflow. Saves best model locally.

Features used:
  N, P, K (soil nutrients), temperature, humidity, ph, rainfall
Target:
  crop (22 crop labels)
"""

import os
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT = "crop_recommendation"
DATA_PATH = "data_pipeline/datasets/crop_recommendation.csv"
ARTIFACTS_DIR = "ml_models/crop_recommendation/artifacts"


# ── Neural Network definition ───────────────────────────────────
class CropNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Data loading ────────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, le, scaler, feature_cols


# ── Random Forest ───────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="random_forest"):
        params = {"n_estimators": 200, "max_depth": 15, "random_state": 42, "n_jobs": -1}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Save locally instead of MLflow registry to avoid permission issues
        joblib.dump(model, f"{ARTIFACTS_DIR}/rf_model.pkl")
        

        logger.info(f"Random Forest — Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return model, acc


# ── XGBoost ─────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test, num_classes):
    with mlflow.start_run(run_name="xgboost"):
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "random_state": 42,
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Save locally
        joblib.dump(model, f"{ARTIFACTS_DIR}/xgb_model.pkl")
        

        logger.info(f"XGBoost — Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return model, acc


# ── Neural Network ───────────────────────────────────────────────
def train_neural_net(X_train, X_test, y_train, y_test, num_classes):
    with mlflow.start_run(run_name="neural_network"):
        params = {"epochs": 50, "batch_size": 64, "lr": 1e-3, "hidden": "128-64"}
        mlflow.log_params(params)

        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.LongTensor(y_train)
        X_te = torch.FloatTensor(X_test)
        y_te = torch.LongTensor(y_test)

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

        model = CropNet(X_train.shape[1], num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(50):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                mlflow.log_metric("train_loss", loss.item(), step=epoch)
                logger.info(f"Epoch {epoch+1}/50 — Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            preds = model(X_te).argmax(dim=1).numpy()

        acc = accuracy_score(y_te.numpy(), preds)
        f1 = f1_score(y_te.numpy(), preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Save locally
        torch.save(model.state_dict(), f"{ARTIFACTS_DIR}/nn_model.pt")
        

        logger.info(f"Neural Network — Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return model, acc


# ── Main training loop ──────────────────────────────────────────
def main():
    # Setup MLflow — use local mlruns folder as artifact store
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Load and split data
    X, y, label_encoder, scaler, feature_cols = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    num_classes = len(label_encoder.classes_)
    logger.info(f"Classes: {num_classes}, Train size: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Crop labels: {list(label_encoder.classes_)}")

    # Train all three models
    results = {}

    logger.info("\n--- Training Random Forest ---")
    rf_model, rf_acc = train_random_forest(X_train, X_test, y_train, y_test)
    results["random_forest"] = (rf_model, rf_acc)

    logger.info("\n--- Training XGBoost ---")
    xgb_model, xgb_acc = train_xgboost(X_train, X_test, y_train, y_test, num_classes)
    results["xgboost"] = (xgb_model, xgb_acc)

    logger.info("\n--- Training Neural Network ---")
    nn_model, nn_acc = train_neural_net(X_train, X_test, y_train, y_test, num_classes)
    results["neural_network"] = (nn_model, nn_acc)

    # Compare and pick best
    acc_scores = {name: acc for name, (model, acc) in results.items()}
    best_name = max(acc_scores, key=acc_scores.get)
    best_model = results[best_name][0]

    logger.info(f"\n{'='*50}")
    logger.info("Model Comparison:")
    for name, acc in acc_scores.items():
        marker = " ← BEST" if name == best_name else ""
        logger.info(f"  {name:20s}: {acc:.4f}{marker}")
    logger.info(f"{'='*50}")

    # Save best model as the production model
    if best_name == "neural_network":
        torch.save(best_model.state_dict(), f"{ARTIFACTS_DIR}/best_model.pt")
        logger.info(f"Best model (Neural Network) saved to {ARTIFACTS_DIR}/best_model.pt")
    else:
        joblib.dump(best_model, f"{ARTIFACTS_DIR}/best_model.pkl")
        logger.info(f"Best model ({best_name}) saved to {ARTIFACTS_DIR}/best_model.pkl")

    # Always save scaler and label encoder
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.pkl")
    joblib.dump(label_encoder, f"{ARTIFACTS_DIR}/label_encoder.pkl")

    # Save model metadata
    import json
    metadata = {
        "best_model": best_name,
        "accuracy": acc_scores[best_name],
        "all_scores": acc_scores,
        "feature_cols": feature_cols,
        "num_classes": num_classes,
        "classes": list(label_encoder.classes_),
    }
    with open(f"{ARTIFACTS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Scaler, label encoder, and metadata saved to {ARTIFACTS_DIR}/")
    logger.info("\n✅ Training complete! Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()