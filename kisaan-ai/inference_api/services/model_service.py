"""
Model Service — loads and manages all ML models at API startup.
Handles model versioning via MLflow model registry.
"""

import os
import logging
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Optional

from inference_api.schemas.models import (
    CropPredictRequest, CropPredictResponse,
    YieldPredictRequest, YieldPredictResponse,
)
from rag_engine.rag_service import RAGService

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.crop_model = None
        self.crop_scaler = None
        self.crop_encoder = None
        self.yield_model = None
        self.rag: Optional[RAGService] = None
        self._startup_time = None
        self._model_versions = {}

    async def load_all(self):
        """Load all models into memory on startup."""
        import time
        self._startup_time = time.time()

        await self._load_crop_model()
        await self._load_yield_model()
        await self._load_rag()
        logger.info("✅ All models loaded successfully")

    async def _load_crop_model(self):
        """Load crop recommendation model from MLflow registry or local."""
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            # Try to load latest production model from registry
            model_uri = "models:/crop_xgb/Production"
            self.crop_model = mlflow.sklearn.load_model(model_uri)
            self._model_versions["crop"] = "mlflow:crop_xgb/Production"
            logger.info("Loaded crop model from MLflow registry")
        except Exception:
            # Fallback to local artifact
            artifact_dir = "ml_models/crop_recommendation/artifacts"
            try:
                self.crop_model = joblib.load(f"{artifact_dir}/best_model.pkl")
                self.crop_scaler = joblib.load(f"{artifact_dir}/scaler.pkl")
                self.crop_encoder = joblib.load(f"{artifact_dir}/label_encoder.pkl")
                self._model_versions["crop"] = "local:v1"
                logger.info("Loaded crop model from local artifacts")
            except Exception as e:
                logger.warning(f"Could not load crop model: {e}. Using mock.")
                self._model_versions["crop"] = "mock"

    async def _load_yield_model(self):
        """Load yield prediction model."""
        try:
            self.yield_model = joblib.load("ml_models/yield_prediction/artifacts/yield_model.pkl")
            self._model_versions["yield"] = "local:v1"
        except Exception as e:
            logger.warning(f"Could not load yield model: {e}. Using mock.")
            self._model_versions["yield"] = "mock"

    async def _load_rag(self):
        """Initialize RAG service."""
        try:
            self.rag = RAGService(
                openai_key=os.getenv("OPENAI_API_KEY"),
                pinecone_key=os.getenv("PINECONE_API_KEY"),
                pinecone_index=os.getenv("PINECONE_INDEX", "kisaan-ai"),
                redis_url=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}",
            )
            self._model_versions["rag"] = "gpt-4o-mini+pinecone"
            logger.info("RAG service initialized")
        except Exception as e:
            logger.warning(f"Could not init RAG: {e}")
            self._model_versions["rag"] = "unavailable"

    async def predict_crop(self, request: CropPredictRequest) -> CropPredictResponse:
        """Run crop prediction inference."""
        features = np.array([[
            request.nitrogen, request.phosphorus, request.potassium,
            request.temperature, request.humidity, request.ph, request.rainfall,
        ]])

        version = self._model_versions.get("crop", "mock")

        if self.crop_model is None or version == "mock":
            # Mock response for development
            return CropPredictResponse(
                recommended_crop="Rice",
                confidence=0.87,
                top_3=[
                    {"crop": "Rice", "confidence": 0.87},
                    {"crop": "Maize", "confidence": 0.09},
                    {"crop": "Cotton", "confidence": 0.04},
                ],
                model_version="mock-v0",
                features_used=request.model_dump(),
            )

        if self.crop_scaler:
            features = self.crop_scaler.transform(features)

        proba = self.crop_model.predict_proba(features)[0]
        top3_idx = proba.argsort()[-3:][::-1]
        classes = self.crop_encoder.classes_ if self.crop_encoder else list(range(len(proba)))

        return CropPredictResponse(
            recommended_crop=str(classes[top3_idx[0]]),
            confidence=float(proba[top3_idx[0]]),
            top_3=[
                {"crop": str(classes[i]), "confidence": float(proba[i])}
                for i in top3_idx
            ],
            model_version=version,
            features_used=request.model_dump(),
        )

    def get_status(self) -> dict:
        import time
        return {
            "models_loaded": self._model_versions,
            "uptime_seconds": time.time() - (self._startup_time or 0),
        }
