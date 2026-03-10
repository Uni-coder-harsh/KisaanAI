"""
Pydantic schemas for all KisaanAI API endpoints.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Crop Recommendation ─────────────────────────────────────────

class CropPredictRequest(BaseModel):
    nitrogen: float = Field(..., ge=0, le=140, description="Soil nitrogen (kg/ha)")
    phosphorus: float = Field(..., ge=0, le=145, description="Soil phosphorus (kg/ha)")
    potassium: float = Field(..., ge=0, le=205, description="Soil potassium (kg/ha)")
    temperature: float = Field(..., ge=0, le=50, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=300, description="Rainfall (mm)")
    district: Optional[str] = Field(None, description="District name for feature store lookup")

    class Config:
        json_schema_extra = {
            "example": {
                "nitrogen": 40,
                "phosphorus": 60,
                "potassium": 45,
                "temperature": 28.5,
                "humidity": 72,
                "ph": 6.5,
                "rainfall": 120,
                "district": "Bangalore Rural"
            }
        }


class CropPredictResponse(BaseModel):
    recommended_crop: str
    confidence: float = Field(..., ge=0, le=1)
    top_3: list[dict]
    model_version: str
    features_used: dict


# ── Yield Prediction ────────────────────────────────────────────

class YieldPredictRequest(BaseModel):
    crop: str = Field(..., description="Crop name (e.g. Rice, Wheat)")
    area_hectares: float = Field(..., gt=0, description="Area under cultivation (hectares)")
    season: str = Field(..., description="Kharif | Rabi | Zaid")
    state: str = Field(..., description="Indian state")
    annual_rainfall: float = Field(..., ge=0, description="Annual rainfall (mm)")
    fertilizer_kg: float = Field(..., ge=0, description="Fertilizer used (kg/ha)")
    pesticide_kg: float = Field(..., ge=0, description="Pesticide used (kg/ha)")


class YieldPredictResponse(BaseModel):
    crop: str
    predicted_yield_kg_per_ha: float
    total_estimated_yield_kg: float
    confidence_interval: dict  # {"lower": x, "upper": y}
    model_version: str


# ── RAG Chat ────────────────────────────────────────────────────

class Language(str, Enum):
    english = "en"
    hindi = "hi"
    kannada = "kn"
    tamil = "ta"
    telugu = "te"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    language: Language = Language.english
    district: Optional[str] = None
    crop_context: Optional[str] = None
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Which crop is best for my soil in Karnataka?",
                "language": "en",
                "district": "Bangalore Rural"
            }
        }


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    language: str
    translated_question: Optional[str] = None
    session_id: str


# ── Disease Detection ───────────────────────────────────────────

class DiseaseDetectResponse(BaseModel):
    disease_name: str
    confidence: float
    severity: str  # "low" | "medium" | "high"
    treatment_recommendations: list[str]
    prevention_tips: list[str]
    model_version: str


# ── Health ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict
    uptime_seconds: float
