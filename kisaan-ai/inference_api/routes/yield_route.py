"""
Yield Prediction Route
POST /api/v1/predict_yield

Request:
  {
    "crop":     "Maize",
    "country":  "India",
    "year":     2024,
    "rainfall": 1200.0,
    "pesticides": 50.0,
    "avg_temp": 25.0
  }

Response:
  {
    "crop": "Maize",
    "country": "India",
    "predicted_yield_hg_ha": 45230,
    "predicted_yield_tonnes_ha": 4.52,
    "yield_category": "High",
    "confidence_note": "...",
    "model_version": "local:v1"
  }
"""

import os
import json
import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

ARTIFACTS = "ml_models/yield_prediction/artifacts"

# ── Load model at startup ────────────────────────────────────────
_yield_model   = None
_area_encoder  = None
_item_encoder  = None
_yield_scaler  = None
_yield_meta    = None


def load_yield_model():
    global _yield_model, _area_encoder, _item_encoder, _yield_scaler, _yield_meta
    try:
        import joblib
        _yield_model  = joblib.load(f"{ARTIFACTS}/yield_model.pkl")
        _area_encoder = joblib.load(f"{ARTIFACTS}/area_encoder.pkl")
        _item_encoder = joblib.load(f"{ARTIFACTS}/item_encoder.pkl")
        _yield_scaler = joblib.load(f"{ARTIFACTS}/yield_scaler.pkl")
        with open(f"{ARTIFACTS}/yield_metadata.json") as f:
            _yield_meta = json.load(f)
        logger.info(f"Yield model loaded: {_yield_meta['best_model']} (R²={_yield_meta['r2']:.4f})")
        return True
    except Exception as e:
        logger.warning(f"Could not load yield model: {e}. Using mock.")
        return False


# Try loading on import
load_yield_model()


# ── Request / Response schemas ───────────────────────────────────
class YieldRequest(BaseModel):
    crop:       str   = Field(..., example="Maize")
    country:    str   = Field(..., example="India")
    year:       int   = Field(2024, ge=1990, le=2030)
    rainfall:   float = Field(..., ge=0,   le=5000,  example=1200.0, description="mm/year")
    pesticides: float = Field(..., ge=0,   le=10000, example=50.0,   description="tonnes")
    avg_temp:   float = Field(..., ge=-10, le=50,    example=25.0,   description="Celsius")


class YieldResponse(BaseModel):
    crop:                     str
    country:                  str
    year:                     int
    predicted_yield_hg_ha:    int
    predicted_yield_tonnes_ha: float
    yield_category:           str
    benchmark_note:           str
    model_version:            str
    supported_crops:          Optional[list] = None


def _yield_category(hg_ha: float) -> str:
    """Categorize yield into Low / Medium / High / Exceptional."""
    tonne = hg_ha / 10_000
    if tonne < 1.0:   return "Low"
    if tonne < 3.0:   return "Medium"
    if tonne < 6.0:   return "High"
    return "Exceptional"


def _mock_predict(req: YieldRequest) -> dict:
    """Fallback when model isn't trained yet."""
    base = {
        "maize": 45000, "wheat": 32000, "rice": 38000,
        "potatoes": 180000, "soybean": 28000, "cotton": 15000,
    }
    hg_ha = base.get(req.crop.lower(), 40000)
    # Adjust by rainfall and temp
    hg_ha = int(hg_ha * (0.8 + req.rainfall / 5000) * (0.9 + min(req.avg_temp, 30) / 100))
    return {
        "predicted_yield_hg_ha": hg_ha,
        "model_version": "mock:v1",
        "supported_crops": None,
    }


# ── Route ────────────────────────────────────────────────────────
@router.post("/predict_yield", response_model=YieldResponse)
async def predict_yield(req: YieldRequest):
    """Predict crop yield in hg/ha and tonnes/ha."""

    if _yield_model is None:
        # Use mock predictor
        mock = _mock_predict(req)
        hg_ha = mock["predicted_yield_hg_ha"]
        tonnes_ha = round(hg_ha / 10_000, 2)
        return YieldResponse(
            crop=req.crop,
            country=req.country,
            year=req.year,
            predicted_yield_hg_ha=hg_ha,
            predicted_yield_tonnes_ha=tonnes_ha,
            yield_category=_yield_category(hg_ha),
            benchmark_note="Mock prediction — train yield model for accurate results.",
            model_version="mock:v1",
            supported_crops=None,
        )

    # Encode area
    try:
        area_enc = _area_encoder.transform([req.country])[0]
    except ValueError:
        # Unknown country — use mean encoding
        area_enc = len(_area_encoder.classes_) // 2
        logger.warning(f"Unknown country '{req.country}', using fallback encoding.")

    # Encode crop
    try:
        item_enc = _item_encoder.transform([req.crop])[0]
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Crop '{req.crop}' not in training data. "
                   f"Supported: {list(_item_encoder.classes_)}"
        )

    # Build feature vector
    features = np.array([[
        area_enc,
        item_enc,
        req.year,
        req.rainfall,
        req.pesticides,
        req.avg_temp,
    ]])
    features_scaled = _yield_scaler.transform(features)

    hg_ha     = int(_yield_model.predict(features_scaled)[0])
    hg_ha     = max(0, hg_ha)  # clamp to non-negative
    tonnes_ha = round(hg_ha / 10_000, 2)

    meta = _yield_meta or {}
    mean_yield = meta.get("yield_mean", 40000)
    pct_diff   = ((hg_ha - mean_yield) / mean_yield) * 100
    if pct_diff >= 0:
        note = f"{pct_diff:.0f}% above global average ({mean_yield/10000:.1f} t/ha)"
    else:
        note = f"{abs(pct_diff):.0f}% below global average ({mean_yield/10000:.1f} t/ha)"

    return YieldResponse(
        crop=req.crop,
        country=req.country,
        year=req.year,
        predicted_yield_hg_ha=hg_ha,
        predicted_yield_tonnes_ha=tonnes_ha,
        yield_category=_yield_category(hg_ha),
        benchmark_note=note,
        model_version=f"local:v1 ({meta.get('best_model','rf')})",
        supported_crops=list(_item_encoder.classes_) if _item_encoder else None,
    )


@router.get("/yield_crops")
async def get_supported_crops():
    """List all crops and countries the yield model supports."""
    if _yield_meta is None:
        return {"status": "model_not_trained", "crops": [], "countries": []}
    return {
        "status": "ok",
        "crops": _yield_meta.get("crops", []),
        "countries": _yield_meta.get("areas", []),
        "model": _yield_meta.get("best_model"),
        "r2_score": _yield_meta.get("r2"),
    }
