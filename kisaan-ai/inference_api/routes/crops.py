"""
Crop recommendation endpoint.
POST /api/v1/predict_crop
"""

import logging
from fastapi import APIRouter, Request, HTTPException
from inference_api.schemas.models import CropPredictRequest, CropPredictResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict_crop", response_model=CropPredictResponse)
async def predict_crop(request: Request, body: CropPredictRequest):
    """
    Predict the best crop to grow given soil and climate parameters.
    
    Optionally provide a `district` to automatically enrich features
    from the feature store (rainfall_30d, humidity_avg_7d, etc).
    """
    try:
        model_service = request.app.state.models
        result = await model_service.predict_crop(body)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Crop prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/crops/supported")
async def list_supported_crops():
    """Returns list of all crops the model can recommend."""
    return {
        "crops": [
            "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
            "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
            "banana", "mango", "grapes", "watermelon", "muskmelon",
            "apple", "orange", "papaya", "coconut", "cotton",
            "jute", "coffee"
        ]
    }
