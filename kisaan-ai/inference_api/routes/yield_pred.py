from fastapi import APIRouter, Request, HTTPException
from inference_api.schemas.models import YieldPredictRequest, YieldPredictResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict_yield", response_model=YieldPredictResponse)
async def predict_yield(request: Request, body: YieldPredictRequest):
    """Predict crop yield in kg/ha given cultivation parameters."""
    try:
        model_service = request.app.state.models
        result = await model_service.predict_yield(body)
        return result
    except Exception as e:
        logger.error(f"Yield prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Yield prediction failed")
