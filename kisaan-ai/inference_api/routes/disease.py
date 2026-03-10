"""
Disease detection endpoint — accepts an image upload (leaf photo)
and returns disease identification + treatment plan.
"""

import logging
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from inference_api.schemas.models import DiseaseDetectResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect_disease", response_model=DiseaseDetectResponse)
async def detect_disease(
    request: Request,
    image: UploadFile = File(..., description="Leaf/crop image"),
):
    """
    Upload a photo of a plant leaf or crop to detect disease.
    Returns disease name, severity, treatment recommendations.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted")

    try:
        contents = await image.read()
        # model_service = request.app.state.models
        # result = await model_service.predict_disease(contents)

        # Mock response for scaffold
        return DiseaseDetectResponse(
            disease_name="Leaf Blast (Magnaporthe oryzae)",
            confidence=0.92,
            severity="medium",
            treatment_recommendations=[
                "Apply Tricyclazole 75% WP @ 0.6g/L of water",
                "Spray Carbendazim 50% WP @ 1g/L as a preventive measure",
                "Avoid water stress and excessive nitrogen application",
            ],
            prevention_tips=[
                "Use resistant varieties like IR-64 or Swarna",
                "Maintain proper spacing between plants for airflow",
                "Avoid late evening irrigation",
            ],
            model_version="disease_cnn_v1",
        )
    except Exception as e:
        logger.error(f"Disease detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Disease detection failed")
