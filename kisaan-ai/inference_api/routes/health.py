from fastapi import APIRouter, Request
from inference_api.schemas.models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    status = request.app.state.models.get_status()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded=status["models_loaded"],
        uptime_seconds=status["uptime_seconds"],
    )
