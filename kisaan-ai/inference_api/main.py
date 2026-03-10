"""
KisaanAI Inference API
FastAPI application serving crop recommendation, yield prediction,
RAG-powered Q&A, and disease detection endpoints.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from inference_api.routes import crops, yield_pred, chat, disease, health
from inference_api.middleware.rate_limiter import RateLimitMiddleware
from inference_api.middleware.logging import RequestLoggingMiddleware

logger = logging.getLogger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────
REQUEST_COUNT = Counter(
    "kisaan_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "kisaan_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "kisaan_predictions_total",
    "Total ML predictions made",
    ["model_type"],
)


# ── App lifespan ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌾 KisaanAI API starting up...")
    # Load models into memory on startup
    from inference_api.services.model_service import ModelService
    app.state.models = ModelService()
    await app.state.models.load_all()
    logger.info("✅ All models loaded.")
    yield
    logger.info("KisaanAI API shutting down.")


# ── App factory ─────────────────────────────────────────────────
app = FastAPI(
    title="KisaanAI API",
    description="Production ML platform for Indian agriculture",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(RequestLoggingMiddleware)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(duration)
    return response


# ── Routers ─────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(crops.router, prefix="/api/v1", tags=["Crop Recommendation"])
app.include_router(yield_pred.router, prefix="/api/v1", tags=["Yield Prediction"])
app.include_router(chat.router, prefix="/api/v1", tags=["AI Chat / RAG"])
app.include_router(disease.router, prefix="/api/v1", tags=["Disease Detection"])


# ── Prometheus scrape endpoint ───────────────────────────────────
@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
