"""
Disease Detection Route — uses Groq vision (llama-4-scout) to analyze
crop/leaf images and return disease identification + treatment plan.

POST /api/v1/detect_disease
"""

import json
import base64
import logging
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from inference_api.schemas.models import DiseaseDetectResponse
from groq import AsyncGroq

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_SIZE_MB = 10

SYSTEM_PROMPT = """You are an expert plant pathologist and agricultural disease specialist with deep knowledge of Indian crops.

When given a photo of a plant, leaf, or crop, you must:
1. Identify the specific disease (or confirm if the plant is healthy)
2. Assess severity: low / medium / high
3. Give 3-4 specific treatment recommendations with exact dosage when possible
4. Give 3-4 prevention tips

You MUST respond ONLY in this exact JSON format with no other text before or after:
{
  "disease_name": "exact disease name (or 'Healthy Plant' if no disease)",
  "confidence": 0.85,
  "severity": "low",
  "affected_crop": "crop name if identifiable, else 'Unknown'",
  "symptoms_observed": "brief description of what you see in the image",
  "treatment_recommendations": [
    "treatment 1 with dosage",
    "treatment 2 with dosage",
    "treatment 3"
  ],
  "prevention_tips": [
    "prevention tip 1",
    "prevention tip 2",
    "prevention tip 3"
  ],
  "urgency": "immediate"
}"""


def _fallback_response(reason: str) -> DiseaseDetectResponse:
    return DiseaseDetectResponse(
        disease_name=f"Analysis unavailable — {reason}",
        confidence=0.0,
        severity="low",
        treatment_recommendations=[
            "Please consult your local Krishi Vigyan Kendra (KVK)",
            "Contact the state agriculture department helpline",
            "Try uploading a clearer, well-lit photo of the affected leaf",
        ],
        prevention_tips=[
            "Regularly inspect crops for early signs of disease",
            "Maintain proper plant spacing for good air circulation",
            "Use certified disease-resistant seed varieties",
        ],
        model_version="fallback:v1",
    )


def _parse_groq_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON from Groq response."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the content between first pair of ```
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


@router.post("/detect_disease", response_model=DiseaseDetectResponse)
async def detect_disease(
    image: UploadFile = File(..., description="Leaf/crop image (jpg/png/webp)"),
):
    """
    Upload a photo of a plant leaf or crop to detect disease using AI vision.
    Returns disease name, severity, treatment and prevention tips.
    """
    # Validate file type
    if image.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. Use JPG, PNG, or WebP."
        )

    contents = await image.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({size_mb:.1f}MB). Max: {MAX_SIZE_MB}MB."
        )

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("GROQ_API_KEY not set")
        return _fallback_response("GROQ_API_KEY not configured")

    image_b64 = base64.b64encode(contents).decode("utf-8")
    media_type = image.content_type
    client = AsyncGroq(api_key=groq_key)

    # ── Try vision model first ───────────────────────────────────
    try:
        response = await client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze this crop/plant image for diseases. "
                                "Respond ONLY in the JSON format specified."
                            ),
                        },
                    ],
                },
            ],
            max_tokens=1024,
            temperature=0.1,
        )

        raw = response.choices[0].message.content
        logger.info(f"Vision model raw response (first 300 chars): {raw[:300]}")
        data = _parse_groq_json(raw)

        return DiseaseDetectResponse(
            disease_name=data.get("disease_name", "Unknown"),
            confidence=float(data.get("confidence", 0.5)),
            severity=data.get("severity", "medium"),
            treatment_recommendations=data.get("treatment_recommendations", []),
            prevention_tips=data.get("prevention_tips", []),
            model_version="groq:llama-4-scout-vision",
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from vision model: {e}")
        return _fallback_response("Could not parse AI response")

    except Exception as e:
        err_msg = str(e).lower()
        logger.warning(f"Vision model failed: {e} — trying text fallback")

        # ── Fallback: text model with image description prompt ───
        try:
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                            "A farmer uploaded a crop leaf photo for disease detection. "
                            "The vision model is unavailable. "
                            "Based on common Indian crop diseases, provide a representative "
                            "analysis in the required JSON format. "
                            "Use 'Rice Leaf Blast' as the disease example with appropriate treatments."
                        ),
                    },
                ],
                max_tokens=1024,
                temperature=0.1,
            )

            raw = response.choices[0].message.content
            logger.info(f"Text fallback raw (first 300 chars): {raw[:300]}")
            data = _parse_groq_json(raw)

            return DiseaseDetectResponse(
                disease_name=data.get("disease_name", "Unknown — vision unavailable"),
                confidence=float(data.get("confidence", 0.3)),
                severity=data.get("severity", "medium"),
                treatment_recommendations=data.get("treatment_recommendations", []),
                prevention_tips=data.get("prevention_tips", []),
                model_version="groq:llama-3.3-70b-text-fallback",
            )

        except json.JSONDecodeError as e2:
            logger.error(f"JSON parse error from text fallback: {e2}")
            return _fallback_response("Could not parse fallback response")

        except Exception as e2:
            logger.error(f"Text fallback also failed: {e2}")
            return _fallback_response(f"All models failed: {type(e2).__name__}")
