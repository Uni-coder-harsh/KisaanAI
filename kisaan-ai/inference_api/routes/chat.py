"""
RAG-powered chat endpoint.
POST /api/v1/ask
Supports multilingual queries (Hindi, Kannada, Tamil, Telugu, English).
"""

import uuid
import logging
from fastapi import APIRouter, Request, HTTPException
from inference_api.schemas.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=ChatResponse)
async def ask(request: Request, body: ChatRequest):
    """
    Ask the AI agriculture advisor a question.
    
    Pipeline:
    1. Detect / translate input language → English
    2. Embed question → search Pinecone vector DB
    3. Retrieve top-k agriculture documents
    4. LLM generates grounded answer
    5. Translate answer back to requested language
    """
    try:
        rag_service = request.app.state.models.rag
        session_id = body.session_id or str(uuid.uuid4())
        result = await rag_service.answer(body, session_id)
        return result
    except Exception as e:
        logger.error(f"RAG chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat failed")


@router.get("/ask/history/{session_id}")
async def get_history(session_id: str, request: Request):
    """Retrieve conversation history for a session."""
    rag_service = request.app.state.models.rag
    history = await rag_service.get_history(session_id)
    return {"session_id": session_id, "messages": history}
