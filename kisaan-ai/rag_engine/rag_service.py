"""
RAG Engine — uses Groq (free) for LLM. No OpenAI required.
"""

import os
import logging
import json

import redis
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are KisaanGPT, an expert AI agriculture advisor for Indian farmers.
You have deep knowledge of:
- Crop cultivation practices for Indian climates
- Soil health and nutrient management
- Pest and disease identification and treatment
- Government schemes and subsidies for farmers (PM-KISAN, PMFBY, etc.)
- Market prices and MSP (Minimum Support Price)
- Organic and sustainable farming practices

Always provide practical, actionable advice. When recommending treatments,
prefer locally available and cost-effective options. Always mention safety
precautions when discussing pesticides. Be empathetic to the farmer's situation."""),
    ("human", "{input}"),
])


class RAGService:
    def __init__(self, openai_key: str, pinecone_key: str, pinecone_index: str, redis_url: str):
        groq_key = os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.3,
            groq_api_key=groq_key,
        )

        self.chain = (
            {"input": RunnablePassthrough()}
            | SYSTEM_PROMPT
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized with Groq (LLM-only mode)")

        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            logger.info("Redis connected for chat history")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Chat history disabled.")
            self.redis = None

    async def translate_to_english(self, text: str, source_lang: str) -> str:
        if source_lang == "en":
            return text
        try:
            msg = HumanMessage(
                content=f"Translate this {source_lang} text to English. Return only the translation, nothing else:\n{text}"
            )
            result = await self.llm.ainvoke([msg])
            return result.content.strip()
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            return text

    async def translate_from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text
        lang_names = {"hi": "Hindi", "kn": "Kannada", "ta": "Tamil", "te": "Telugu"}
        lang_name = lang_names.get(target_lang, "Hindi")
        try:
            msg = HumanMessage(
                content=f"Translate this English text to {lang_name}. Return only the translation:\n{text}"
            )
            result = await self.llm.ainvoke([msg])
            return result.content.strip()
        except Exception as e:
            logger.error(f"Translation from English failed: {e}")
            return text

    def _history_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    async def get_history(self, session_id: str) -> list:
        if not self.redis:
            return []
        try:
            raw = self.redis.get(self._history_key(session_id))
            return json.loads(raw) if raw else []
        except Exception:
            return []

    async def save_history(self, session_id: str, messages: list) -> None:
        if not self.redis:
            return
        try:
            self.redis.setex(
                self._history_key(session_id),
                60 * 60 * 24,
                json.dumps(messages),
            )
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")

    async def answer(self, request, session_id: str) -> dict:
        english_q = await self.translate_to_english(request.question, request.language)
        logger.info(f"[{session_id}] Q (en): {english_q[:100]}")

        try:
            english_answer = await self.chain.ainvoke(english_q)
        except Exception as e:
            logger.error(f"Chain invocation failed: {e}")
            english_answer = "I'm sorry, I'm having trouble answering right now. Please try again in a moment."

        final_answer = await self.translate_from_english(english_answer, request.language)

        history = await self.get_history(session_id)
        history.extend([
            {"role": "user", "content": request.question},
            {"role": "assistant", "content": final_answer},
        ])
        await self.save_history(session_id, history[-20:])

        return {
            "answer": final_answer,
            "sources": [],
            "language": request.language,
            "translated_question": english_q if request.language != "en" else None,
            "session_id": session_id,
        }
