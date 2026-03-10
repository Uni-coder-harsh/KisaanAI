"""
RAG Engine — Retrieval Augmented Generation for agriculture Q&A.

Pipeline:
  1. Translate non-English queries to English (IndicTrans2 / Google Translate)
  2. Embed query via OpenAI embeddings
  3. Search Pinecone vector index for top-k relevant documents
  4. Construct prompt with retrieved context
  5. LLM generates grounded answer
  6. Translate answer back to user's language
"""

import os
import logging
import uuid
from typing import Optional
import redis
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are KisaanGPT, an expert AI agriculture advisor for Indian farmers.
You have deep knowledge of:
- Crop cultivation practices for Indian climates
- Soil health and nutrient management
- Pest and disease identification and treatment
- Government schemes and subsidies for farmers (PM-KISAN, PMFBY, etc.)
- Market prices and MSP (Minimum Support Price)
- Organic and sustainable farming practices

Always provide practical, actionable advice. When recommending treatments,
prefer locally available and cost-effective options. Always mention safety precautions
when discussing pesticides. Be empathetic to the farmer's situation.

Context from agriculture knowledge base:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"],
)


class RAGService:
    def __init__(self, openai_key: str, pinecone_key: str, pinecone_index: str, redis_url: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=openai_key,
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=pinecone_index,
            embedding=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        self.redis = redis.from_url(redis_url, decode_responses=True)

    async def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English if not already English."""
        if source_lang == "en":
            return text
        # Use LLM for translation (replace with IndicTrans2 in production)
        msg = HumanMessage(
            content=f"Translate this {source_lang} text to English. Return only the translation, nothing else:\n{text}"
        )
        result = await self.llm.ainvoke([msg])
        return result.content.strip()

    async def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English answer back to target language."""
        if target_lang == "en":
            return text
        lang_names = {"hi": "Hindi", "kn": "Kannada", "ta": "Tamil", "te": "Telugu"}
        lang_name = lang_names.get(target_lang, "Hindi")
        msg = HumanMessage(
            content=f"Translate this English text to {lang_name}. Return only the translation:\n{text}"
        )
        result = await self.llm.ainvoke([msg])
        return result.content.strip()

    def _history_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    async def get_history(self, session_id: str) -> list:
        raw = self.redis.get(self._history_key(session_id))
        return json.loads(raw) if raw else []

    async def save_history(self, session_id: str, messages: list) -> None:
        self.redis.setex(
            self._history_key(session_id),
            60 * 60 * 24,  # 24h TTL
            json.dumps(messages),
        )

    async def answer(self, request, session_id: str) -> dict:
        """Full RAG pipeline: translate → retrieve → generate → translate back."""
        # Step 1: Translate to English
        english_q = await self.translate_to_english(request.question, request.language)
        logger.info(f"[{session_id}] Q (en): {english_q[:100]}")

        # Step 2 + 3 + 4 + 5: RAG chain
        result = await self.chain.ainvoke({"query": english_q})
        english_answer = result["result"]
        sources = [
            {
                "title": doc.metadata.get("title", "Agriculture Document"),
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0),
            }
            for doc in result.get("source_documents", [])
        ]

        # Step 6: Translate answer back
        final_answer = await self.translate_from_english(english_answer, request.language)

        # Save history
        history = await self.get_history(session_id)
        history.extend([
            {"role": "user", "content": request.question},
            {"role": "assistant", "content": final_answer},
        ])
        await self.save_history(session_id, history[-20:])  # Keep last 10 turns

        return {
            "answer": final_answer,
            "sources": sources,
            "language": request.language,
            "translated_question": english_q if request.language != "en" else None,
            "session_id": session_id,
        }
