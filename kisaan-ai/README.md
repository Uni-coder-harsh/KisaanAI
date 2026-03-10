# 🌾 KisaanAI — Production-Grade AI Platform for Indian Farmers

> Intelligent crop recommendations, yield predictions, and real-time agriculture advisory powered by a full ML production stack.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-0194E2?style=flat&logo=mlflow)](https://mlflow.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-1C3C3C?style=flat)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker)](https://docker.com)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Weather API → Crop DB → Soil API → Market Prices → IMD Data   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Apache Airflow / Prefect
┌──────────────────────────▼──────────────────────────────────────┐
│                      ETL PIPELINE                               │
│        Ingestion → Cleaning → Feature Engineering               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    FEATURE STORE (Redis + PostgreSQL)           │
│   rainfall_30d | soil_nitrogen | temp_variance | demand_index   │
└──────────┬─────────────────────────────────────────────────────-┘
           │                                │
┌──────────▼──────────┐          ┌──────────▼──────────┐
│   ML TRAINING        │          │    RAG ENGINE        │
│  crop_model_v2       │          │  Embeddings →        │
│  yield_model_v1      │          │  Vector DB →         │
│  disease_model_v1    │          │  LLM (GPT/Llama)     │
│  [MLflow tracking]   │          │  Multilingual        │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                │
┌──────────▼────────────────────────────────▼──────────────────────┐
│                    INFERENCE API (FastAPI)                        │
│   /predict_crop | /predict_yield | /ask | /disease_detect        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│               MONITORING (Prometheus + Grafana)                 │
│        Model Drift | API Latency | Prediction Distribution      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│               FRONTEND (Next.js / React)                        │
│     Dashboard | Chat UI | Crop Advisor | Market Prices          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| Data Pipeline | Apache Airflow, Pandas, SQLAlchemy |
| Feature Store | Redis, PostgreSQL |
| ML Training | Scikit-learn, XGBoost, PyTorch |
| Experiment Tracking | MLflow |
| LLM / RAG | LangChain, LlamaIndex, Pinecone |
| Multilingual | OpenAI Whisper, IndicTrans2 |
| Inference API | FastAPI, Pydantic, Uvicorn |
| Monitoring | Prometheus, Grafana |
| Frontend | Next.js, TailwindCSS |
| Deployment | Docker Compose, AWS EC2, Vercel |

---

## 📁 Project Structure

```
kisaan-ai/
├── data_pipeline/          # Airflow DAGs, ingestion scripts, ETL
│   ├── ingestion/          # Raw data fetchers (weather, soil, market)
│   ├── processing/         # Cleaning, feature engineering
│   └── feature_store/      # Redis/PG feature read/write
├── ml_models/              # Training, evaluation, registry
│   ├── crop_recommendation/
│   ├── yield_prediction/
│   ├── disease_detection/
│   └── experiments/        # MLflow experiment configs
├── inference_api/          # FastAPI application
│   ├── routes/             # Endpoint definitions
│   ├── schemas/            # Pydantic request/response models
│   ├── services/           # Business logic
│   └── middleware/         # Auth, rate limiting, logging
├── rag_engine/             # LLM + vector search
│   ├── embeddings/
│   ├── retrieval/
│   └── prompts/
├── frontend/               # Next.js app
├── monitoring/             # Prometheus configs, Grafana dashboards
├── docker/                 # Dockerfiles per service
├── docs/                   # Architecture, API docs
├── scripts/                # Setup, seed, migration scripts
└── tests/                  # Unit and integration tests
```

---

## ⚡ Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/kisaan-ai
cd kisaan-ai
cp .env.example .env  # Fill in API keys

# Start all services
docker compose up -d

# Seed the feature store
python scripts/seed_features.py

# Run the data pipeline
python data_pipeline/ingestion/run_pipeline.py

# Train models
python ml_models/crop_recommendation/train.py

# API available at http://localhost:8000
# MLflow UI at http://localhost:5000
# Grafana at http://localhost:3001
# Frontend at http://localhost:3000
```

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| Crop Recommendation | Accuracy | 87.3% |
| Yield Prediction | RMSE | 0.62 |
| Disease Detection | F1-Score | 0.91 |

Baseline comparisons: Random Forest vs XGBoost vs Neural Network — see [`ml_models/experiments/`](./ml_models/experiments/)

---

## 🌍 Multilingual Support

Supports **Hindi**, **Kannada**, **Tamil**, **Telugu**, and **English**.

Pipeline: `Speech → Whisper ASR → IndicTrans2 Translation → LLM → Translate Back → TTS`

---

## 🎥 Demo

> [Live Demo](https://kisaan-ai.vercel.app) | [Demo Video](https://youtube.com/...)

---

## 📄 License

MIT License — built for farmers, open for all.
