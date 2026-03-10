# KisaanAI — Build Roadmap

A phased plan from zero to production-deployed AI platform.

---

## Phase 1 — Foundation (Week 1–2)

**Goal**: Get the database, pipeline, and API skeleton running locally.

- [ ] Clone repo, copy `.env.example` → `.env`, fill in keys
- [ ] `docker compose up -d postgres redis`
- [ ] Run `scripts/init_db.sql` to create tables
- [ ] Download the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) → `data_pipeline/datasets/crop_recommendation.csv`
- [ ] Run `data_pipeline/ingestion/weather_ingester.py` (set OPENWEATHER key)
- [ ] Verify data is in PostgreSQL
- [ ] Run `docker compose up -d api` and hit `GET /health`

---

## Phase 2 — ML Models (Week 2–3)

**Goal**: Train all three models and register them in MLflow.

- [ ] `docker compose up -d mlflow`
- [ ] `python ml_models/crop_recommendation/train.py`
  - Compares RF vs XGBoost vs Neural Net
  - Best model auto-registered in MLflow
- [ ] Download [Indian Agriculture Yield dataset](https://www.kaggle.com/datasets/srinivas1/agricuture-crops-production-in-india) → train `ml_models/yield_prediction/train.py`
- [ ] Download [Plant Disease dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) → train `ml_models/disease_detection/train.py`
- [ ] Open MLflow UI at `localhost:5000` and promote best models to `Production`

---

## Phase 3 — RAG Engine (Week 3–4)

**Goal**: Get the multilingual AI chat working.

- [ ] Create a Pinecone index `kisaan-ai` (free tier)
- [ ] Run `rag_engine/ingest_documents.py` to embed agriculture PDFs
  - Sources: ICAR documents, Krishi Vigyan Kendra guides, government schemes PDFs
- [ ] Test `POST /api/v1/ask` with a question
- [ ] Test multilingual: ask in Hindi, verify answer comes back in Hindi
- [ ] Add more documents to the vector store over time

---

## Phase 4 — Feature Store + Pipeline Automation (Week 4)

**Goal**: Features auto-refresh daily from real data.

- [ ] Verify `data_pipeline/feature_store/store.py` reads/writes Redis correctly
- [ ] Run `python data_pipeline/scheduler.py` — it should fetch weather + compute features
- [ ] Check Redis keys: `redis-cli keys "feature:*"`
- [ ] Verify `POST /predict_crop` with a `district` uses enriched features from store

---

## Phase 5 — Monitoring (Week 5)

**Goal**: Full observability stack running.

- [ ] `docker compose up -d prometheus grafana`
- [ ] Import Grafana dashboard from `monitoring/dashboards/`
- [ ] Verify API metrics appear at `localhost:9090`
- [ ] Set up alerts for:
  - API latency > 500ms
  - Prediction confidence consistently < 60% (model drift)
  - Error rate > 5%

---

## Phase 6 — Frontend (Week 5–6)

**Goal**: Beautiful UI that any farmer (or recruiter!) can use.

- [ ] `cd frontend && npm install && npm run dev`
- [ ] Pages to build:
  - `/` — Landing page with demo
  - `/advisor` — Crop recommendation form
  - `/chat` — Multilingual AI chat
  - `/disease` — Upload leaf image for disease detection
  - `/market` — Market price dashboard

---

## Phase 7 — Deployment (Week 6–7)

**Goal**: Everything live and publicly accessible.

- [ ] **Backend**: Deploy to AWS EC2 (t3.medium) with `docker compose`
- [ ] **MLflow**: Same EC2 or separate small instance
- [ ] **Frontend**: Deploy to Vercel (`vercel --prod`)
- [ ] **Database**: Upgrade to Supabase (free tier) for managed Postgres
- [ ] **Vector DB**: Pinecone (already cloud-native)
- [ ] Set up domain + HTTPS via Let's Encrypt or Cloudflare
- [ ] Add `LIVE DEMO` link to README

---

## Phase 8 — Polish for GitHub (Week 7)

**Goal**: Make the GitHub repo recruiter-ready.

- [ ] Record a 3-minute demo video (Loom or OBS)
- [ ] Add architecture diagram image to README (use Excalidraw or draw.io)
- [ ] Add model comparison table with real numbers
- [ ] Write a proper `CONTRIBUTING.md`
- [ ] Add GitHub Actions CI: lint + tests on every PR
- [ ] Pin the repo + add topics: `machine-learning`, `fastapi`, `mlops`, `agriculture`, `india`

---

## Key Datasets

| Dataset | Source |
|---|---|
| Crop Recommendation | Kaggle: atharvaingle/crop-recommendation-dataset |
| Yield Prediction | Kaggle: srinivas1/agricuture-crops-production-in-india |
| Plant Disease (CNN) | Kaggle: vipoooool/new-plant-diseases-dataset |
| Soil Data | ICAR / data.gov.in |
| Market Prices | agmarknet.gov.in (free API) |

---

## Architecture Decisions

**Why FastAPI over Flask?**
Async-native, Pydantic validation, automatic OpenAPI docs — production standard.

**Why MLflow over manual versioning?**
Reproducible experiments, model registry, artifact storage. Industry standard.

**Why Redis for feature store?**
Sub-millisecond reads for online inference. PostgreSQL as offline/historical backup.

**Why Pinecone over local vector DB?**
Cloud-native, scales without ops burden. Swap with Weaviate/Qdrant for fully self-hosted.

**Why Prefect over Airflow?**
Lighter for solo/small team projects. Easy local + cloud scheduling. Switch to Airflow if team grows.
