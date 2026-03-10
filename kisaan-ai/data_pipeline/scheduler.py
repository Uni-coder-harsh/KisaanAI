"""
KisaanAI Data Pipeline Scheduler (Prefect)
Runs the full ingestion + feature computation pipeline on a schedule.
"""

import os
import asyncio
import logging
from prefect import flow, task
from prefect.schedules import CronSchedule
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT', 6379)}"


@task(retries=3, retry_delay_seconds=60)
async def ingest_weather():
    from data_pipeline.ingestion.weather_ingester import WeatherIngester
    ingester = WeatherIngester(
        api_key=os.getenv("OPENWEATHER_API_KEY"),
        db_url=DB_URL,
    )
    records = await ingester.run()
    logger.info(f"Ingested {len(records)} weather records")
    return len(records)


@task(retries=2)
def compute_features():
    from data_pipeline.feature_store.store import FeatureStore

    districts = [
        "Bangalore Rural", "Pune", "Ludhiana",
        "Warangal", "Coimbatore",
    ]
    store = FeatureStore(redis_url=REDIS_URL, db_url=DB_URL)
    for district in districts:
        try:
            features = store.compute_district_features(district, DB_URL)
            logger.info(f"Features computed for {district}: {features}")
        except Exception as e:
            logger.error(f"Feature compute failed for {district}: {e}")


@task
def log_pipeline_run(weather_count: int):
    logger.info(f"Pipeline complete: {weather_count} weather records, features refreshed")


@flow(name="kisaan_ai_daily_pipeline")
async def daily_pipeline():
    """Full daily data pipeline."""
    logger.info("🌾 Starting KisaanAI daily pipeline")
    weather_count = await ingest_weather()
    compute_features()
    log_pipeline_run(weather_count)
    logger.info("✅ Pipeline complete")


if __name__ == "__main__":
    # Run once immediately, then on schedule
    asyncio.run(daily_pipeline())
