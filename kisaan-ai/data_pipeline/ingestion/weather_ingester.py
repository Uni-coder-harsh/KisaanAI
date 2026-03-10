"""
Weather data ingestion from OpenWeatherMap API.
Fetches current + forecast data for major Indian agricultural districts.
"""

import os
import logging
from datetime import datetime
from typing import Optional
import httpx
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DISTRICTS = [
    {"name": "Bangalore Rural", "lat": 13.00, "lon": 77.57, "state": "Karnataka"},
    {"name": "Pune", "lat": 18.52, "lon": 73.85, "state": "Maharashtra"},
    {"name": "Ludhiana", "lat": 30.90, "lon": 75.85, "state": "Punjab"},
    {"name": "Warangal", "lat": 17.98, "lon": 79.59, "state": "Telangana"},
    {"name": "Coimbatore", "lat": 11.00, "lon": 76.96, "state": "Tamil Nadu"},
]

BASE_URL = "https://api.openweathermap.org/data/2.5"


class WeatherIngester:
    def __init__(self, api_key: str, db_url: str):
        self.api_key = api_key
        self.engine = create_engine(db_url)

    async def fetch_current(self, lat: float, lon: float) -> dict:
        """Fetch current weather for a coordinate."""
        url = f"{BASE_URL}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()

    async def fetch_forecast(self, lat: float, lon: float) -> dict:
        """Fetch 5-day forecast for a coordinate."""
        url = f"{BASE_URL}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()

    def parse_weather(self, raw: dict, district: dict) -> dict:
        """Parse raw weather API response into flat record."""
        return {
            "district": district["name"],
            "state": district["state"],
            "lat": district["lat"],
            "lon": district["lon"],
            "temperature_c": raw["main"]["temp"],
            "feels_like_c": raw["main"]["feels_like"],
            "humidity_pct": raw["main"]["humidity"],
            "pressure_hpa": raw["main"]["pressure"],
            "wind_speed_ms": raw["wind"]["speed"],
            "rainfall_mm": raw.get("rain", {}).get("1h", 0.0),
            "weather_desc": raw["weather"][0]["description"],
            "fetched_at": datetime.utcnow(),
        }

    def save_to_db(self, records: list[dict]) -> None:
        """Persist weather records to PostgreSQL."""
        df = pd.DataFrame(records)
        df.to_sql(
            "weather_raw",
            self.engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        logger.info(f"Saved {len(records)} weather records to DB")

    async def run(self) -> list[dict]:
        """Main ingestion runner — fetches all districts."""
        records = []
        for district in DISTRICTS:
            try:
                raw = await self.fetch_current(district["lat"], district["lon"])
                record = self.parse_weather(raw, district)
                records.append(record)
                logger.info(f"Fetched weather for {district['name']}")
            except Exception as e:
                logger.error(f"Failed to fetch {district['name']}: {e}")
        self.save_to_db(records)
        return records


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    ingester = WeatherIngester(
        api_key=os.getenv("OPENWEATHER_API_KEY"),
        db_url=f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}",
    )
    asyncio.run(ingester.run())
