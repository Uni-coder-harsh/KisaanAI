"""
Feature Store — read/write agricultural features using Redis (online)
and PostgreSQL (offline/historical).

Features tracked:
  - rainfall_30d        : mm of rain in last 30 days
  - soil_nitrogen       : soil nitrogen level (kg/ha)
  - soil_phosphorus     : soil phosphorus (kg/ha)
  - soil_potassium      : soil potassium (kg/ha)
  - temp_variance_7d    : temperature variance over 7 days
  - humidity_avg_7d     : avg humidity last 7 days
  - crop_demand_index   : market demand score (0–1) for crops
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import redis
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, redis_url: str, db_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.engine = create_engine(db_url)
        self.ttl_seconds = 60 * 60 * 6  # 6-hour cache

    # ── Online Store (Redis) ────────────────────────────────────

    def _feature_key(self, entity_id: str, feature_name: str) -> str:
        return f"feature:{entity_id}:{feature_name}"

    def set_feature(self, entity_id: str, feature_name: str, value: Any) -> None:
        """Write a single feature to Redis with TTL."""
        key = self._feature_key(entity_id, feature_name)
        self.redis.setex(key, self.ttl_seconds, json.dumps(value))

    def get_feature(self, entity_id: str, feature_name: str) -> Optional[Any]:
        """Read a single feature from Redis (cache)."""
        key = self._feature_key(entity_id, feature_name)
        raw = self.redis.get(key)
        return json.loads(raw) if raw else None

    def get_feature_vector(self, entity_id: str, feature_names: list[str]) -> dict:
        """
        Fetch a full feature vector for inference.
        Falls back to offline store if not in Redis.
        """
        vector = {}
        missing = []
        for name in feature_names:
            val = self.get_feature(entity_id, name)
            if val is not None:
                vector[name] = val
            else:
                missing.append(name)

        if missing:
            logger.info(f"Cache miss for {entity_id}: {missing}. Fetching from offline store.")
            offline = self._fetch_offline(entity_id, missing)
            vector.update(offline)
            # Backfill Redis
            for name, val in offline.items():
                self.set_feature(entity_id, name, val)

        return vector

    def set_feature_vector(self, entity_id: str, features: dict) -> None:
        """Write multiple features to Redis at once."""
        pipe = self.redis.pipeline()
        for name, value in features.items():
            key = self._feature_key(entity_id, name)
            pipe.setex(key, self.ttl_seconds, json.dumps(value))
        pipe.execute()

    # ── Offline Store (PostgreSQL) ──────────────────────────────

    def _fetch_offline(self, entity_id: str, feature_names: list[str]) -> dict:
        """Pull latest features from PostgreSQL."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT feature_name, feature_value
                    FROM feature_store
                    WHERE entity_id = :entity_id
                      AND feature_name = ANY(:names)
                    ORDER BY created_at DESC
                """),
                {"entity_id": entity_id, "names": feature_names},
            )
            rows = result.fetchall()
        return {row.feature_name: json.loads(row.feature_value) for row in rows}

    def persist_features(self, entity_id: str, features: dict) -> None:
        """Persist features to PostgreSQL for historical tracking."""
        records = [
            {
                "entity_id": entity_id,
                "feature_name": name,
                "feature_value": json.dumps(value),
                "created_at": datetime.utcnow(),
            }
            for name, value in features.items()
        ]
        df = pd.DataFrame(records)
        df.to_sql("feature_store", self.engine, if_exists="append", index=False, method="multi")
        logger.info(f"Persisted {len(records)} features for entity {entity_id}")

    # ── Compute + Store Pipeline Features ──────────────────────

    def compute_district_features(self, district: str, db_url: str) -> dict:
        """
        Compute rolling aggregates from raw weather + soil tables
        and store into the feature store.
        """
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Rainfall last 30 days
            rainfall = conn.execute(
                text("""
                    SELECT COALESCE(SUM(rainfall_mm), 0) as total
                    FROM weather_raw
                    WHERE district = :district
                      AND fetched_at >= NOW() - INTERVAL '30 days'
                """),
                {"district": district},
            ).scalar()

            # Temp variance last 7 days
            temp_variance = conn.execute(
                text("""
                    SELECT COALESCE(VARIANCE(temperature_c), 0) as variance
                    FROM weather_raw
                    WHERE district = :district
                      AND fetched_at >= NOW() - INTERVAL '7 days'
                """),
                {"district": district},
            ).scalar()

            # Avg humidity last 7 days
            humidity_avg = conn.execute(
                text("""
                    SELECT COALESCE(AVG(humidity_pct), 0) as avg_humidity
                    FROM weather_raw
                    WHERE district = :district
                      AND fetched_at >= NOW() - INTERVAL '7 days'
                """),
                {"district": district},
            ).scalar()

        features = {
            "rainfall_30d": float(rainfall),
            "temp_variance_7d": float(temp_variance),
            "humidity_avg_7d": float(humidity_avg),
        }

        entity_id = f"district:{district}"
        self.set_feature_vector(entity_id, features)
        self.persist_features(entity_id, features)
        logger.info(f"Computed + stored features for {district}: {features}")
        return features
