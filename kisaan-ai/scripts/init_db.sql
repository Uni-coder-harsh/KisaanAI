-- KisaanAI Database Schema

-- Weather raw data
CREATE TABLE IF NOT EXISTS weather_raw (
    id SERIAL PRIMARY KEY,
    district TEXT NOT NULL,
    state TEXT,
    lat FLOAT,
    lon FLOAT,
    temperature_c FLOAT,
    feels_like_c FLOAT,
    humidity_pct FLOAT,
    pressure_hpa FLOAT,
    wind_speed_ms FLOAT,
    rainfall_mm FLOAT DEFAULT 0,
    weather_desc TEXT,
    fetched_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_weather_district ON weather_raw(district);
CREATE INDEX idx_weather_fetched_at ON weather_raw(fetched_at);

-- Feature store (offline)
CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    entity_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feature_entity ON feature_store(entity_id, feature_name);
CREATE INDEX idx_feature_created ON feature_store(created_at);

-- Prediction logs (for monitoring/drift detection)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    model_type TEXT NOT NULL,       -- 'crop' | 'yield' | 'disease'
    model_version TEXT,
    input_features JSONB,
    prediction TEXT,
    confidence FLOAT,
    latency_ms FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_pred_model ON prediction_logs(model_type, created_at);

-- Soil data
CREATE TABLE IF NOT EXISTS soil_data (
    id SERIAL PRIMARY KEY,
    district TEXT NOT NULL,
    state TEXT,
    nitrogen FLOAT,
    phosphorus FLOAT,
    potassium FLOAT,
    ph FLOAT,
    organic_carbon FLOAT,
    source TEXT,
    recorded_at DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Market prices
CREATE TABLE IF NOT EXISTS market_prices (
    id SERIAL PRIMARY KEY,
    crop TEXT NOT NULL,
    state TEXT,
    district TEXT,
    market TEXT,
    min_price FLOAT,
    max_price FLOAT,
    modal_price FLOAT,
    price_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_market_crop_date ON market_prices(crop, price_date);
