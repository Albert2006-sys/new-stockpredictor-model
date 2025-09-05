-- init_db.sql

-- Create the table for minute-by-minute stock price data
CREATE TABLE stock_prices (
    timestamp TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    volume BIGINT
);

-- This is the TimescaleDB magic! Turn the regular table into a hypertable partitioned by time.
SELECT create_hypertable('stock_prices', 'timestamp');

-- Add an index for faster queries when filtering by ticker and time
CREATE INDEX ON stock_prices (ticker, timestamp DESC);

---
-- Create the table for news sentiment data
CREATE TABLE news_sentiment (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    headline TEXT,
    url VARCHAR(512) UNIQUE NOT NULL,
    source VARCHAR(100),
    published_at TIMESTAMPTZ NOT NULL,
    sentiment_score DOUBLE PRECISION
);

-- Add an index for faster queries
CREATE INDEX ON news_sentiment (ticker, published_at DESC);

---
-- Create the table for macroeconomic indicators
CREATE TABLE macro_indicators (
    date DATE NOT NULL,
    indicator VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (date, indicator)
);

---
-- Create a table to track processed corporate actions to prevent duplicate adjustments
CREATE TABLE processed_corporate_actions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    action_date DATE NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    details VARCHAR(255),
    processed_at TIMESTAMPTZ NOT NULL,
    UNIQUE(ticker, action_date, action_type)
);

\echo 'Database tables created successfully!'