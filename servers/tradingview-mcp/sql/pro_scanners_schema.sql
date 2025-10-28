-- Schema for professional scanners persistence

CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    entry NUMERIC(20, 8),
    sl NUMERIC(20, 8),
    tp JSONB,
    leverage NUMERIC(10, 2),
    confidence INT,
    reasons JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    signal_id INT REFERENCES signals(id) ON DELETE CASCADE,
    action VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    period VARCHAR(10) NOT NULL,
    strategy VARCHAR(50),
    total_signals INT,
    take_rate NUMERIC(5, 2),
    avg_confidence NUMERIC(5, 2),
    avg_leverage NUMERIC(5, 2),
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS backtests (
    id SERIAL PRIMARY KEY,
    strategy VARCHAR(50),
    profile VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    symbol_universe JSONB NOT NULL,
    metrics JSONB,
    generated_signals JSONB,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_feedback_signal ON feedback(signal_id);
CREATE INDEX IF NOT EXISTS idx_metrics_calculated ON metrics(calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtests_strategy ON backtests(strategy);
