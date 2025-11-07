-- Supervisor MCP Database Initialization
-- Creates database and schemas for supervisor-mcp

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE supervisor OWNER tradingview'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'supervisor')\gexec

-- Connect to supervisor database
\c supervisor

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schema for supervisor-mcp
CREATE SCHEMA IF NOT EXISTS supervisor;

-- Metrics table - technical metrics
CREATE TABLE IF NOT EXISTS supervisor.metrics (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_ts ON supervisor.metrics(ts);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON supervisor.metrics(name);
CREATE INDEX IF NOT EXISTS idx_metrics_tags ON supervisor.metrics USING GIN(tags);

-- Facts table - business facts
CREATE TABLE IF NOT EXISTS supervisor.facts (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    kind VARCHAR(255) NOT NULL,
    actor VARCHAR(255),
    correlation_id VARCHAR(255),
    payload JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_facts_ts ON supervisor.facts(ts);
CREATE INDEX IF NOT EXISTS idx_facts_kind ON supervisor.facts(kind);
CREATE INDEX IF NOT EXISTS idx_facts_actor ON supervisor.facts(actor);
CREATE INDEX IF NOT EXISTS idx_facts_correlation ON supervisor.facts(correlation_id);
CREATE INDEX IF NOT EXISTS idx_facts_payload ON supervisor.facts USING GIN(payload);

-- Aggregates table - pre-calculated aggregates
CREATE TABLE IF NOT EXISTS supervisor.aggregates (
    id BIGSERIAL PRIMARY KEY,
    window VARCHAR(50) NOT NULL,
    kind VARCHAR(100) NOT NULL,
    metrics JSONB DEFAULT '{}',
    facts_count INTEGER DEFAULT 0,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    calculated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(window, kind, period_start)
);

CREATE INDEX IF NOT EXISTS idx_aggregates_window ON supervisor.aggregates(window);
CREATE INDEX IF NOT EXISTS idx_aggregates_kind ON supervisor.aggregates(kind);
CREATE INDEX IF NOT EXISTS idx_aggregates_period ON supervisor.aggregates(period_start, period_end);

-- MCP Registry table
CREATE TABLE IF NOT EXISTS supervisor.mcp_registry (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    version VARCHAR(50) NOT NULL,
    protocol VARCHAR(50) NOT NULL,
    endpoint VARCHAR(500),
    capabilities JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'unknown',
    last_seen TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_registry_name ON supervisor.mcp_registry(name);
CREATE INDEX IF NOT EXISTS idx_registry_status ON supervisor.mcp_registry(status);

-- Health Status table
CREATE TABLE IF NOT EXISTS supervisor.health_status (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    response_time_ms DOUBLE PRECISION,
    error TEXT,
    last_check TIMESTAMP NOT NULL,
    uptime_seconds DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, last_check)
);

CREATE INDEX IF NOT EXISTS idx_health_name ON supervisor.health_status(name);
CREATE INDEX IF NOT EXISTS idx_health_status ON supervisor.health_status(status);
CREATE INDEX IF NOT EXISTS idx_health_last_check ON supervisor.health_status(last_check);

-- Alert Rules table
CREATE TABLE IF NOT EXISTS supervisor.alert_rules (
    id BIGSERIAL PRIMARY KEY,
    rule_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    condition TEXT NOT NULL,
    severity VARCHAR(50) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    cooldown_minutes INTEGER DEFAULT 5,
    actions JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alert_rules_rule_id ON supervisor.alert_rules(rule_id);
CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON supervisor.alert_rules(enabled);

-- Active Alerts table
CREATE TABLE IF NOT EXISTS supervisor.active_alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    rule_id VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    triggered_at TIMESTAMP NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_alert_id ON supervisor.active_alerts(alert_id);
CREATE INDEX IF NOT EXISTS idx_alerts_rule_id ON supervisor.active_alerts(rule_id);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON supervisor.active_alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON supervisor.active_alerts(severity);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION supervisor.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_mcp_registry_updated_at ON supervisor.mcp_registry;
CREATE TRIGGER update_mcp_registry_updated_at 
    BEFORE UPDATE ON supervisor.mcp_registry
    FOR EACH ROW EXECUTE FUNCTION supervisor.update_updated_at_column();

DROP TRIGGER IF EXISTS update_alert_rules_updated_at ON supervisor.alert_rules;
CREATE TRIGGER update_alert_rules_updated_at 
    BEFORE UPDATE ON supervisor.alert_rules
    FOR EACH ROW EXECUTE FUNCTION supervisor.update_updated_at_column();

-- Create user and grant permissions
-- NOTE: Password should be changed after initial setup using:
-- ALTER USER supervisor WITH PASSWORD 'your_secure_password';
-- For development, default password is 'supervisor' (change in production!)
CREATE USER IF NOT EXISTS supervisor WITH PASSWORD 'supervisor';
GRANT CONNECT ON DATABASE supervisor TO supervisor;
GRANT USAGE ON SCHEMA supervisor TO supervisor;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA supervisor TO supervisor;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA supervisor TO supervisor;

