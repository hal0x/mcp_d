-- Supervisor MCP Database Schema
-- PostgreSQL schema for metrics, facts, aggregates, registry, and alerts

-- Metrics table - technical metrics
CREATE TABLE IF NOT EXISTS metrics (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_metrics_ts ON metrics(ts);
CREATE INDEX idx_metrics_name ON metrics(name);
CREATE INDEX idx_metrics_tags ON metrics USING GIN(tags);

-- Facts table - business facts
CREATE TABLE IF NOT EXISTS facts (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    kind VARCHAR(255) NOT NULL,
    actor VARCHAR(255),
    correlation_id VARCHAR(255),
    payload JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_facts_ts ON facts(ts);
CREATE INDEX idx_facts_kind ON facts(kind);
CREATE INDEX idx_facts_actor ON facts(actor);
CREATE INDEX idx_facts_correlation ON facts(correlation_id);
CREATE INDEX idx_facts_payload ON facts USING GIN(payload);

-- Aggregates table - pre-calculated aggregates
CREATE TABLE IF NOT EXISTS aggregates (
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

CREATE INDEX idx_aggregates_window ON aggregates(window);
CREATE INDEX idx_aggregates_kind ON aggregates(kind);
CREATE INDEX idx_aggregates_period ON aggregates(period_start, period_end);

-- MCP Registry table
CREATE TABLE IF NOT EXISTS mcp_registry (
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

CREATE INDEX idx_registry_name ON mcp_registry(name);
CREATE INDEX idx_registry_status ON mcp_registry(status);

-- Health Status table
CREATE TABLE IF NOT EXISTS health_status (
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

CREATE INDEX idx_health_name ON health_status(name);
CREATE INDEX idx_health_status ON health_status(status);
CREATE INDEX idx_health_last_check ON health_status(last_check);

-- Alert Rules table
CREATE TABLE IF NOT EXISTS alert_rules (
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

CREATE INDEX idx_alert_rules_rule_id ON alert_rules(rule_id);
CREATE INDEX idx_alert_rules_enabled ON alert_rules(enabled);

-- Active Alerts table
CREATE TABLE IF NOT EXISTS active_alerts (
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

CREATE INDEX idx_alerts_alert_id ON active_alerts(alert_id);
CREATE INDEX idx_alerts_rule_id ON active_alerts(rule_id);
CREATE INDEX idx_alerts_acknowledged ON active_alerts(acknowledged);
CREATE INDEX idx_alerts_severity ON active_alerts(severity);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_mcp_registry_updated_at 
    BEFORE UPDATE ON mcp_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alert_rules_updated_at 
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

