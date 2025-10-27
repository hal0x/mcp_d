-- Ensure dedicated role for HALv1 trading storage
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mcp') THEN
        CREATE ROLE mcp LOGIN PASSWORD 'mcp';
    END IF;
END
$$;

-- Create database if missing and assign ownership
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'mcp') THEN
        CREATE DATABASE mcp OWNER mcp;
    END IF;
END
$$;

GRANT ALL PRIVILEGES ON DATABASE mcp TO mcp;
