-- Create database and user for policy-mcp
CREATE DATABASE policy;
CREATE USER policy WITH PASSWORD 'policy';
GRANT ALL PRIVILEGES ON DATABASE policy TO policy;

-- Connect to policy database and grant schema privileges
\c policy
GRANT ALL ON SCHEMA public TO policy;

