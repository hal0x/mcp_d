# Policy MCP Server

Decision profiles and policy management for the MCP ecosystem. Uses PostgreSQL for durable storage of profiles and their version history.

## Features

- **Decision Profiles**: Create and manage decision profiles
- **Policy Rules**: Define and evaluate policy rules
- **Decision Evaluation**: Evaluate decisions against active profiles
- **Integration**: Works with learning-mcp for adaptive policies

## Quick Start

```bash
cp .env.example .env        # обновите при необходимости креды БД
python -m policy_mcp.server --stdio
# или HTTP-режим
python -m policy_mcp.server --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `POLICY_DB_URL` | PostgreSQL connection URL | `postgresql+asyncpg://policy:policy@localhost:5432/policy` |
| `POLICY_DB_POOL_SIZE` | (Optional) Primary connection pool size | `None` |
| `POLICY_DB_MAX_OVERFLOW` | (Optional) Max overflow connections | `None` |
| `POLICY_DB_ECHO` | Echo SQL statements for debug | `false` |
| `POLICY_LOG_LEVEL` | Logging level | `INFO` |

## Tools

### Profile Management
- `create_profile` - Create a new decision profile
- `get_profile` - Get a decision profile by ID
- `get_active_profile` - Fetch the latest active profile
- `list_profiles` - List all decision profiles
- `update_profile` - Update a decision profile
- `activate_profile` - Mark a profile as active
- `delete_profile` - Delete a decision profile
- `list_profile_versions` - Show historical versions of a profile
- `rollback_profile` - Restore profile state from a previous version
- `configure_profile_experiment` - Attach A/B experiment metadata to a profile
- `list_profile_experiments` - List configured experiments across profiles

### Decision Evaluation
- `evaluate_decision` - Evaluate a decision request under a policy

## REST API

Полезно для интеграции с learning-mcp и другими сервисами:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check service status and active profile count |
| `GET` | `/profiles` | List profiles (`?active_only=true`) |
| `GET` | `/profiles/{profile_id}` | Retrieve profile by id |
| `GET` | `/profiles/active` | Get the latest active profile (optional `profile_id`) |
| `POST` | `/profiles` | Create/update profile payload (`activate=true` to immediately activate) |
| `POST` | `/profiles/{profile_id}/activate` | Activate an existing profile |
| `GET` | `/profiles/{profile_id}/versions` | List profile version history |
| `POST` | `/profiles/{profile_id}/rollback` | Rollback profile to a stored version |
| `POST` | `/profiles/{profile_id}/experiment` | Configure A/B experiment metadata |
| `GET` | `/profiles/experiments` | List experiments across profiles |
| `DELETE` | `/profiles/{profile_id}` | Delete profile |

## Development

```bash
# Install
pip install -e .

# Tests
pytest tests/

# Linting
black .
ruff check .
mypy .
```

## License

MIT License
