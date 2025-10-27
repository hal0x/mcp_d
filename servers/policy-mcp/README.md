# Policy MCP Server

Decision profiles and policy management for the MCP ecosystem.

## Features

- **Decision Profiles**: Create and manage decision profiles
- **Policy Rules**: Define and evaluate policy rules
- **Decision Evaluation**: Evaluate decisions against active profiles
- **Integration**: Works with learning-mcp for adaptive policies

## Quick Start

```bash
# Stdio mode
python -m policy_mcp.server --stdio

# HTTP mode
python -m policy_mcp.server --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `POLICY_DB_URL` | PostgreSQL connection URL | `postgresql+asyncpg://policy:policy@localhost:5432/policy` |
| `POLICY_LOG_LEVEL` | Logging level | `INFO` |

## Tools

### Profile Management
- `create_profile` - Create a new decision profile
- `get_profile` - Get a decision profile by ID
- `list_profiles` - List all decision profiles
- `update_profile` - Update a decision profile
- `delete_profile` - Delete a decision profile

### Decision Evaluation
- `evaluate_decision` - Evaluate a decision request under a policy

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

