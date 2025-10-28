# Orchestrator MCP

Тонкий оркестратор, который координирует выполнение планов между MCP-сервисами.

## Возможности

- **Планы**: приём последовательностей шагов (analyze → decide → execute).
- **Dry-run**: симуляция без фактического выполнения.
- **Rollback**: откат шагов при ошибках.
- **Integration**: общение с Supervisor/Policy/Learning для мониторинга и принятия решений.

## Quick Start

```bash
cp .env.example .env
python -m orchestrator_mcp.server --stdio
# или HTTP режим
python -m orchestrator_mcp.server --host 0.0.0.0 --port 8010

# Пример плана (JSON)
[
  {"action": "fetch_supervisor_health"},
  {"action": "check_policy_profile", "params": {"profile_id": "demo"}},
  {"action": "trigger_learning_online"}
]
```

## Tools

- `execute_plan` — выполнить план действий (принимает список шагов).
- `dry_run_plan` — симулировать план без сайд-эффектов.

### Поддерживаемые действия

- `fetch_supervisor_health`
- `check_policy_profile`
- `activate_policy_profile`
- `configure_policy_experiment`
- `list_policy_profiles`
- `list_policy_experiments`
- `trigger_learning_online`
- `list_learning_policy_profiles`

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ORCH_SUPERVISOR_URL` | Supervisor MCP base URL | `http://localhost:8001` |
| `ORCH_POLICY_URL` | Policy MCP base URL | `http://localhost:8002` |
| `ORCH_LEARNING_URL` | Learning MCP base URL | `http://localhost:8003` |
| `ORCH_DEFAULT_TIMEOUT` | Timeout для внешних вызовов | `30` |
| `ORCH_LOG_LEVEL` | Log level | `INFO` |

## Development

```bash
pip install -e .[dev]
pytest
black .
ruff check .
mypy .
```

## License

MIT License
