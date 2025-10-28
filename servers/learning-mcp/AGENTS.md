# Learning MCP — Архитектура и Руководство

## Назначение

Learning MCP анализирует исторические метрики/факты из Supervisor MCP и формирует decision profiles для Policy MCP. Сервис отвечает за off-line обучение, паттерн-анализ и подготовку профилей, которые могут быть активированы для HAL/Policy.

## Архитектура

```
┌────────────────────────┐
│ FastMCP (learning-mcp) │
│  └─ MCP Tools          │
└────────────┬───────────┘
             │
             ▼
├ TrainerService ── HTTP → Supervisor /query/agg, /query/facts
│                 ← агрегаты, факты
├ PatternAnalyzer ─ local analysis
└ PolicyClient ── HTTP → Policy MCP /profiles, /health
```

## Основные компоненты

- **TrainerService** — собирает агрегаты (7d/30d) из Supervisor MCP, обучает модель (RandomForest) и формирует `DecisionProfile`.
- **PatternAnalyzer** — извлекает успешные/неуспешные паттерны из фактов.
- **PolicyClient** — синхронизирует профили с Policy MCP (create/update/activate), health-check.
- **OnlineLearningService** — фоновые циклы online learning, сравнение профилей и автоматическая активация.
- **Benchmark script** — `python -m learning_mcp.scripts.benchmark` для сравнения окон/метрик обучения.
- **MCP tools** — `train_offline`, `propose_profile`, `validate_profile`, `policy_health`, `list_policy_profiles`, `get_active_policy_profile`.

## Поток данных

1. `train_offline` → TrainerService → Supervisor `/query/agg`/`/query/facts` → модель → DecisionProfile.
2. `propose_profile` → PolicyClient `/profiles` (активация по флагу).
3. `policy_health` → Policy MCP `/health`.
4. Фоновый цикл `OnlineLearningService` запускается при старте HTTP сервера каждые `LEARNING_ONLINE_LEARNING_INTERVAL_SECONDS` секунд.

## Настройка

- Конфигурация через `LEARNING_*` переменные (`config.py`), пример — `.env.example`.
- Для e2e тестов использована SQLite + httpx ASGI клиента Policy MCP.

## TODO

- Добавить scheduler для периодического обучения.
- Внедрить онлайн обучение и A/B testing.
- Разработать расширенные ML-пайплайны (feature engineering, explainability).
