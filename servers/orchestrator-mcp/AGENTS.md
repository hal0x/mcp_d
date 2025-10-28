# Orchestrator MCP — Architecture

## Overview

Orchestrator MCP принимает планы действий, координирует вызовы MCP-сервисов (Supervisor, Policy, Learning и др.), обрабатывает ошибки и при необходимости откатывает выполненные шаги.

## Architecture

```
┌────────────────────────┐
│ FastMCP (orchestrator) │
│  ├─ MCP Tools          │
│  └─ HTTP /stdio        │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ OrchestratorService    │
│  ├─ execute_plan       │
│  ├─ dry_run_plan       │
│  └─ rollback           │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐        ┌────────────────────────┐
│ SupervisorClient       │        │ PolicyClient           │
│ (metrics, health)      │        │ (profiles, experiments)│
└────────────┬───────────┘        └────────────┬───────────┘
             │                                   │
             ▼                                   ▼
         ┌────────────────────────┐
         │ LearningClient         │
         │ (online learning API)  │
         └────────────────────────┘
```

## Flow

1. MCP tool `execute_plan` получает описание плана и опциональные параметры (dry_run, metadata). 
2. OrchestratorService проходится по шагам, вызывает Supervisor/Policy/Learning действия (`fetch_supervisor_health`, `check_policy_profile`, `trigger_learning_online`, и т.д.).
3. При ошибке выполняется rollback для уже завершённых шагов (логический отчёт).

## Config

- `ORCH_SUPERVISOR_URL`, `ORCH_POLICY_URL`, `ORCH_LEARNING_URL` — внешние интеграции.
- `ORCH_DEFAULT_TIMEOUT` — таймаут для httpx-клиентов.

## TODO

- Расширить поддержку Learning MCP и пользовательских шагов.
- Добавить persistence для историй планов.
- Интегрировать с feature flag системой.
