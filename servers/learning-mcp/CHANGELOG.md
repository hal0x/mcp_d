# Changelog

Все заметные изменения Learning MCP фиксируются здесь. Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

## [0.2.0] - 2025-10-28
### Added
- PolicyClient с REST-интеграцией и health-check инструментами.
- MCP инструмент `policy_health`, e2e тест на взаимодействие с Policy MCP.
- Документ `AGENTS.md` и `.env.example` с основными переменными.
- OnlineLearningService с автоматическим циклом и A/B сравнением.
- Feature engineering pipeline, GridSearchCV, permutation importance и реестр моделей.
- Скрипт `learning-mcp-benchmark` для запуска ML benchmarking сценариев.

### Changed
- Обновлён сервер для инициализации Policy/OnlineLearning клиентов.

## [0.1.0] - 2025-10-23
### Added
- Первичная версия Learning MCP с offline training, pattern analyzer и генерацией профилей.
