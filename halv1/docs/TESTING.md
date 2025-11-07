# 🧪 Тестирование

## Статус
7/8 интеграционных тестов проходят, падает `test_multi_step_task`.

## Как запускать

### Рекомендуемые способы (без предупреждений SWIG)
```bash
# Групповые тесты (рекомендуется)
python tests/run_tests.py agent      # тесты агента
python tests/run_tests.py memory     # тесты памяти
python tests/run_tests.py all        # все тесты

# Интеграционные тесты
python tests/run_integration_tests.py --mode fast   # быстрые тесты
python tests/run_integration_tests.py --mode full   # полный прогон

# Скрипт без предупреждений
python scripts/run_tests_no_warnings.py tests/test_agent -q
```

### Прямой запуск pytest (с предупреждениями SWIG)
```bash
python -m pytest tests/ -q                          # все тесты
python -m pytest tests/test_agent -q                # тесты агента
```

### Подавление предупреждений вручную
```bash
PYTHONWARNINGS="ignore::DeprecationWarning" python -m pytest tests/ -q
```

## Структура тестов
- `tests/test_planner/` – планировщик
- `tests/test_core/` – ядро и потоковая логика
- `tests/test_executor/` – исполнители и генерация кода
- `tests/test_memory/` – уровни памяти L0–L∞
- `tests/integration/` – интеграционные сценарии

## Диагностика
- Docker: `docker info`
- Многошаговые задачи: `python -m pytest tests/integration/test_integration_full.py::TestIntegrationFull::test_multi_step_task -v`
- Event bus завершение: `python -m pytest tests/test_agent/test_multistep_final_event.py`
- Нестабильный LLM: `--skip-llm-check`

## Предупреждения SWIG

### Проблема
При запуске тестов появляются предупреждения от библиотеки `faiss-cpu`:
```
DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

### Решение
- **Рекомендуется**: использовать скрипты `tests/run_tests.py` или `scripts/run_tests_no_warnings.py`
- **Вручную**: установить переменную окружения `PYTHONWARNINGS="ignore::DeprecationWarning"`
- **В коде**: предупреждения подавляются в `main.py` и `config/warnings_config.py`

### Техническая информация
- Предупреждения генерируются на уровне `importlib._bootstrap` при импорте `faiss-cpu`
- Не влияют на функциональность, только засоряют вывод
- Исправление требует обновления библиотеки `faiss-cpu` или использования альтернативы

## Известные проблемы
- LLM planner: несовместимость идентификаторов шагов и `expected_output`
- File I/O: различия в поведении `open()` и создании директорий
- Event bus: публикации и таймауты при завершении
- Docker недоступен: нужен локальный fallback

## Быстрые ссылки
- [INDEX.md](INDEX.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)
