# Тесты бота

Этот каталог содержит интеграционные тесты для проверки работы Telegram бота.

## Файлы

- `test_bot_simple.py` - Простые тесты с минимальной настройкой
- `test_bot_queries.py` - Полные тесты с проверкой разных типов запросов
- `run_bot_tests.py` - Скрипт для запуска тестов

## Быстрый тест

Для быстрой проверки работы бота можно использовать скрипт:

```bash
python tests/integration/test_bot_quick.py
```

## Запуск тестов

### Простые тесты
```bash
python tests/run_bot_tests.py --simple
```

### Полные тесты
```bash
python tests/run_bot_tests.py --full
```

### Все тесты
```bash
python tests/run_bot_tests.py
```

### С подробным выводом
```bash
python tests/run_bot_tests.py --verbose
```

## Требования

- Ollama должен быть запущен на localhost:11434
- Docker должен быть доступен для исполнителя кода
- Python 3.11+

## Что тестируется

### Простые тесты (`test_bot_simple.py`)
- Базовая обработка сообщений
- Простые вопросы
- Использование памяти
- Несколько сообщений подряд

### Полные тесты (`test_bot_queries.py`)
- Простые запросы с контекстом памяти
- Сложные запросы (планирование и выполнение)
- Команды бота
- Обработка ошибок
- Сохранение информации в память
- Множественные запросы

## Структура тестов

Каждый тест создает изолированную тестовую среду с:
- Event Bus для обработки сообщений
- AgentCore для основной логики
- MemoryServiceAdapter для памяти агента
- LLM клиент для генерации ответов
- Планировщик задач
- Исполнитель кода

## Отладка

Если тесты не проходят:

1. Проверьте, что Ollama запущен:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Проверьте, что Docker доступен:
   ```bash
   docker --version
   ```

3. Запустите с подробным выводом:
   ```bash
   python tests/run_bot_tests.py --verbose
   ```

4. Проверьте логи в консоли

## Добавление новых тестов

Для добавления нового теста:

1. Создайте новый метод в классе `TestSimpleBot` или `TestBotQueries`
2. Используйте декоратор `@pytest.mark.asyncio`
3. Используйте фикстуру `simple_bot` или `bot_env`
4. Вызывайте `await bot.send_message("текст")` для отправки сообщений
5. Проверяйте результаты с помощью `assert`

Пример:
```python
@pytest.mark.asyncio
async def test_my_feature(self, simple_bot):
    """Тест моей функции."""
    replies = await simple_bot.send_message("тестовое сообщение")
    assert len(replies) > 0
    assert "ожидаемый ответ" in replies[0].reply
```
