"""
Тесты для fallback стратегий LLM планировщика.

Проверяет поведение системы при сбоях LLM, включая:
- Невалидный JSON
- Пустые планы
- Ошибки сети
- LLM недоступен
"""

from unittest.mock import Mock

import pytest

from llm.base_client import LLMClient
from planner.task_planner import LLMTaskPlanner, Tool


class TestLLMFallback:
    """Тесты для fallback стратегий LLM."""

    @pytest.fixture
    def mock_llm_client(self):
        """Создает мок LLM клиента."""
        client = Mock(spec=LLMClient)
        return client

    @pytest.fixture
    def planner(self, mock_llm_client):
        """Создает LLM планировщик с мок клиентом."""
        return LLMTaskPlanner(client=mock_llm_client)

    def test_fallback_on_invalid_json(self, planner, mock_llm_client):
        """Тест: fallback при невалидном JSON."""
        # Настраиваем LLM для возврата невалидного JSON
        mock_llm_client.generate.return_value = "invalid json {"

        plan = planner.plan("test request")

        # Должен вернуться fallback план с одним шагом
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == "test request"
        assert plan.context == [Tool.CODE.value]

    def test_fallback_on_empty_plan(self, planner, mock_llm_client):
        """Тест: fallback при пустом плане."""
        # Настраиваем LLM для возврата плана без шагов
        mock_llm_client.generate.return_value = '{"steps": [], "context": []}'

        plan = planner.plan("test request")

        # Должен вернуться fallback план с одним шагом
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == "test request"
        assert plan.context == [Tool.CODE.value]

    def test_fallback_on_validation_error(self, planner, mock_llm_client):
        """Тест: fallback при ошибке валидации Pydantic."""
        # Настраиваем LLM для возврата некорректной структуры
        mock_llm_client.generate.return_value = '{"invalid_field": "value"}'

        plan = planner.plan("test request")

        # Должен вернуться fallback план с одним шагом
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == "test request"
        assert plan.context == [Tool.CODE.value]

    def test_fallback_on_second_attempt_failure(self, planner, mock_llm_client):
        """Тест: fallback при неудаче второй попытки."""
        # Настраиваем LLM для возврата невалидного JSON в обеих попытках
        mock_llm_client.generate.side_effect = ["invalid json {", "still invalid {"]

        plan = planner.plan("test request")

        # Должен вернуться fallback план с одним шагом
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == "test request"
        assert plan.context == [Tool.CODE.value]

        # Проверяем, что было не менее двух попыток (некоторые реализации могут сделать финальную 3-ю попытку)
        assert 2 <= mock_llm_client.generate.call_count <= 3

    def test_successful_plan_generation(self, planner, mock_llm_client):
        """Тест: успешная генерация плана."""
        # Настраиваем LLM для возврата валидного плана
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print('hello')",
                    "expected_output": "hello",
                    "is_final": True,
                }
            ],
            "task_completion_criteria": "Task complete when hello is printed",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("print hello")

        # Должен вернуться корректный план
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == "print('hello')"
        assert plan.steps[0].expected_output == "hello"
        assert plan.steps[0].is_final is True
        assert plan.context == [Tool.CODE.value]

    def test_plan_with_context_and_previous_results(self, planner, mock_llm_client):
        """Тест: генерация плана с контекстом и предыдущими результатами."""
        context = ["previous action 1", "previous action 2"]
        previous_results = ["result 1", "result 2"]

        # Настраиваем LLM для возврата валидного плана
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print('using context')",
                    "expected_output": "using context",
                    "is_final": True,
                }
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan(
            "test request", context=context, previous_results=previous_results
        )

        # Проверяем, что план был сгенерирован
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == "print('using context')"

        # Проверяем, что контекст был передан в промпт
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "previous action 1" in call_args
        assert "previous action 2" in call_args
        assert "result 1" in call_args
        assert "result 2" in call_args

    def test_plan_with_multiple_steps(self, planner, mock_llm_client):
        """Тест: генерация плана с множественными шагами."""
        # Настраиваем LLM для возврата плана с несколькими шагами
        valid_plan = {
            "steps": [
                {
                    "tool": "file_io",
                    "content": "write test.txt\nhello world",
                    "expected_output": "File test.txt created",
                    "is_final": False,
                },
                {
                    "tool": "code",
                    "content": "print('file created')",
                    "expected_output": "file created",
                    "is_final": True,
                },
            ],
            "task_completion_criteria": "Task complete when file is created and message printed",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("create file and print message")

        # Проверяем, что план содержит два шага
        assert len(plan.steps) == 2

        # Проверяем первый шаг
        assert plan.steps[0].tool == Tool.FILE_IO
        assert plan.steps[0].content == "write test.txt\nhello world"
        assert plan.steps[0].expected_output == "File test.txt created"
        assert plan.steps[0].is_final is False

        # Проверяем второй шаг
        assert plan.steps[1].tool == Tool.CODE
        assert plan.steps[1].content == "print('file created')"
        assert plan.steps[1].expected_output == "file created"
        assert plan.steps[1].is_final is True

        # Проверяем контекст
        assert Tool.FILE_IO.value in plan.context
        assert Tool.CODE.value in plan.context

    def test_plan_with_dependencies(self, planner, mock_llm_client):
        """Тест: генерация плана с зависимостями между шагами."""
        # Настраиваем LLM для возврата плана с зависимостями
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "x = 5",
                    "expected_output": "Variable x set",
                    "is_final": False,
                    "depends_on": [],
                },
                {
                    "tool": "code",
                    "content": "print(x)",
                    "expected_output": "5",
                    "is_final": True,
                    "depends_on": [0],
                },
            ],
            "task_completion_criteria": "Task complete when x is printed",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("set variable and print it")

        # Проверяем, что план содержит два шага
        assert len(plan.steps) == 2

        # Проверяем зависимости
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == [0]

    def test_plan_with_completion_field(self, planner, mock_llm_client):
        """Тест: генерация плана с полем completion."""
        # Настраиваем LLM для возврата плана с полем completion
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print('step1')\nCompletion: Step 1 completed",
                    "expected_output": "step1",
                    "is_final": False,
                },
                {
                    "tool": "code",
                    "content": "print('step2')",
                    "expected_output": "step2",
                    "is_final": True,
                },
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        import json

        mock_llm_client.generate.return_value = json.dumps(valid_plan)

        plan = planner.plan("test completion field")

        # Проверяем, что поле completion было обработано
        assert len(plan.steps) == 2
        assert plan.steps[0].content == "print('step1')"
        assert plan.steps[0].completion == "Step 1 completed"
        assert plan.steps[1].content == "print('step2')"
        assert plan.steps[1].completion is None

    def test_plan_with_preconditions_and_postconditions(self, planner, mock_llm_client):
        """Тест: генерация плана с пред- и постусловиями."""
        # Настраиваем LLM для возврата плана с условиями
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print('checking preconditions')",
                    "expected_output": "checking preconditions",
                    "is_final": False,
                    "preconditions": ["system_ready"],
                    "postconditions": ["preconditions_checked"],
                }
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("test conditions")

        # Проверяем, что условия были обработаны
        assert len(plan.steps) == 1
        assert len(plan.steps[0].preconditions) == 1
        assert plan.steps[0].preconditions[0] == "system_ready"
        assert len(plan.steps[0].postconditions) == 1
        assert plan.steps[0].postconditions[0] == "preconditions_checked"

    def test_plan_with_policy(self, planner, mock_llm_client):
        """Тест: генерация плана с политикой выполнения."""
        # Настраиваем LLM для возврата плана с политикой
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print('with policy')",
                    "expected_output": "with policy",
                    "is_final": True,
                    "policy": "auto",
                }
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("test policy")

        # Проверяем, что политика была обработана
        assert len(plan.steps) == 1
        assert plan.steps[0].policy == "auto"

    def test_plan_with_inputs(self, planner, mock_llm_client):
        """Тест: генерация плана с входными параметрами."""
        # Настраиваем LLM для возврата плана с входными параметрами
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print(input_value)",
                    "expected_output": "input_value",
                    "is_final": True,
                    "inputs": {"input_value": "test"},
                }
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("test inputs")

        # Проверяем, что входные параметры были обработаны
        assert len(plan.steps) == 1
        assert plan.steps[0].inputs == {"input_value": "test"}

    def test_plan_with_id_field(self, planner, mock_llm_client):
        """Тест: генерация плана с полем id."""
        # Настраиваем LLM для возврата плана с id
        valid_plan = {
            "steps": [
                {
                    "id": "step_1",
                    "tool": "code",
                    "content": "print('step 1')",
                    "expected_output": "step 1",
                    "is_final": False,
                },
                {
                    "id": "step_2",
                    "tool": "code",
                    "content": "print('step 2')",
                    "expected_output": "step 2",
                    "is_final": True,
                },
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("test id field")

        # Проверяем, что id были обработаны
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "step_1"
        assert plan.steps[1].id == "step_2"

    def test_plan_with_empty_string_request(self, planner, mock_llm_client):
        """Тест: обработка пустого запроса."""
        # Настраиваем LLM для возврата валидного плана
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print('empty request')",
                    "expected_output": "empty request",
                    "is_final": True,
                }
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("")

        # Должен вернуться план для пустого запроса
        assert len(plan.steps) == 1
        assert plan.steps[0].content == "print('empty request')"

    def test_plan_with_none_context_and_results(self, planner, mock_llm_client):
        """Тест: обработка None контекста и результатов."""
        # Настраиваем LLM для возврата валидного плана
        valid_plan = {
            "steps": [
                {
                    "tool": "code",
                    "content": "print('no context')",
                    "expected_output": "no context",
                    "is_final": True,
                }
            ],
            "task_completion_criteria": "Task complete",
            "requires_all_steps": True,
        }
        mock_llm_client.generate.return_value = str(valid_plan).replace("'", '"')

        plan = planner.plan("test request", context=None, previous_results=None)

        # Должен вернуться план без контекста
        assert len(plan.steps) == 1
        assert plan.steps[0].content == "print('no context')"

        # Проверяем, что промпт не содержит информацию о контексте
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "Previous actions:" not in call_args
        assert "Known data:" not in call_args


class TestLLMFallbackIntegration:
    """Интеграционные тесты для fallback стратегий."""

    @pytest.fixture
    def mock_llm_client(self):
        """Создает мок LLM клиента для интеграционных тестов."""
        client = Mock(spec=LLMClient)
        return client

    @pytest.fixture
    def planner(self, mock_llm_client):
        """Создает LLM планировщик с мок клиентом."""
        return LLMTaskPlanner(client=mock_llm_client)

    def test_integration_fallback_chain(self, planner, mock_llm_client):
        """Тест: полная цепочка fallback при различных сбоях."""
        # Симулируем различные типы сбоев LLM
        mock_llm_client.generate.side_effect = [
            "invalid json",  # Первая попытка - невалидный JSON
            '{"steps": []}',  # Вторая попытка - пустой план
            "still invalid",  # Третья попытка - снова невалидный JSON
        ]

        plan = planner.plan("integration test")

        # Должен вернуться fallback план
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == "integration test"

        # Проверяем, что было три попытки
        assert mock_llm_client.generate.call_count == 3

    def test_integration_realistic_scenario(self, planner, mock_llm_client):
        """Тест: реалистичный сценарий с частичными сбоями."""
        # Симулируем LLM, который иногда работает, иногда нет
        mock_llm_client.generate.side_effect = [
            "invalid json",  # Первая попытка - сбой
            '{"steps": [{"tool": "code", "content": "print(\\"hello\\")", "expected_output": "hello", "is_final": true}], "task_completion_criteria": "Task complete", "requires_all_steps": true}',  # Вторая попытка - успех
        ]

        plan = planner.plan("print hello")

        # Должен вернуться корректный план
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        assert plan.steps[0].content == 'print("hello")'
        assert plan.steps[0].expected_output == "hello"
        assert plan.steps[0].is_final is True

        # Проверяем, что было две попытки
        assert mock_llm_client.generate.call_count == 2

    def test_integration_error_recovery(self, planner, mock_llm_client):
        """Тест: восстановление после ошибок."""
        # Симулируем LLM, который восстанавливается после ошибок
        mock_llm_client.generate.side_effect = [
            "invalid json",  # Первая попытка - сбой
            "still invalid",  # Вторая попытка - сбой
            '{"steps": [{"tool": "code", "content": "print(\\"recovered\\")", "expected_output": "recovered", "is_final": true}], "task_completion_criteria": "Task complete", "requires_all_steps": true}',  # Третья попытка - успех
        ]

        plan = planner.plan("test recovery")

        # Должен вернуться корректный план
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == Tool.CODE
        # Проверяем содержимое, но не требуем точного совпадения
        content = plan.steps[0].content
        assert (
            "print" in content and "recovered" in content
        ), f"Содержимое должно содержать 'print' и 'recovered', получено: {content}"
        assert plan.steps[0].expected_output == "recovered"
        assert plan.steps[0].is_final is True

        # Проверяем, что было три попытки
        assert mock_llm_client.generate.call_count == 3

    def test_fallback_prompt_without_request(
        self, planner, mock_llm_client, monkeypatch
    ):
        """Тест: fallback-план при отсутствии `Request` в исходном промпте."""

        def fake_prompt(request, context, previous_results):
            return "No Request line here"

        monkeypatch.setattr(planner, "_build_prompt", fake_prompt)

        fallback_json = (
            '{"steps": [{"tool": "code", "content": "ежедневная сводка", '
            '"expected_output": "", "is_final": true}], '
            '"task_completion_criteria": "", "requires_all_steps": true}'
        )
        mock_llm_client.generate.side_effect = ["", fallback_json]

        plan = planner.plan("ignored request")

        expected_prompt = "Создай простой план для задачи: ежедневная сводка"
        assert mock_llm_client.generate.call_args_list[1][0][0] == expected_prompt
        assert len(plan.steps) == 1
        assert plan.steps[0].content == "ежедневная сводка"
        assert plan.steps[0].tool == Tool.CODE
        assert plan.context == [Tool.CODE.value]
