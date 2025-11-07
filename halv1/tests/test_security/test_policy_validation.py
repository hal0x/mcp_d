"""
Тесты для валидации политик безопасности.

Проверяет логику PolicyEngine, включая:
- Загрузку валидных YAML политик
- Обработку невалидных YAML
- Валидацию структуры политик
- Обработку отсутствующих обязательных полей
"""

from unittest.mock import patch

import pytest

from executor import ToolPolicy
from security.policy_engine import PolicyEngine
from tools import Tool


class TestPolicyValidation:
    """Тесты для валидации политик безопасности."""

    @pytest.fixture
    def valid_policy_yaml(self):
        """Возвращает валидный YAML для политик."""
        return """
        code:
          max_wall_time_s: 30.0
          max_mem_mb: 128
          cpu_quota: 0.5
          network_mode: "none"
          cap_drop: "ALL"
          seccomp_profile: "security/seccomp_profile.json"
        
        file_io:
          max_wall_time_s: 10.0
          max_mem_mb: 64
          network_mode: "none"
          cap_drop: "ALL"
        
        search:
          max_wall_time_s: 60.0
          max_mem_mb: 256
          network_mode: "host"
          cap_drop: "NET_ADMIN"
        
        shell:
          max_wall_time_s: 15.0
          max_mem_mb: 128
          cpu_quota: 0.3
          network_mode: "none"
          cap_drop: "ALL"
        
        http:
          max_wall_time_s: 45.0
          max_mem_mb: 128
          network_mode: "host"
          cap_drop: "NET_ADMIN"
        """

    @pytest.fixture
    def invalid_policy_yaml(self):
        """Возвращает невалидный YAML для политик."""
        return """
        code:
          invalid_field: "value"
          max_wall_time_s: "not_a_number"
          max_mem_mb: -1
          cpu_quota: 2.5  # Превышает лимит
        
        file_io:
          max_wall_time_s: 0  # Нулевое время
          network_mode: "invalid_mode"
        """

    @pytest.fixture
    def malformed_yaml(self):
        """Возвращает синтаксически неверный YAML."""
        return """
        code:
          max_wall_time_s: 30.0
          max_mem_mb: 128
          cpu_quota: 0.5
          network_mode: "none"
          cap_drop: "ALL"
          seccomp_profile: "security/seccomp_profile.json"
        
        file_io:
          max_wall_time_s: 10.0
          max_mem_mb: 64
          network_mode: "none"
          cap_drop: "ALL"
        
        search:
          max_wall_time_s: 60.0
          max_mem_mb: 256
          network_mode: "host"
          cap_drop: "NET_ADMIN"
        
        shell:
          max_wall_time_s: 15.0
          max_mem_mb: 128
          cpu_quota: 0.3
          network_mode: "none"
          cap_drop: "ALL"
        
        http:
          max_wall_time_s: 45.0
          max_mem_mb: 128
          network_mode: "host"
          cap_drop: "NET_ADMIN"
        
        # Неправильная структура - отсутствует закрывающая скобка
        invalid_tool:
          max_wall_time_s: 30.0
          max_mem_mb: 128
          network_mode: "none"
        """

    def test_load_valid_policy(self, valid_policy_yaml, tmp_path):
        """Тест: загрузка валидной политики."""
        policy_file = tmp_path / "policies.yaml"
        policy_file.write_text(valid_policy_yaml)

        engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики были загружены
        assert len(engine._policies) == 5

        # Проверяем конкретные политики
        code_policy = engine.get_policy(Tool.CODE)
        assert code_policy.max_wall_time_s == 30.0
        assert code_policy.max_mem_mb == 128
        assert code_policy.cpu_quota == 0.5
        assert code_policy.network_mode == "none"
        assert code_policy.cap_drop == "ALL"
        assert code_policy.seccomp_profile == "security/seccomp_profile.json"

        search_policy = engine.get_policy(Tool.SEARCH)
        assert search_policy.max_wall_time_s == 60.0
        assert search_policy.max_mem_mb == 256
        assert search_policy.network_mode == "host"
        assert search_policy.cap_drop == "NET_ADMIN"

    def test_load_invalid_policy_gracefully(self, invalid_policy_yaml, tmp_path):
        """Тест: graceful обработка невалидной политики."""
        policy_file = tmp_path / "policies.yaml"
        policy_file.write_text(invalid_policy_yaml)

        # Должен загрузиться без ошибок, но с предупреждениями
        engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики были загружены (только валидные)
        assert len(engine._policies) > 0

        # Проверяем, что невалидные поля были проигнорированы
        code_policy = engine.get_policy(Tool.CODE)
        # Должны использоваться значения по умолчанию для невалидных полей
        assert code_policy.max_wall_time_s is None  # По умолчанию
        assert code_policy.max_mem_mb is None  # По умолчанию
        assert code_policy.cpu_quota is None  # По умолчанию

    def test_load_malformed_yaml_gracefully(self, malformed_yaml, tmp_path):
        """Тест: graceful обработка синтаксически неверного YAML."""
        policy_file = tmp_path / "policies.yaml"
        policy_file.write_text(malformed_yaml)

        # Должен загрузиться без ошибок, но с предупреждениями
        engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики были загружены (только валидные)
        assert len(engine._policies) > 0

    def test_load_nonexistent_file(self, tmp_path):
        """Тест: загрузка несуществующего файла политик."""
        policy_file = tmp_path / "nonexistent.yaml"

        # Должен загрузиться без ошибок, но с пустыми политиками
        engine = PolicyEngine(str(policy_file))

        assert len(engine._policies) == 0

        # Проверяем, что возвращается политика по умолчанию
        default_policy = engine.get_policy(Tool.CODE)
        assert isinstance(default_policy, ToolPolicy)

    def test_load_empty_file(self, tmp_path):
        """Тест: загрузка пустого файла политик."""
        policy_file = tmp_path / "empty.yaml"
        policy_file.write_text("")

        # Должен загрузиться без ошибок, но с пустыми политиками
        engine = PolicyEngine(str(policy_file))

        assert len(engine._policies) == 0

    def test_load_file_with_only_comments(self, tmp_path):
        """Тест: загрузка файла только с комментариями."""
        policy_file = tmp_path / "comments.yaml"
        policy_file.write_text("# This is a comment\n# Another comment")

        # Должен загрузиться без ошибок, но с пустыми политиками
        engine = PolicyEngine(str(policy_file))

        assert len(engine._policies) == 0

    def test_load_file_with_invalid_tool_names(self, tmp_path):
        """Тест: загрузка файла с невалидными именами инструментов."""
        invalid_tools_yaml = """
        invalid_tool_name:
          max_wall_time_s: 30.0
        
        another_invalid:
          max_mem_mb: 128
        
        code:  # Валидный инструмент
          max_wall_time_s: 30.0
        """

        policy_file = tmp_path / "invalid_tools.yaml"
        policy_file.write_text(invalid_tools_yaml)

        engine = PolicyEngine(str(policy_file))

        # Проверяем, что только валидные инструменты были загружены
        assert len(engine._policies) == 1
        assert Tool.CODE in engine._policies

        # Проверяем, что политика для валидного инструмента загружена
        code_policy = engine.get_policy(Tool.CODE)
        assert code_policy.max_wall_time_s == 30.0

    def test_load_file_with_mixed_valid_invalid(self, tmp_path):
        """Тест: загрузка файла со смешанными валидными и невалидными политиками."""
        mixed_yaml = """
        code:
          max_wall_time_s: 30.0
          invalid_field: "should_be_ignored"
        
        file_io:
          max_wall_time_s: "not_a_number"  # Невалидное значение
          max_mem_mb: 64
        
        search:
          max_wall_time_s: 60.0
          network_mode: "invalid_mode"  # Невалидный режим
        """

        policy_file = tmp_path / "mixed.yaml"
        policy_file.write_text(mixed_yaml)

        engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики были загружены
        assert len(engine._policies) == 3

        # Проверяем, что валидные поля загружены, а невалидные проигнорированы
        code_policy = engine.get_policy(Tool.CODE)
        assert code_policy.max_wall_time_s == 30.0

        file_io_policy = engine.get_policy(Tool.FILE_IO)
        assert file_io_policy.max_mem_mb == 64
        assert (
            file_io_policy.max_wall_time_s is None
        )  # Невалидное значение проигнорировано

        search_policy = engine.get_policy(Tool.SEARCH)
        assert search_policy.max_wall_time_s == 60.0
        assert search_policy.network_mode == "none"  # Значение по умолчанию

    def test_get_policy_for_unknown_tool(self):
        """Тест: получение политики для неизвестного инструмента."""
        engine = PolicyEngine("nonexistent.yaml")

        # Должен вернуться политика по умолчанию
        policy = engine.get_policy(Tool.CODE)
        assert isinstance(policy, ToolPolicy)

        # Проверяем значения по умолчанию
        assert policy.max_wall_time_s is None
        assert policy.max_mem_mb is None
        assert policy.cpu_quota is None
        assert policy.network_mode == "none"
        assert policy.cap_drop == "ALL"
        assert policy.seccomp_profile == "security/seccomp_profile.json"

    def test_policy_immutability(self, valid_policy_yaml, tmp_path):
        """Тест: неизменяемость загруженных политик."""
        policy_file = tmp_path / "policies.yaml"
        policy_file.write_text(valid_policy_yaml)

        engine = PolicyEngine(str(policy_file))

        # Получаем политику
        original_policy = engine.get_policy(Tool.CODE)

        # Изменяем файл политик
        updated_yaml = """
        code:
          max_wall_time_s: 60.0
          max_mem_mb: 256
        """
        policy_file.write_text(updated_yaml)

        # Создаем новый движок
        new_engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики изменились
        new_policy = new_engine.get_policy(Tool.CODE)
        assert new_policy.max_wall_time_s == 60.0
        assert new_policy.max_mem_mb == 256

        # Проверяем, что оригинальная политика не изменилась
        assert original_policy.max_wall_time_s == 30.0
        assert original_policy.max_mem_mb == 128

    def test_policy_with_all_fields(self, tmp_path):
        """Тест: политика со всеми возможными полями."""
        complete_policy_yaml = """
        code:
          max_wall_time_s: 30.0
          max_mem_mb: 128
          cpu_quota: 0.5
          network_mode: "bridge"
          network_proxy: "http://proxy:8080"
          userns: "host"
          cap_drop: "NET_ADMIN"
          seccomp_profile: "custom_seccomp.json"
          apparmor_profile: "custom_apparmor"
        """

        policy_file = tmp_path / "complete.yaml"
        policy_file.write_text(complete_policy_yaml)

        engine = PolicyEngine(str(policy_file))

        code_policy = engine.get_policy(Tool.CODE)
        assert code_policy.max_wall_time_s == 30.0
        assert code_policy.max_mem_mb == 128
        assert code_policy.cpu_quota == 0.5
        assert code_policy.network_mode == "bridge"
        assert code_policy.network_proxy == "http://proxy:8080"
        assert code_policy.userns == "host"
        assert code_policy.cap_drop == "NET_ADMIN"
        assert code_policy.seccomp_profile == "custom_seccomp.json"
        assert code_policy.apparmor_profile == "custom_apparmor"

    def test_policy_with_numeric_strings(self, tmp_path):
        """Тест: политика с числовыми значениями в виде строк."""
        string_numbers_yaml = """
        code:
          max_wall_time_s: "30.0"
          max_mem_mb: "128"
          cpu_quota: "0.5"
        """

        policy_file = tmp_path / "string_numbers.yaml"
        policy_file.write_text(string_numbers_yaml)

        engine = PolicyEngine(str(policy_file))

        code_policy = engine.get_policy(Tool.CODE)
        # YAML автоматически конвертирует строки в числа
        assert code_policy.max_wall_time_s == 30.0
        assert code_policy.max_mem_mb == 128
        assert code_policy.cpu_quota == 0.5

    def test_policy_with_boolean_values(self, tmp_path):
        """Тест: политика с булевыми значениями."""
        boolean_yaml = """
        code:
          max_wall_time_s: 30.0
          network_mode: "none"
          cap_drop: "ALL"
        
        search:
          max_wall_time_s: 60.0
          network_mode: "host"
          cap_drop: "NET_ADMIN"
        """

        policy_file = tmp_path / "boolean.yaml"
        policy_file.write_text(boolean_yaml)

        engine = PolicyEngine(str(policy_file))

        code_policy = engine.get_policy(Tool.CODE)
        search_policy = engine.get_policy(Tool.SEARCH)

        assert code_policy.network_mode == "none"
        assert search_policy.network_mode == "host"

    def test_policy_with_unicode_characters(self, tmp_path):
        """Тест: политика с Unicode символами."""
        unicode_yaml = """
        code:
          max_wall_time_s: 30.0
          max_mem_mb: 128
          network_mode: "none"
          cap_drop: "ВСЕ"  # Unicode в комментарии
        
        file_io:
          max_wall_time_s: 10.0
          max_mem_mb: 64
          network_mode: "none"
          cap_drop: "ALL"
        """

        policy_file = tmp_path / "unicode.yaml"
        policy_file.write_text(unicode_yaml)

        engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики загрузились корректно
        assert len(engine._policies) == 2

        code_policy = engine.get_policy(Tool.CODE)
        file_io_policy = engine.get_policy(Tool.FILE_IO)

        assert code_policy.max_wall_time_s == 30.0
        assert file_io_policy.max_wall_time_s == 10.0

    def test_policy_with_nested_structures(self, tmp_path):
        """Тест: политика с вложенными структурами (должны быть проигнорированы)."""
        nested_yaml = """
        code:
          max_wall_time_s: 30.0
          max_mem_mb: 128
          nested_config:
            sub_config:
              value: "ignored"
            another_value: "also_ignored"
        
        file_io:
          max_wall_time_s: 10.0
          max_mem_mb: 64
        """

        policy_file = tmp_path / "nested.yaml"
        policy_file.write_text(nested_yaml)

        engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики загрузились, но вложенные структуры проигнорированы
        assert len(engine._policies) == 2

        code_policy = engine.get_policy(Tool.CODE)
        assert code_policy.max_wall_time_s == 30.0
        assert code_policy.max_mem_mb == 128

        # Вложенные структуры не должны быть доступны
        assert not hasattr(code_policy, "nested_config")

    def test_policy_with_environment_variables(self, tmp_path):
        """Тест: политика с переменными окружения (должны быть проигнорированы)."""
        env_yaml = """
        code:
          max_wall_time_s: 30.0
          max_mem_mb: 128
          network_mode: "none"
          cap_drop: "ALL"
        
        file_io:
          max_wall_time_s: 10.0
          max_mem_mb: 64
          network_mode: "none"
          cap_drop: "ALL"
        """

        policy_file = tmp_path / "env.yaml"
        policy_file.write_text(env_yaml)

        # Устанавливаем переменные окружения
        with patch.dict("os.environ", {"CUSTOM_TIMEOUT": "60", "CUSTOM_MEMORY": "256"}):
            engine = PolicyEngine(str(policy_file))

            # Проверяем, что политики загрузились корректно
            assert len(engine._policies) == 2

            code_policy = engine.get_policy(Tool.CODE)
            assert code_policy.max_wall_time_s == 30.0
            assert code_policy.max_mem_mb == 128

    def test_policy_with_comments_and_whitespace(self, tmp_path):
        """Тест: политика с комментариями и пробелами."""
        commented_yaml = """
        # Основные настройки для выполнения кода
        code:
          # Максимальное время выполнения в секундах
          max_wall_time_s: 30.0
          
          # Максимальное использование памяти в МБ
          max_mem_mb: 128
          
          # Квота CPU (0.0 - 1.0)
          cpu_quota: 0.5
          
          # Режим сети
          network_mode: "none"
          
          # Какие capabilities отключить
          cap_drop: "ALL"
        
        # Настройки для файловых операций
        file_io:
          max_wall_time_s: 10.0
          max_mem_mb: 64
          network_mode: "none"
          cap_drop: "ALL"
        """

        policy_file = tmp_path / "commented.yaml"
        policy_file.write_text(commented_yaml)

        engine = PolicyEngine(str(policy_file))

        # Проверяем, что политики загрузились корректно
        assert len(engine._policies) == 2

        code_policy = engine.get_policy(Tool.CODE)
        assert code_policy.max_wall_time_s == 30.0
        assert code_policy.max_mem_mb == 128
        assert code_policy.cpu_quota == 0.5
        assert code_policy.network_mode == "none"
        assert code_policy.cap_drop == "ALL"

        file_io_policy = engine.get_policy(Tool.FILE_IO)
        assert file_io_policy.max_wall_time_s == 10.0
        assert file_io_policy.max_mem_mb == 64
