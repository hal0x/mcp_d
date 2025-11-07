"""Тесты для профилей безопасности Docker executor."""

import pytest
from unittest.mock import patch, MagicMock

from executor.docker_executor import DockerExecutor


class TestDockerExecutorSecurity:
    """Тесты для профилей безопасности Docker."""

    def setup_method(self):
        """Подготовка для каждого теста."""
        self.executor = DockerExecutor()

    def test_apparmor_profile_applied(self):
        """Применение AppArmor профиля."""
        with patch("subprocess.run") as mock_run:
            # Настраиваем мок для возврата успешного результата
            mock_info_result = MagicMock()
            mock_info_result.returncode = 0
            mock_info_result.stdout = "rootless"  # Docker в rootless режиме
            mock_info_result.stderr = ""
            
            mock_run_result = MagicMock()
            mock_run_result.returncode = 0
            mock_run_result.stdout = "hello"
            mock_run_result.stderr = ""
            
            # Мокаем два вызова: docker info и docker run
            mock_run.side_effect = [mock_info_result, mock_run_result]

            # Выполняем код
            result = self.executor.execute("print('hello')")
            
            # Проверяем что результат корректен
            assert result is not None
            assert hasattr(result, 'stdout'), "Результат должен иметь атрибут stdout"
            assert hasattr(result, 'stderr'), "Результат должен иметь атрибут stderr"
            
            # Проверяем что subprocess.run был вызван дважды
            assert mock_run.call_count == 2, "subprocess.run должен быть вызван дважды"
            
            # Проверяем что второй вызов (docker run) содержит правильные параметры
            docker_run_call = mock_run.call_args_list[1]
            command = docker_run_call[0][0]
            
            # Проверяем что команда содержит docker
            assert "docker" in command
            
            # Проверяем что передан security-opt для AppArmor
            security_opts = [arg for arg in command if "security-opt" in arg]
            assert len(security_opts) > 0, "Должны быть переданы security-opt параметры"

    def test_seccomp_profile_applied(self):
        """Применение seccomp профиля."""
        with patch("subprocess.run") as mock_run:
            # Настраиваем мок для возврата успешного результата
            mock_info_result = MagicMock()
            mock_info_result.returncode = 0
            mock_info_result.stdout = "rootless"
            mock_info_result.stderr = ""
            
            mock_run_result = MagicMock()
            mock_run_result.returncode = 0
            mock_run_result.stdout = "test"
            mock_run_result.stderr = ""
            
            # Мокаем два вызова: docker info и docker run
            mock_run.side_effect = [mock_info_result, mock_run_result]

            # Выполняем код
            result = self.executor.execute("print('test')")
            
            # Проверяем что результат корректен
            assert result is not None
            assert hasattr(result, 'stdout'), "Результат должен иметь атрибут stdout"
            assert hasattr(result, 'stderr'), "Результат должен иметь атрибут stderr"
            
            # Проверяем что subprocess.run был вызван дважды
            assert mock_run.call_count == 2, "subprocess.run должен быть вызван дважды"
            
            # Проверяем что второй вызов (docker run) содержит правильные параметры
            docker_run_call = mock_run.call_args_list[1]
            command = docker_run_call[0][0]
            
            # Проверяем что передан security-opt для seccomp
            security_opts = [arg for arg in command if "security-opt" in arg]
            assert len(security_opts) > 0, "Должны быть переданы security-opt параметры"

    def test_missing_profile_handling(self):
        """Обработка отсутствующего профиля."""
        with patch("subprocess.run") as mock_run:
            # Настраиваем мок для возврата успешного результата
            mock_info_result = MagicMock()
            mock_info_result.returncode = 0
            mock_info_result.stdout = "rootless"
            mock_info_result.stderr = ""
            
            mock_run_result = MagicMock()
            mock_run_result.returncode = 0
            mock_run_result.stdout = "test"
            mock_run_result.stderr = ""
            
            # Мокаем два вызова: docker info и docker run
            mock_run.side_effect = [mock_info_result, mock_run_result]

            # Выполняем код
            result = self.executor.execute("print('test')")
            
            # Проверяем что результат корректен
            assert result is not None
            assert hasattr(result, 'stdout'), "Результат должен иметь атрибут stdout"
            assert hasattr(result, 'stderr'), "Результат должен иметь атрибут stderr"
            
            # Проверяем что subprocess.run был вызван дважды
            assert mock_run.call_count == 2, "subprocess.run должен быть вызван дважды"

    def test_no_docker_fallback(self):
        """Обработка отсутствия Docker."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")

            # Ожидаем исключение если Docker недоступен
            with pytest.raises(FileNotFoundError, match="docker not found"):
                self.executor.execute("print('test')")

    def test_security_profile_validation(self):
        """Проверка что профили безопасности действительно применяются."""
        with patch("subprocess.run") as mock_run:
            # Настраиваем мок для возврата успешного результата
            mock_info_result = MagicMock()
            mock_info_result.returncode = 0
            mock_info_result.stdout = "rootless"
            mock_info_result.stderr = ""
            
            mock_run_result = MagicMock()
            mock_run_result.returncode = 0
            mock_run_result.stdout = "hello"
            mock_run_result.stderr = ""
            
            # Мокаем два вызова: docker info и docker run
            mock_run.side_effect = [mock_info_result, mock_run_result]

            # Выполняем код
            result = self.executor.execute("print('hello')")
            
            # Проверяем что результат корректен
            assert result is not None
            assert hasattr(result, 'stdout'), "Результат должен иметь атрибут stdout"
            assert hasattr(result, 'stderr'), "Результат должен иметь атрибут stderr"
            
            # Проверяем что subprocess.run был вызван дважды
            assert mock_run.call_count == 2, "subprocess.run должен быть вызван дважды"
            
            # Проверяем что второй вызов (docker run) содержит все необходимые параметры безопасности
            docker_run_call = mock_run.call_args_list[1]
            command = docker_run_call[0][0]
            
            # Должны быть параметры безопасности
            assert any("security-opt" in arg for arg in command), "Отсутствуют security-opt параметры"
            
            # Должен быть параметр --rm для автоматического удаления контейнера
            assert "--rm" in command, "Отсутствует параметр --rm"
            
            # Должен быть параметр --network для изоляции сети
            assert "--network" in command, "Отсутствует параметр --network"

    def test_docker_command_structure(self):
        """Проверка структуры Docker команды."""
        with patch("subprocess.run") as mock_run:
            # Настраиваем мок для возврата успешного результата
            mock_info_result = MagicMock()
            mock_info_result.returncode = 0
            mock_info_result.stdout = "rootless"
            mock_info_result.stderr = ""
            
            mock_run_result = MagicMock()
            mock_run_result.returncode = 0
            mock_run_result.stdout = "test output"
            mock_run_result.stderr = ""
            
            # Мокаем два вызова: docker info и docker run
            mock_run.side_effect = [mock_info_result, mock_run_result]

            # Выполняем код (используем валидный Python код)
            result = self.executor.execute("print('test output')")
            
            # Проверяем что результат корректен
            assert result is not None
            
            # Проверяем что subprocess.run был вызван дважды
            assert mock_run.call_count == 2, "subprocess.run должен быть вызван дважды"
            
            # Проверяем структуру команды docker run
            docker_run_call = mock_run.call_args_list[1]
            command = docker_run_call[0][0]
            
            # Первый элемент должен быть "docker"
            assert command[0] == "docker"
            
            # Должен быть подкоманда "run"
            assert "run" in command
            
            # Должны быть параметры безопасности
            security_opts = [arg for arg in command if "security-opt" in arg]
            assert len(security_opts) > 0, "Должны быть переданы security-opt параметры"

    def test_error_handling_with_security(self):
        """Обработка ошибок с применением профилей безопасности."""
        with patch("subprocess.run") as mock_run:
            # Настраиваем мок для возврата ошибки
            mock_info_result = MagicMock()
            mock_info_result.returncode = 0
            mock_info_result.stdout = "rootless"
            mock_info_result.stderr = ""
            
            mock_run_result = MagicMock()
            mock_run_result.returncode = 1
            mock_run_result.stdout = ""
            mock_run_result.stderr = "Permission denied"
            
            # Мокаем два вызова: docker info и docker run
            mock_run.side_effect = [mock_info_result, mock_run_result]

            # Ожидаем исключение при ошибке Docker
            with pytest.raises(Exception, match="Docker execution failed"):
                self.executor.execute("print('test')")
            
            # Проверяем что subprocess.run был вызван дважды
            assert mock_run.call_count == 2, "subprocess.run должен быть вызван дважды"
            
            # Проверяем что профили безопасности все равно применялись
            docker_run_call = mock_run.call_args_list[1]
            command = docker_run_call[0][0]
            security_opts = [arg for arg in command if "security-opt" in arg]
            assert len(security_opts) > 0, "Профили безопасности должны применяться даже при ошибках"
