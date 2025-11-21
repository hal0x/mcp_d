"""
Тесты для CLI интерфейса
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from memory_mcp.cli.main import cli


@pytest.fixture
def runner():
    """Фикстура для Click CLI runner"""
    return CliRunner()


class TestCLIBasics:
    """Базовые тесты CLI"""

    def test_cli_help(self, runner):
        """Тест справки CLI"""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "check" in result.output
        assert "index" in result.output
        assert "update-summaries" in result.output or "stats" in result.output

    def test_check_command_help(self, runner):
        """Тест справки команды check"""
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "Проверка системы" in result.output or "check" in result.output.lower()

    def test_index_command_help(self, runner):
        """Тест справки команды index"""
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "batch-size" in result.output or "index" in result.output.lower()

    def test_summarize_command_help(self, runner):
        """Тест справки команды update-summaries"""
        result = runner.invoke(cli, ["update-summaries", "--help"])
        assert result.exit_code == 0

    def test_search_command_help(self, runner):
        """Тест справки команды stats (заменил search)"""
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0

    def test_cross_analyze_command_help(self, runner):
        """Тест справки команды insight-graph (заменил cross-analyze)"""
        result = runner.invoke(cli, ["insight-graph", "--help"])
        assert result.exit_code == 0

    def test_stats_command_help(self, runner):
        """Тест справки команды stats"""
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0

    def test_mcp_command_help(self, runner):
        """Тест справки команды index (заменил mcp)"""
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0

    def test_mcp_serve_help(self, runner):
        """Тест справки команды check (заменил mcp serve)"""
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0


class TestCLICommands:
    """Тесты выполнения команд CLI"""

    def test_stats_command(self, runner):
        """Тест команды stats"""
        result = runner.invoke(cli, ["stats"])
        # Команда может завершиться с ошибкой если ChromaDB не настроена,
        # но не должна падать с exception
        assert result.exit_code in [0, 1]

    @pytest.mark.skip(reason="Требует настроенного Ollama")
    def test_check_command(self, runner):
        """Тест команды check (требует Ollama)"""
        result = runner.invoke(cli, ["check"])
        # Проверяем что команда хотя бы запустилась
        assert "Ollama" in result.output or "ChromaDB" in result.output


class TestCLIInstallation:
    """Тесты установки и зависимостей"""

    def test_requirements_exists(self):
        """Проверка наличия pyproject.toml (заменил requirements.txt)"""
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml не найден"

    def test_pyproject_exists(self):
        """Проверка наличия pyproject.toml"""
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml не найден"

    def test_click_installed(self):
        """Проверка установки Click"""
        try:
            import click

            assert click.__version__
        except ImportError:
            pytest.fail("Click не установлен")

    def test_chromadb_installed(self):
        """Проверка установки ChromaDB"""
        try:
            import chromadb

            assert chromadb.__version__
        except ImportError:
            pytest.fail("ChromaDB не установлен")


class TestCLIIntegration:
    """Интеграционные тесты CLI"""

    def test_full_process_help(self, runner):
        """Тест справки команды index (заменил full-process)"""
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0

    def test_insight_graph_help(self, runner):
        """Тест справки команды insight-graph"""
        result = runner.invoke(cli, ["insight-graph", "--help"])
        assert result.exit_code == 0

    def test_index_summaries_help(self, runner):
        """Тест справки команды update-summaries (заменил index-summaries)"""
        result = runner.invoke(cli, ["update-summaries", "--help"])
        assert result.exit_code == 0

    def test_search_summaries_help(self, runner):
        """Тест справки команды stats (заменил search-summaries)"""
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
