"""Загрузчик конфигураций для executor."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ExecutorConfigLoader:
    """Загрузчик конфигураций для executor из YAML файлов."""

    def __init__(self, config_dir: str = "config") -> None:
        """Инициализация загрузчика конфигураций.
        
        Args:
            config_dir: Путь к директории с конфигурациями
        """
        self.config_dir = Path(config_dir)
        self._settings_cache: Optional[Dict[str, Any]] = None
        self._policies_cache: Optional[Dict[str, Any]] = None

    def load_settings(self) -> Dict[str, Any]:
        """Загружает настройки из settings.yaml."""
        if self._settings_cache is None:
            settings_path = self.config_dir / "settings.yaml"
            with open(settings_path, "r", encoding="utf-8") as f:
                self._settings_cache = yaml.safe_load(f)
        return self._settings_cache

    def load_policies(self) -> Dict[str, Any]:
        """Загружает политики из policies.yaml."""
        if self._policies_cache is None:
            policies_path = self.config_dir / "policies.yaml"
            with open(policies_path, "r", encoding="utf-8") as f:
                self._policies_cache = yaml.safe_load(f)
        return self._policies_cache

    def get_executor_config(self) -> Dict[str, Any]:
        """Получает конфигурацию executor из settings.yaml."""
        settings = self.load_settings()
        return settings.get("executor", {})

    def get_docker_policy(self) -> Dict[str, Any]:
        """Получает политику Docker из policies.yaml."""
        policies = self.load_policies()
        return policies.get("docker", {})

    def get_tool_policy(self, tool_type: str) -> Dict[str, Any]:
        """Получает политику для конкретного типа инструмента.
        
        Args:
            tool_type: Тип инструмента (code, shell, http, docker)
            
        Returns:
            Словарь с настройками политики
        """
        policies = self.load_policies()
        return policies.get(tool_type, {})

    def merge_configs(self) -> Dict[str, Any]:
        """Объединяет конфигурации из settings.yaml и policies.yaml.
        
        Returns:
            Объединенная конфигурация для Docker executor
        """
        executor_config = self.get_executor_config()
        docker_policy = self.get_docker_policy()
        
        # Объединяем конфигурации, приоритет у policies.yaml
        merged = executor_config.copy()
        if "docker" in executor_config:
            merged.update(executor_config["docker"])
        merged.update(docker_policy)
        
        return merged


# Глобальный экземпляр загрузчика
config_loader = ExecutorConfigLoader()
