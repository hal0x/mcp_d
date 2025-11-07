"""Менеджер промтов для HAL AI-агента."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from core.utils.json_io import load_json, save_json
from metrics import AB_ASSIGN, ERRORS

logger = logging.getLogger(__name__)


class PromptManager:
    """Управляет промтами и их настройками для HAL AI-агента."""

    def __init__(self, config_path: str = "config/prompts.yaml"):
        """Инициализация менеджера промтов.
        
        Args:
            config_path: Путь к файлу конфигурации промтов
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Загружает конфигурацию промтов из YAML файла."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Загружена конфигурация промтов из {self.config_path}")
        except Exception as exc:
            logger.error(f"Ошибка загрузки конфигурации промтов: {exc}")
            self.config = {}

    def get_system_prompt(self, prompt_name: str, **variables) -> str:
        """Получает системный промт с подстановкой переменных.
        
        Args:
            prompt_name: Имя промта (base_role, coordinator)
            **variables: Переменные для подстановки
            
        Returns:
            Обработанный промт
        """
        prompt = self.config.get("system_prompts", {}).get(prompt_name, "")
        return self._substitute_variables(prompt, **variables)

    def get_module_prompt(self, module: str, memory_level: str, **variables) -> str:
        """Получает промт модуля для определенного уровня памяти.
        
        Args:
            module: Имя модуля (events, themes, emotions)
            memory_level: Уровень памяти (short_term, long_term, episodic)
            **variables: Переменные для подстановки
            
        Returns:
            Обработанный промт
        """
        prompt = (
            self.config
            .get("module_prompts", {})
            .get(module, {})
            .get(memory_level, "")
        )
        return self._substitute_variables(prompt, **variables)

    def get_tool_prompt(self, tool: str, prompt_name: str, **variables) -> str:
        """Получает промт инструмента.
        
        Args:
            tool: Имя инструмента (search, code, planning)
            prompt_name: Имя промта инструмента
            **variables: Переменные для подстановки
            
        Returns:
            Обработанный промт
        """
        prompt = (
            self.config
            .get("tool_prompts", {})
            .get(tool, {})
            .get(prompt_name, "")
        )
        return self._substitute_variables(prompt, **variables)

    def get_adaptation_prompt(self, prompt_name: str, **variables) -> str:
        """Получает промт адаптации.
        
        Args:
            prompt_name: Имя промта (learning, self_analysis)
            **variables: Переменные для подстановки
            
        Returns:
            Обработанный промт
        """
        prompt = self.config.get("adaptation_prompts", {}).get(prompt_name, "")
        return self._substitute_variables(prompt, **variables)
    
    def select_prompt_variant(self, experiment_name: str, user_id: str, weights: Optional[Dict[str, float]] = None) -> str:
        """Выбирает вариант промта для A/B тестирования.
        
        Args:
            experiment_name: Название эксперимента
            user_id: ID пользователя для sticky assignment
            weights: Веса вариантов (по умолчанию равномерное распределение)
            
        Returns:
            Выбранный вариант промта
        """
        try:
            # Простая sticky assignment на основе хеша user_id
            import hashlib
            user_hash = int(hashlib.md5(f"{experiment_name}_{user_id}".encode()).hexdigest(), 16)
            
            # Получаем варианты из конфигурации
            experiment_config = self.config.get("experiments", {}).get(experiment_name, {})
            variants = experiment_config.get("variants", ["default"])
            
            if not variants:
                variants = ["default"]
            
            # Выбираем вариант на основе хеша
            variant_index = user_hash % len(variants)
            selected_variant = variants[variant_index]
            
            # Записываем метрику назначения
            AB_ASSIGN.labels(experiment=experiment_name, variant=selected_variant).inc()
            
            return selected_variant
        except Exception as e:
            logger.error(f"Ошибка выбора варианта промта для эксперимента {experiment_name}: {e}")
            ERRORS.labels(component="prompt", etype=type(e).__name__).inc()
            return "default"

    def get_module_settings(self, module: str) -> Dict[str, Any]:
        """Получает настройки модуля.
        
        Args:
            module: Имя модуля
            
        Returns:
            Настройки модуля
        """
        return self.config.get("module_settings", {}).get(module, {})

    def get_tool_settings(self, tool: str) -> Dict[str, Any]:
        """Получает настройки инструмента.
        
        Args:
            tool: Имя инструмента
            
        Returns:
            Настройки инструмента
        """
        return self.config.get("tool_settings", {}).get(tool, {})

    def get_memory_settings(self, memory_level: str) -> Dict[str, Any]:
        """Получает настройки уровня памяти.
        
        Args:
            memory_level: Уровень памяти
            
        Returns:
            Настройки уровня памяти
        """
        return self.config.get("memory_settings", {}).get(memory_level, {})

    def get_system_settings(self) -> Dict[str, Any]:
        """Получает системные настройки.
        
        Returns:
            Системные настройки
        """
        return self.config.get("system", {})

    def _substitute_variables(self, prompt: str, **variables) -> str:
        """Подставляет переменные в промт.
        
        Args:
            prompt: Исходный промт
            **variables: Переменные для подстановки
            
        Returns:
            Промт с подставленными переменными
        """
        if not prompt:
            return ""
        
        # Добавляем стандартные переменные
        default_variables = {
            "current_time": datetime.now().isoformat(),
            "user_name": "пользователь",
            "timezone": "Asia/Bangkok",
        }
        default_variables.update(variables)
        
        try:
            return prompt.format(**default_variables)
        except KeyError as exc:
            logger.warning(f"Не найдена переменная {exc} в промте")
            return prompt

    def get_available_modules(self) -> List[str]:
        """Возвращает список доступных модулей.
        
        Returns:
            Список модулей
        """
        return list(self.config.get("module_prompts", {}).keys())

    def get_available_tools(self) -> List[str]:
        """Возвращает список доступных инструментов.
        
        Returns:
            Список инструментов
        """
        return list(self.config.get("tool_prompts", {}).keys())

    def get_available_memory_levels(self) -> List[str]:
        """Возвращает список доступных уровней памяти.
        
        Returns:
            Список уровней памяти
        """
        return ["short_term", "long_term", "episodic"]

    def reload_config(self) -> None:
        """Перезагружает конфигурацию из файла."""
        self._load_config()
        logger.info("Конфигурация промтов перезагружена")

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Обновляет конфигурацию и сохраняет в файл.
        
        Args:
            updates: Обновления конфигурации
        """
        try:
            # Обновляем конфигурацию
            self._deep_update(self.config, updates)
            
            # Сохраняем в файл
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"Конфигурация промтов обновлена и сохранена в {self.config_path}")
        except Exception as exc:
            logger.error(f"Ошибка обновления конфигурации промтов: {exc}")

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Рекурсивно обновляет словарь.
        
        Args:
            base_dict: Базовый словарь
            update_dict: Словарь с обновлениями
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def validate_config(self) -> List[str]:
        """Валидирует конфигурацию промтов.
        
        Returns:
            Список ошибок валидации
        """
        errors = []
        
        # Проверяем обязательные секции
        required_sections = ["system_prompts", "module_prompts", "tool_prompts", "adaptation_prompts"]
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Отсутствует обязательная секция: {section}")
        
        # Проверяем системные промты
        system_prompts = self.config.get("system_prompts", {})
        required_system_prompts = ["base_role", "coordinator"]
        for prompt in required_system_prompts:
            if prompt not in system_prompts:
                errors.append(f"Отсутствует обязательный системный промт: {prompt}")
        
        # Проверяем модульные промты
        modules = self.config.get("module_prompts", {})
        memory_levels = self.get_available_memory_levels()
        for module in modules:
            for level in memory_levels:
                if level not in modules[module]:
                    errors.append(f"Отсутствует промт для модуля {module}, уровень {level}")
        
        return errors

    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по промтам.
        
        Returns:
            Статистика промтов
        """
        stats = {
            "total_prompts": 0,
            "system_prompts": len(self.config.get("system_prompts", {})),
            "module_prompts": 0,
            "tool_prompts": 0,
            "adaptation_prompts": len(self.config.get("adaptation_prompts", {})),
            "modules": list(self.config.get("module_prompts", {}).keys()),
            "tools": list(self.config.get("tool_prompts", {}).keys()),
        }
        
        # Подсчитываем модульные промты
        for module_prompts in self.config.get("module_prompts", {}).values():
            stats["module_prompts"] += len(module_prompts)
        
        # Подсчитываем промты инструментов
        for tool_prompts in self.config.get("tool_prompts", {}).values():
            stats["tool_prompts"] += len(tool_prompts)
        
        stats["total_prompts"] = (
            stats["system_prompts"] + 
            stats["module_prompts"] + 
            stats["tool_prompts"] + 
            stats["adaptation_prompts"]
        )
        
        return stats
