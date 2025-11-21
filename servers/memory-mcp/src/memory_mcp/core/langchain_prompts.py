"""LangChain PromptTemplate менеджер для работы с промптами."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
    from langchain.prompts.pipeline import PipelinePromptTemplate
    from langchain_core.prompts import BasePromptTemplate
except ImportError:
    PromptTemplate = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    FewShotPromptTemplate = None  # type: ignore
    PipelinePromptTemplate = None  # type: ignore
    BasePromptTemplate = None  # type: ignore

logger = logging.getLogger(__name__)


class LangChainPromptManager:
    """Менеджер промптов на базе LangChain PromptTemplate.
    
    Предоставляет интерфейс, совместимый с PromptTemplateManager,
    но использует LangChain PromptTemplate для валидации и форматирования.
    """

    def __init__(self, base_dir: Path, use_langchain: bool = True):
        """Инициализация менеджера промптов.
        
        Args:
            base_dir: Базовая директория с шаблонами промптов
            use_langchain: Использовать LangChain PromptTemplate (True) или простой формат (False)
        """
        self.base_dir = base_dir
        self.use_langchain = use_langchain and PromptTemplate is not None
        self.cache: Dict[str, str] = {}
        self.template_cache: Dict[str, BasePromptTemplate] = {}

    def load(self, name: str) -> str:
        """Загрузка шаблона промпта из файла.
        
        Args:
            name: Имя шаблона (без расширения)
            
        Returns:
            Содержимое шаблона как строка
        """
        if name in self.cache:
            return self.cache[name]

        possible_extensions = [".txt", ".md", ".jinja", ""]
        errors = []

        for ext in possible_extensions:
            path = self.base_dir / f"{name}{ext}"
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8")
                self.cache[name] = content
                return content
            except Exception as exc:
                errors.append((path, exc))

        if errors:
            for path, exc in errors:
                logger.warning(
                    "Не удалось прочитать шаблон %s (%s): %s", name, path, exc
                )
        else:
            logger.warning(
                "Промпт %s не найден (%s + %s)",
                name,
                self.base_dir,
                possible_extensions,
            )

        default = ""
        self.cache[name] = default
        return default

    def load_template(self, name: str, template_type: str = "prompt") -> Optional[BasePromptTemplate]:
        """Загрузка LangChain PromptTemplate из файла.
        
        Args:
            name: Имя шаблона
            template_type: Тип шаблона ("prompt", "chat", "few_shot")
            
        Returns:
            LangChain PromptTemplate или None
        """
        if not self.use_langchain:
            return None
        
        if name in self.template_cache:
            return self.template_cache[name]
        
        template_str = self.load(name)
        if not template_str:
            return None
        
        try:
            if template_type == "chat":
                if ChatPromptTemplate is None:
                    logger.warning("ChatPromptTemplate не доступен, используем обычный PromptTemplate")
                    template = PromptTemplate.from_template(template_str)
                else:
                    # Пытаемся определить формат чата
                    if "System:" in template_str or "system:" in template_str.lower():
                        # Многошаговый чат промпт
                        template = ChatPromptTemplate.from_messages([
                            ("system", self._extract_system_message(template_str)),
                            ("human", self._extract_human_message(template_str)),
                        ])
                    else:
                        template = ChatPromptTemplate.from_messages([("human", template_str)])
            elif template_type == "few_shot":
                if FewShotPromptTemplate is None:
                    logger.warning("FewShotPromptTemplate не доступен, используем обычный PromptTemplate")
                    template = PromptTemplate.from_template(template_str)
                else:
                    # Few-shot требует примеры, которые нужно передать отдельно
                    template = PromptTemplate.from_template(template_str)
            else:
                template = PromptTemplate.from_template(template_str)
            
            self.template_cache[name] = template
            return template
        except Exception as e:
            logger.warning(f"Не удалось создать LangChain шаблон для {name}: {e}")
            return None

    def format(self, name: str, **kwargs: Any) -> str:
        """Форматирование промпта с переменными.
        
        Args:
            name: Имя шаблона
            **kwargs: Переменные для подстановки
            
        Returns:
            Отформатированный промпт
        """
        if self.use_langchain:
            template = self.load_template(name)
            if template:
                try:
                    # LangChain валидирует переменные
                    return template.format(**kwargs)
                except Exception as e:
                    logger.warning(f"Ошибка форматирования LangChain шаблона {name}: {e}, используем простой формат")
        
        # Fallback на простой формат
        template_str = self.load(name)
        try:
            return template_str.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Отсутствует переменная {e} в шаблоне {name}")
            return template_str

    def format_template(self, name: str, template_type: str = "prompt", **kwargs: Any) -> str:
        """Форматирование LangChain шаблона.
        
        Args:
            name: Имя шаблона
            template_type: Тип шаблона ("prompt", "chat", "few_shot")
            **kwargs: Переменные для подстановки
            
        Returns:
            Отформатированный промпт
        """
        if self.use_langchain:
            template = self.load_template(name, template_type)
            if template:
                try:
                    return template.format(**kwargs)
                except Exception as e:
                    logger.warning(f"Ошибка форматирования LangChain шаблона {name}: {e}")
        
        # Fallback
        return self.format(name, **kwargs)

    def _extract_system_message(self, template_str: str) -> str:
        """Извлечение системного сообщения из шаблона."""
        lines = template_str.split("\n")
        system_lines = []
        in_system = False
        
        for line in lines:
            if "System:" in line or "system:" in line.lower():
                in_system = True
                # Убираем маркер
                line = line.replace("System:", "").replace("system:", "").strip()
                if line:
                    system_lines.append(line)
            elif in_system and (line.strip().startswith("Human:") or line.strip().startswith("human:")):
                break
            elif in_system:
                system_lines.append(line)
        
        return "\n".join(system_lines).strip()

    def _extract_human_message(self, template_str: str) -> str:
        """Извлечение человеческого сообщения из шаблона."""
        lines = template_str.split("\n")
        human_lines = []
        in_human = False
        
        for line in lines:
            if "Human:" in line or "human:" in line.lower():
                in_human = True
                # Убираем маркер
                line = line.replace("Human:", "").replace("human:", "").strip()
                if line:
                    human_lines.append(line)
            elif in_human:
                human_lines.append(line)
        
        return "\n".join(human_lines).strip() if human_lines else template_str

    def create_prompt_template(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        template_type: str = "prompt",
    ) -> Optional[BasePromptTemplate]:
        """Создание LangChain PromptTemplate из строки.
        
        Args:
            template: Текст шаблона
            input_variables: Список переменных (опционально, будет определен автоматически)
            template_type: Тип шаблона ("prompt", "chat")
            
        Returns:
            LangChain PromptTemplate или None
        """
        if not self.use_langchain:
            return None
        
        try:
            if template_type == "chat" and ChatPromptTemplate is not None:
                if input_variables:
                    return ChatPromptTemplate.from_messages([
                        ("human", template)
                    ])
                else:
                    return ChatPromptTemplate.from_template(template)
            else:
                if input_variables:
                    return PromptTemplate(input_variables=input_variables, template=template)
                else:
                    return PromptTemplate.from_template(template)
        except Exception as e:
            logger.warning(f"Не удалось создать LangChain шаблон: {e}")
            return None

