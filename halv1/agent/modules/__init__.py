"""Модули анализа памяти для HAL AI-агента."""

from .base import PromptModule, ModuleResult, RequestContext
from .events_module import EventsModule
from .themes_module import ThemesModule
from .emotions_module import EmotionsModule

__all__ = [
    "PromptModule",
    "ModuleResult",
    "RequestContext",
    "EventsModule",
    "ThemesModule",
    "EmotionsModule"
]
