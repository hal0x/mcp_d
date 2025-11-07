#!/usr/bin/env python3
"""Фабрика для создания контекстно-осведомленных LLM клиентов."""

from __future__ import annotations

from typing import Any, Dict

from .factory import create_llm_client
from .context_aware_client import ContextAwareWrapper, ContextAwareCodeGenerator, ContextAwareSearchClient


def create_context_aware_client(
    provider: str,
    llm_cfg: Dict[str, Any] | None = None,
    ollama_cfg: Dict[str, Any] | None = None,
) -> ContextAwareWrapper:
    """Создать контекстно-осведомленный LLM клиент.
    
    Parameters
    ----------
    provider: str
        Провайдер LLM ("ollama" или "lmstudio")
    llm_cfg: Dict[str, Any], optional
        Общая конфигурация LLM
    ollama_cfg: Dict[str, Any], optional
        Специфичная конфигурация Ollama
        
    Returns
    -------
    ContextAwareWrapper
        Обертка с поддержкой переиспользования контекста
    """
    # Создаем базовый клиент
    base_client = create_llm_client(provider, llm_cfg, ollama_cfg)
    
    # Оборачиваем в контекстно-осведомленную обертку
    return ContextAwareWrapper(base_client)


def create_context_aware_code_generator(
    provider: str,
    llm_cfg: Dict[str, Any] | None = None,
    ollama_cfg: Dict[str, Any] | None = None,
) -> ContextAwareCodeGenerator:
    """Создать контекстно-осведомленный генератор кода.
    
    Parameters
    ----------
    provider: str
        Провайдер LLM ("ollama" или "lmstudio")
    llm_cfg: Dict[str, Any], optional
        Общая конфигурация LLM
    ollama_cfg: Dict[str, Any], optional
        Специфичная конфигурация Ollama
        
    Returns
    -------
    ContextAwareCodeGenerator
        Генератор кода с поддержкой переиспользования контекста
    """
    client = create_context_aware_client(provider, llm_cfg, ollama_cfg)
    return ContextAwareCodeGenerator(client)


def create_context_aware_search_client(
    provider: str,
    llm_cfg: Dict[str, Any] | None = None,
    ollama_cfg: Dict[str, Any] | None = None,
) -> ContextAwareSearchClient:
    """Создать контекстно-осведомленный поисковый клиент.
    
    Parameters
    ----------
    provider: str
        Провайдер LLM ("ollama" или "lmstudio")
    llm_cfg: Dict[str, Any], optional
        Общая конфигурация LLM
    ollama_cfg: Dict[str, Any], optional
        Специфичная конфигурация Ollama
        
    Returns
    -------
    ContextAwareSearchClient
        Поисковый клиент с поддержкой переиспользования контекста
    """
    client = create_context_aware_client(provider, llm_cfg, ollama_cfg)
    return ContextAwareSearchClient(client)
