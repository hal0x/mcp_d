#!/usr/bin/env python3
"""
Модули для работы с сущностями
"""

from .entity_extraction import EntityExtractor
from .entity_dictionary import EntityDictionary, get_entity_dictionary

__all__ = [
    "EntityExtractor",
    "EntityDictionary",
    "get_entity_dictionary",
]

