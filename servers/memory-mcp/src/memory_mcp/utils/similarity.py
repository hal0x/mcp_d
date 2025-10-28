#!/usr/bin/env python3
"""
Утилиты для вычисления схожести в векторном поиске
"""

from typing import List

import numpy as np


def normalize_similarity_scores(distances: List[float]) -> List[float]:
    """
    Нормализует расстояния в схожесть с хорошим различием между результатами.

    Args:
        distances: Список расстояний от ChromaDB

    Returns:
        Список нормализованных значений схожести [0, 1]
    """
    if not distances:
        return []

    distances_array = np.array(distances)
    min_dist = np.min(distances_array)
    max_dist = np.max(distances_array)

    # Если все расстояния одинаковые, возвращаем равные схожести
    if max_dist == min_dist:
        return [1.0] * len(distances)

    # Нормализация: чем меньше расстояние, тем больше схожесть
    similarities = 1.0 - (distances_array - min_dist) / (max_dist - min_dist)

    return similarities.tolist()


def calculate_similarity_from_distance(
    distance: float, min_distance: float, max_distance: float
) -> float:
    """
    Вычисляет схожесть для одного расстояния на основе диапазона.

    Args:
        distance: Расстояние для нормализации
        min_distance: Минимальное расстояние в выборке
        max_distance: Максимальное расстояние в выборке

    Returns:
        Нормализованное значение схожести [0, 1]
    """
    if max_distance == min_distance:
        return 1.0

    return 1.0 - (distance - min_distance) / (max_distance - min_distance)


def calculate_rank_based_similarity(rank: int, total_results: int) -> float:
    """
    Вычисляет схожесть на основе ранга результата.

    Args:
        rank: Ранг результата (1-based)
        total_results: Общее количество результатов

    Returns:
        Ранговая схожесть [0, 1]
    """
    if total_results <= 1:
        return 1.0

    return (total_results - rank + 1) / total_results
