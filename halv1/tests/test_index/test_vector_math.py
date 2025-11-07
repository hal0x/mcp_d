import pytest

from utils.vector_math import cosine_similarity


def test_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


@pytest.mark.parametrize("a,b", [([1, 0], [1, 0]), ([1, 2, 3], [4, 5, 6])])
def test_cosine_similarity_symmetry_and_bounds(a: list[float], b: list[float]) -> None:
    first = cosine_similarity(a, b)
    second = cosine_similarity(b, a)
    assert first == pytest.approx(second)
    assert -1.0 <= first <= 1.0
