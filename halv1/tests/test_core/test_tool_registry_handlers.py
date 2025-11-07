from tools import Tool
from tools.registry import ToolRegistry, register_builtin_handlers


def _dummy(step) -> dict[str, str]:
    return {"stdout": "ok"}


def test_register_builtin_handlers_sets_models() -> None:
    registry = ToolRegistry()
    register_builtin_handlers(
        registry,
        code=_dummy,
        search=_dummy,
        file_io=_dummy,
        shell=_dummy,
        http=_dummy,
    )
    assert registry.get(Tool.SEARCH) is _dummy
    assert registry.get_model(Tool.HTTP).__name__ == "HttpArgs"
