from pathlib import Path

from executor import ToolPolicy
from security import PolicyEngine
from tools import Tool


def test_policy_engine_missing_file_returns_default(tmp_path: Path) -> None:
    cfg = tmp_path / "missing.yaml"
    engine = PolicyEngine(cfg)
    policy = engine.get_policy(Tool.CODE)
    assert isinstance(policy, ToolPolicy)


def test_policy_engine_ignores_unknown_tool(tmp_path: Path) -> None:
    cfg = tmp_path / "policies.yaml"
    cfg.write_text("unknown_tool:\n  max_wall_time_s: 2\n", encoding="utf-8")
    engine = PolicyEngine(cfg)
    policy = engine.get_policy(Tool.CODE)
    assert isinstance(policy, ToolPolicy)
    assert getattr(policy, "max_wall_time_s", None) != 2
