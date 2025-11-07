from pathlib import Path

from executor import ToolPolicy
from security import PolicyEngine
from tools import Tool


def test_policy_engine_loads_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "policies.yaml"
    cfg.write_text("code:\n  max_wall_time_s: 2\n", encoding="utf-8")
    engine = PolicyEngine(cfg)
    policy = engine.get_policy(Tool.CODE)
    assert isinstance(policy, ToolPolicy)
    assert policy.max_wall_time_s == 2
