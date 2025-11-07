from types import SimpleNamespace

import subprocess

from executor.code_executor import SubprocessCodeExecutor, ToolPolicy


def test_execute_applies_tool_policy_limits(monkeypatch):
    executor = SubprocessCodeExecutor(timeout=5.0, memory_limit=10_000_000, cpu_limit=2.0)
    policy = ToolPolicy(max_wall_time_s=0.5, max_mem_mb=1, cpu_quota=0.25)

    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        captured["memory_limit"] = executor.memory_limit
        captured["cpu_limit"] = executor.cpu_limit
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute("print('hi')", policy=policy)

    assert result.returncode == 0
    assert captured["timeout"] == policy.max_wall_time_s
    assert captured["memory_limit"] == policy.max_mem_mb * 1024 * 1024
    assert captured["cpu_limit"] == policy.cpu_quota
    assert callable(captured["preexec_fn"])
    assert executor.timeout == 5.0
    assert executor.memory_limit == 10_000_000
    assert executor.cpu_limit == 2.0


def test_execute_applies_policy_in_container(monkeypatch):
    executor = SubprocessCodeExecutor(
        timeout=5.0,
        memory_limit=10_000_000,
        cpu_limit=2.0,
        use_container=True,
        container_runtime="docker",
    )
    policy = ToolPolicy(max_wall_time_s=1.5, max_mem_mb=2, cpu_quota=1.0)

    captured: dict[str, object] = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        captured["memory_limit"] = executor.memory_limit
        captured["cpu_limit"] = executor.cpu_limit
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute("print('hi')", policy=policy)

    assert result.returncode == 0
    assert captured["timeout"] == policy.max_wall_time_s
    assert "--memory" in captured["cmd"]
    memory_index = captured["cmd"].index("--memory") + 1
    assert captured["cmd"][memory_index] == f"{policy.max_mem_mb * 1024 * 1024}b"
    assert "--cpus" in captured["cmd"]
    cpus_index = captured["cmd"].index("--cpus") + 1
    assert captured["cmd"][cpus_index] == str(policy.cpu_quota)
    assert executor.timeout == 5.0
    assert executor.memory_limit == 10_000_000
    assert executor.cpu_limit == 2.0
