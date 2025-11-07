"""Tests for subprocess executor running in container mode."""

from __future__ import annotations

import textwrap

from executor.code_executor import SubprocessCodeExecutor


def test_container_execution_exposes_written_files(monkeypatch) -> None:
    """Ensure files created inside the container are collected on the host."""

    executor = SubprocessCodeExecutor(use_container=True)

    def fake_run_in_container(self, code, env, work_dir, pkg_dir):  # type: ignore[no-untyped-def]
        # Workspace path must be prepended for imports when running inside the container.
        assert "sys.path.insert(0, r'/workspace')" in code

        helper = work_dir / "container_test_module.py"
        helper.write_text("VALUE = 42", encoding="utf-8")

        local_code = code.replace("r'/workspace'", repr(str(work_dir)))

        return self._run_locally(local_code, env, work_dir)

    monkeypatch.setattr(
        SubprocessCodeExecutor,
        "_run_in_container",
        fake_run_in_container,
        raising=False,
    )

    code = textwrap.dedent(
        """
        import container_test_module

        with open('result.txt', 'w', encoding='utf-8') as file:
            file.write(str(container_test_module.VALUE))
        """
    )

    result = executor.execute(code)

    assert result.returncode == 0
    assert result.files["result.txt"] == b"42"
