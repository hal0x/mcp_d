from __future__ import annotations

import atexit
import json
import logging
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, List

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from ..config import get_settings
from ..services.docker_executor import DockerExecutor
from ..services.script_store import ScriptStore

logger = logging.getLogger(__name__)


class RunCodeArgs(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _parse_from_json(cls, data: Any) -> Any:
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                import ast

                try:
                    parsed = ast.literal_eval(data)
                except Exception as exc:  # pylint: disable=broad-except
                    raise ValueError("run_code payload must be JSON object") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("run_code payload must be JSON object") from None
                return parsed
        return data

    code: str | None = Field(
        default=None, description="Source code to execute inside the container"
    )
    language: str = Field(
        default="python",
        description="Script language",
        pattern="^(python|bash|sh|shell|node)$",
        examples=["python", "bash", "node"],
    )
    image: str | None = Field(default=None, description="Docker image override")
    command: str | None = Field(
        default=None, description="Custom command; use '{script}' to refer to file"
    )
    network_enabled: bool | None = Field(
        default=None, description="Override default network policy"
    )
    timeout_seconds: PositiveInt = Field(
        default=120, description="Max execution time in seconds"
    )
    env: list[str] | None = Field(
        default=None, description="Environment variables: KEY=VALUE"
    )
    memory: str | None = Field(
        default=None,
        description="Per-call memory limit (Docker format: digits with optional k/m/g).",
        pattern="^\\d+[kKmMgG]?$",
        examples=["256m", "1g"],
    )
    cpus: str | None = Field(
        default=None,
        description="Per-call CPU limit (decimal value, e.g. 0.5).",
        pattern="^\\d+(\\.\\d+)?$",
        examples=["0.5", "1", "2.5"],
    )
    readonly_fs: bool | None = Field(
        default=None,
        description="Request read-only filesystem (cannot disable if globally on)",
    )
    dependencies: list[str] | None = Field(
        default=None,
        description="Python packages to install via pip before execution",
        examples=[["requests"], ["numpy", "pandas"]],
    )
    out_artifacts_path: str | None = Field(
        default=None,
        description="Host path to store produced artifacts (relative paths resolve against server cwd)",
        examples=["./artifacts", "/tmp/run-output"],
    )
    script_path: str | None = Field(
        default=None,
        description="Path on host to an existing script file; if provided, its contents are executed",
    )
    save_name: str | None = Field(
        default=None, description="Save script under this name when execution succeeds"
    )

    @field_validator("env")
    @classmethod
    def _validate_env(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        for item in value:
            if "=" not in item:
                raise ValueError(f"Invalid env format: '{item}'. Use KEY=VALUE.")
        return value

    @field_validator("memory")
    @classmethod
    def _validate_memory(cls, value: str | None) -> str | None:
        if value is None:
            return value
        # Basic validation: number optionally followed by k/m/g/K/M/G
        import re

        if not re.fullmatch(r"\d+[kKmMgG]?", value):
            raise ValueError("memory must be like '512m', '1g', or bytes as integer")
        return value

    @field_validator("cpus")
    @classmethod
    def _validate_cpus(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            float(value)
        except Exception:
            raise ValueError(
                "cpus must be a number as string, e.g. '0.5' or '1'"
            ) from None
        return value

    @field_validator("dependencies")
    @classmethod
    def _validate_deps(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            return None
        return cleaned

    @field_validator("out_artifacts_path")
    @classmethod
    def _validate_out_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("script_path")
    @classmethod
    def _validate_script_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"script_path does not exist or is not a file: {value}")
        return str(path)

    @field_validator("save_name")
    @classmethod
    def _validate_save_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _ensure_source_present(self) -> RunCodeArgs:
        if not self.code and not self.script_path:
            raise ValueError("Either 'code' or 'script_path' must be provided")
        return self


class RunSavedArgs(BaseModel):
    name: str = Field(..., description="Имя или слаг сохранённого скрипта")
    timeout_seconds: PositiveInt = Field(
        default=120, description="Max execution time in seconds"
    )
    network_enabled: bool | None = Field(
        default=None, description="Override default network policy"
    )
    env: list[str] | None = Field(
        default=None,
        description="Environment variables as list of 'KEY=VALUE' strings",
        examples=[["API_KEY=secret", "DEBUG=true"]],
    )
    memory: str | None = Field(
        default=None,
        description="Per-call memory limit (Docker format: digits with optional k/m/g)",
        pattern="^\\d+[kKmMgG]?$",
        examples=["256m", "1g"],
    )
    cpus: str | None = Field(
        default=None,
        description="Per-call CPU limit (decimal value, e.g. 0.5)",
        pattern="^\\d+(\\.\\d+)?$",
        examples=["0.5", "1", "2.5"],
    )
    readonly_fs: bool | None = Field(
        default=None, description="Request read-only filesystem"
    )
    dependencies: list[str] | None = Field(
        default=None,
        description="Python packages to install via pip before execution",
        examples=[["requests"], ["numpy", "pandas"]],
    )
    out_artifacts_path: str | None = Field(
        default=None,
        description="Host path to store produced artifacts",
        examples=["./artifacts"],
    )
    save_name: str | None = Field(
        default=None,
        description="Сохранить результат под новым именем после успешного запуска",
    )

    _validate_env = RunCodeArgs._validate_env
    _validate_memory = RunCodeArgs._validate_memory
    _validate_cpus = RunCodeArgs._validate_cpus
    _validate_deps = RunCodeArgs._validate_deps


class RunScriptsBatchRequest(BaseModel):
    scripts: List[RunSavedArgs] = Field(
        ..., description="List of saved script executions to run sequentially"
    )


class ScheduleScriptArgs(BaseModel):
    script: RunSavedArgs = Field(..., description="Saved script execution settings")
    run_at: datetime | None = Field(
        default=None,
        description="ISO-8601 timestamp (UTC) when the script should run."
        " If omitted, delay_seconds must be provided.",
    )
    delay_seconds: PositiveInt | None = Field(
        default=None,
        description="Schedule the script to run after this many seconds."
        " Ignored when run_at is provided.",
    )
    repeat_seconds: PositiveInt | None = Field(
        default=None,
        description="Optional repeat interval in seconds for recurring runs.",
    )

    @model_validator(mode="after")
    def _validate_schedule(self) -> "ScheduleScriptArgs":
        if self.run_at is None and self.delay_seconds is None:
            raise ValueError("Either run_at or delay_seconds must be provided")
        return self

    @field_validator("run_at")
    @classmethod
    def _normalize_run_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            # Assume UTC if timezone missing
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class ScheduledJobResult(BaseModel):
    id: str
    name: str
    next_run: datetime | None
    repeat_seconds: PositiveInt | None
    active: bool
    last_run: datetime | None = None
    last_success: bool | None = None
    last_error: str | None = None

    @classmethod
    def from_job(cls, job: dict[str, Any]) -> "ScheduledJobResult":
        return cls(
            id=job["id"],
            name=job["script"].name,
            next_run=job.get("next_run"),
            repeat_seconds=job.get("repeat_seconds"),
            active=job.get("active", False),
            last_run=job.get("last_run"),
            last_success=job.get("last_success"),
            last_error=job.get("last_error"),
        )

    def model_dump_public(self) -> dict[str, Any]:
        def _dt(value: datetime | None) -> str | None:
            if value is None:
                return None
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        data = self.model_dump()
        data["next_run"] = _dt(self.next_run)
        data["last_run"] = _dt(self.last_run)
        return data


class ScriptScheduler:
    def __init__(self, run_saved: callable[[RunSavedArgs], dict[str, Any]]):
        self._run_saved = run_saved
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def schedule(self, args: ScheduleScriptArgs) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        if args.run_at:
            next_run = args.run_at
        else:
            next_run = now + timedelta(seconds=int(args.delay_seconds or 0))
        delay = max(0.0, (next_run - now).total_seconds())

        job = {
            "id": job_id,
            "script": args.script,
            "next_run": next_run,
            "repeat_seconds": args.repeat_seconds,
            "timer": None,
            "active": True,
            "last_run": None,
            "last_success": None,
            "last_error": None,
        }

        timer = threading.Timer(delay, self._execute_job, args=[job_id])
        timer.daemon = True
        job["timer"] = timer

        with self._lock:
            self._jobs[job_id] = job
        timer.start()
        return ScheduledJobResult.from_job(job).model_dump_public()

    def _execute_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            script_args: RunSavedArgs = job["script"]
            repeat = job.get("repeat_seconds")

        try:
            result = self._run_saved(script_args)
            success = True
            error_text = None
            # optionally we could store result but avoid large payload
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Scheduled script failed", extra={"name": script_args.name})
            success = False
            error_text = str(exc)

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["last_run"] = datetime.now(timezone.utc)
            job["last_success"] = success
            job["last_error"] = error_text

            if repeat:
                next_run = datetime.now(timezone.utc) + timedelta(seconds=int(repeat))
                job["next_run"] = next_run
                timer = threading.Timer(float(repeat), self._execute_job, args=[job_id])
                timer.daemon = True
                job["timer"] = timer
                timer.start()
            else:
                job["active"] = False
                job["next_run"] = None
                job["timer"] = None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [ScheduledJobResult.from_job(job) for job in self._jobs.values()]
        return [job.model_dump_public() for job in jobs]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Scheduled job {job_id} not found")
            timer: threading.Timer | None = job.get("timer")
            if timer is not None:
                timer.cancel()
            job["active"] = False
            job["next_run"] = None
            job["timer"] = None
            return ScheduledJobResult.from_job(job).model_dump_public()

    def shutdown(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                timer: threading.Timer | None = job.get("timer")
                if timer is not None:
                    timer.cancel()
            self._jobs.clear()


RunScriptsBatchRequest.model_rebuild()
ScheduleScriptArgs.model_rebuild()
ScheduledJobResult.model_rebuild()


def _normalize_env(env_vars: list[str] | None) -> dict[str, str]:
    if not env_vars:
        return {}
    parsed: dict[str, str] = {}
    for item in env_vars:
        key, value = item.split("=", 1)
        parsed[key.strip()] = value
    return parsed


def _mem_to_bytes(s: str) -> int:
    suf = s[-1].lower() if s[-1].isalpha() else ""
    num = int(s[:-1]) if suf else int(s)
    if not suf:
        return num
    if suf == "k":
        return num * 1024
    if suf == "m":
        return num * 1024 * 1024
    if suf == "g":
        return num * 1024 * 1024 * 1024
    # Unknown suffix, fallback to raw int
    return int(s)


def _choose_stricter_memory(req: str | None, glob: str | None) -> str | None:
    if req and glob:
        return req if _mem_to_bytes(req) <= _mem_to_bytes(glob) else glob
    return req or glob


def _choose_stricter_cpus(req: str | None, glob: str | None) -> str | None:
    if req and glob:
        return req if float(req) <= float(glob) else glob
    return req or glob


@contextmanager
def _maybe_semaphore_ctx(
    sem: threading.BoundedSemaphore | None,
) -> Generator[None, None, None]:
    if sem is None:
        yield
        return
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def bind(mcp: FastMCP) -> None:
    executor = DockerExecutor()
    store = ScriptStore()
    s = get_settings()
    sem: threading.BoundedSemaphore | None = (
        threading.BoundedSemaphore(s.MAX_CONCURRENCY)
        if (s.MAX_CONCURRENCY and s.MAX_CONCURRENCY > 0)
        else None
    )

    def _execute(args: RunCodeArgs) -> dict[str, object]:
        env_map = _normalize_env(args.env)
        env_keys = sorted(env_map.keys())
        deps_count = len(args.dependencies or [])

        if args.script_path:
            source_path = Path(args.script_path)
            try:
                source_code = source_path.read_text(encoding="utf-8")
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    "run_code failed to read script_path=%s", args.script_path
                )
                raise ValueError(
                    f"Failed to read script at '{args.script_path}': {exc}"
                ) from exc
        else:
            source_code = args.code or ""

        if not source_code:
            raise ValueError("No code provided to execute")

        logger.info(
            "run_code request language=%s image=%s script_path=%s code_length=%d timeout=%s network_override=%s "
            "memory=%s cpus=%s readonly_override=%s env_keys=%s dependencies=%d save_name=%s artifacts_path=%s",
            args.language,
            args.image,
            args.script_path,
            len(source_code),
            args.timeout_seconds,
            args.network_enabled,
            args.memory,
            args.cpus,
            args.readonly_fs,
            env_keys,
            deps_count,
            args.save_name,
            args.out_artifacts_path,
        )

        with _maybe_semaphore_ctx(sem):
            # Network: cannot enable if globally disabled
            network_eff = (
                (s.DEFAULT_NETWORK and args.network_enabled)
                if args.network_enabled is not None
                else s.DEFAULT_NETWORK
            )
            # Limits: per-request can only tighten globals
            mem_eff = _choose_stricter_memory(args.memory, s.MEMORY)
            cpus_eff = _choose_stricter_cpus(args.cpus, s.CPUS)
            readonly_eff = s.READONLY_FS or bool(args.readonly_fs)
            logger.debug(
                "run_code effective_limits image=%s network=%s memory=%s cpus=%s readonly=%s",
                args.image or s.DEFAULT_IMAGE,
                network_eff,
                mem_eff,
                cpus_eff,
                readonly_eff,
            )
            try:
                result = executor.run(
                    code=source_code,
                    language=args.language,
                    image=args.image,
                    command=args.command,
                    network=network_eff,
                    timeout=args.timeout_seconds,
                    env=env_map,
                    memory=mem_eff,
                    cpus=cpus_eff,
                    readonly_fs=readonly_eff,
                    dependencies=args.dependencies,
                    out_artifacts_path=args.out_artifacts_path,
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception(
                    "run_code execution error language=%s image=%s script_path=%s timeout=%s",
                    args.language,
                    args.image or s.DEFAULT_IMAGE,
                    args.script_path,
                    args.timeout_seconds,
                )
                raise
            result_dict = result.to_dict()
            if args.save_name and result.exit_code == 0:
                try:
                    saved_meta = store.save(args.save_name, args.language, source_code)
                    result_dict["saved_script"] = saved_meta
                    logger.info(
                        "run_code saved_script name=%s path=%s",
                        args.save_name,
                        saved_meta.get("path"),
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    result_dict["save_error"] = str(exc)
                    logger.error(
                        "run_code save_error name=%s error=%s",
                        args.save_name,
                        exc,
                    )
            logger.info(
                "run_code result language=%s exit_code=%s timed_out=%s artifacts=%d",
                args.language,
                result.exit_code,
                result.timed_out,
                len(result.artifacts),
            )
            return result_dict

    def _build_multi_step_wrapper(step_list: list[str]) -> str:
        if not step_list:
            raise ValueError("steps must not be empty")
        steps_json = json.dumps(step_list)
        template = (
            "import json, sys, contextlib\n"
            "from io import StringIO\n"
            "outputs = []\nerrors = []\nns = {}\n"
            "def run(code):\n"
            "    stdout_buf = StringIO()\n"
            "    stderr_buf = StringIO()\n"
            "    try:\n"
            "        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):\n"
            "            exec(code, ns, ns)\n"
            "        outputs.append(stdout_buf.getvalue().strip())\n"
            "        errors.append(stderr_buf.getvalue().strip())\n"
            "    except Exception as e:\n"
            "        outputs.append(stdout_buf.getvalue().strip())\n"
            "        err_output = stderr_buf.getvalue().strip()\n"
            '        message = f"Step failed: {e}"\n'
            '        errors.append(f"{err_output}\\n{message}" if err_output else message)\n'
            "for code in [STEPS_PLACEHOLDER]:\n    run(code)\n"
            "print(json.dumps(outputs))\n"
            "print(json.dumps(errors), file=sys.stderr)\n"
        )
        return template.replace("[STEPS_PLACEHOLDER]", steps_json)

    @mcp.tool()
    def run_code_multi_step(
        steps: Annotated[
            list[str],
            Field(description="List of Python code fragments executed sequentially"),
        ],
        language: Annotated[
            str,
            Field(
                description="Language for execution (currently only python is supported)",
                pattern="^(python)$",
                examples=["python"],
            ),
        ] = "python",
        image: Annotated[
            str | None, Field(description="Docker image override (optional)")
        ] = None,
        command: Annotated[
            str | None,
            Field(
                description="Custom command to run inside the container; use '{script}' placeholder for file path"
            ),
        ] = None,
        timeout_seconds: Annotated[
            int, Field(description="Max execution time in seconds", ge=1)
        ] = 120,
        network_enabled: Annotated[
            bool | None,
            Field(description="Enable outbound network access for this run"),
        ] = None,
        env: Annotated[
            list[str] | None,
            Field(
                description="Environment variables list, each item formatted as 'KEY=VALUE'",
                examples=[["API_KEY=secret", "DEBUG=true"]],
            ),
        ] = None,
        memory: Annotated[
            str | None,
            Field(
                description="Memory limit for the container (Docker format such as '256m' or '1g')",
                pattern="^\\d+[kKmMgG]?$",
                examples=["256m", "1g"],
            ),
        ] = None,
        cpus: Annotated[
            str | None,
            Field(
                description="CPU quota for the container (decimal value, e.g. '0.5')",
                pattern="^\\d+(\\.\\d+)?$",
                examples=["0.5", "1", "2.5"],
            ),
        ] = None,
        readonly_fs: Annotated[
            bool | None,
            Field(description="Request read-only filesystem for the container"),
        ] = None,
        dependencies: Annotated[
            list[str] | None,
            Field(
                description="Python packages to install via pip before execution",
                examples=[["requests"], ["numpy", "pandas"]],
            ),
        ] = None,
        out_artifacts_path: Annotated[
            str | None,
            Field(
                description="Host path where produced artifacts will be exported",
                examples=["./artifacts", "/tmp/run-output"],
            ),
        ] = None,
        save_name: Annotated[
            str | None,
            Field(description="Persist successful run under this name in script store"),
        ] = None,
    ) -> dict[str, object]:
        """Executes multiple Python snippets sequentially inside one container.

        Example:
            run_code_multi_step(
                steps=[
                    "import math",
                    "result = math.sqrt(16)",
                    "print(result)",
                ]
            )

        Tips:
            - ``steps`` must be a list of strings; each element runs as an isolated step.
            - Import libraries in early steps and reuse them later, e.g. ``\"import requests\"`` then ``\"print(requests.get('https://example.com').status_code)\"``.
        """
        lang = (language or "").strip().lower()
        if lang and lang != "python":
            raise ValueError(
                "run_code_multi_step currently supports only the Python language"
            )
        wrapper_code = _build_multi_step_wrapper(steps)
        args = RunCodeArgs(
            code=wrapper_code,
            language="python",
            image=image,
            command=command,
            timeout_seconds=timeout_seconds,
            network_enabled=network_enabled,
            env=env,
            memory=memory,
            cpus=cpus,
            readonly_fs=readonly_fs,
            dependencies=dependencies,
            out_artifacts_path=out_artifacts_path,
            script_path=None,
            save_name=save_name,
        )
        return _execute(args)

    @mcp.tool()
    def run_code_simple(
        code: Annotated[
            str | None,
            Field(
                description="Inline source code to execute (mutually exclusive with script_path)"
            ),
        ] = None,
        language: Annotated[
            str,
            Field(
                description="Script language",
                pattern="^(python|bash|sh|shell|node)$",
                examples=["python", "bash", "node"],
            ),
        ] = "python",
        image: Annotated[
            str | None, Field(description="Docker image override (optional)")
        ] = None,
        command: Annotated[
            str | None,
            Field(
                description="Custom command for container start; use '{script}' placeholder when providing a script"
            ),
        ] = None,
        timeout_seconds: Annotated[
            int, Field(description="Max execution time in seconds", ge=1)
        ] = 120,
        network_enabled: Annotated[
            bool | None,
            Field(description="Enable outbound network access for this run"),
        ] = None,
        env: Annotated[
            list[str] | None,
            Field(
                description="Environment variables list, each item formatted as 'KEY=VALUE'",
                examples=[["API_KEY=secret", "DEBUG=true"]],
            ),
        ] = None,
        memory: Annotated[
            str | None,
            Field(
                description="Memory limit for the container (Docker format such as '256m' or '1g')",
                pattern="^\\d+[kKmMgG]?$",
                examples=["256m", "1g"],
            ),
        ] = None,
        cpus: Annotated[
            str | None,
            Field(
                description="CPU quota for the container (decimal value, e.g. '0.5')",
                pattern="^\\d+(\\.\\d+)?$",
                examples=["0.5", "1", "2.5"],
            ),
        ] = None,
        readonly_fs: Annotated[
            bool | None,
            Field(description="Request read-only filesystem for the container"),
        ] = None,
        dependencies: Annotated[
            list[str] | None,
            Field(
                description="Python packages to install via pip before execution",
                examples=[["requests"], ["numpy", "pandas"]],
            ),
        ] = None,
        out_artifacts_path: Annotated[
            str | None,
            Field(
                description="Host path where produced artifacts will be exported",
                examples=["./artifacts", "/tmp/run-output"],
            ),
        ] = None,
        script_path: Annotated[
            str | None,
            Field(
                description="Path on host to an existing script file (mutually exclusive with code)"
            ),
        ] = None,
        save_name: Annotated[
            str | None,
            Field(description="Persist successful run under this name in script store"),
        ] = None,
    ) -> dict[str, object]:
        """Executes code inside a container (`code` XOR `script_path` must be provided).

        Example:
            run_code_simple(
                code="print('Hello')",
                env=["DEBUG=true", "API_TOKEN=secret123"],
                dependencies=["requests"],
            )

        Tips:
            - Use ``script_path=\"/workspace/script.py\"`` to run an existing file on disk.
            - Declare dependencies with ``dependencies=[\"numpy\", \"pandas\"]`` to install them before execution.
            - Switch to shell mode via ``language=\"bash\"`` and override ``command`` if you need a custom entrypoint.
            - ``language=\"node\"`` picks the default ``node:20-slim`` image; override ``image``/``command`` for custom runtimes.
            - Always pass environment variables as ``KEY=VALUE`` strings, e.g. ``[\"TZ=UTC\", \"MODE=debug\"]``.
        """
        args = RunCodeArgs(
            code=code,
            language=language,
            image=image,
            command=command,
            timeout_seconds=timeout_seconds,
            network_enabled=network_enabled,
            env=env,
            memory=memory,
            cpus=cpus,
            readonly_fs=readonly_fs,
            dependencies=dependencies,
            out_artifacts_path=out_artifacts_path,
            script_path=script_path,
            save_name=save_name,
        )
        return _execute(args)

    @mcp.tool()
    def list_saved_scripts() -> list[dict[str, str]]:
        """Returns metadata for saved scripts."""
        return store.list()

    @mcp.tool()
    def run_saved_script(args: RunSavedArgs) -> dict[str, object]:
        """Runs a previously saved script with optional overrides.

        Parameters:
            name (str): Script name or slug from ``list_saved_scripts``.
            timeout_seconds (int): Execution timeout (default 120 seconds).
            env (list[str]): Environment variables as ``KEY=VALUE`` strings.
            memory/cpus (str): Docker-style resource limits such as ``256m`` or ``0.5``.
            dependencies (list[str]): Pip packages installed before the run.

        Example:
            run_saved_script(
                args={
                    "name": "analyze-logs",
                    "timeout_seconds": 60,
                    "env": ["MODE=debug"],
                    "dependencies": ["pandas"],
                }
            )
        """
        try:
            meta = store.get(args.name)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        run_args = RunCodeArgs(
            code=None,
            language=meta.get("language", "python"),
            image=None,
            command=None,
            timeout_seconds=args.timeout_seconds,
            network_enabled=args.network_enabled,
            env=args.env,
            memory=args.memory,
            cpus=args.cpus,
            readonly_fs=args.readonly_fs,
            dependencies=args.dependencies,
            out_artifacts_path=args.out_artifacts_path,
            script_path=meta.get("path"),
            save_name=args.save_name,
        )
        return _execute(run_args)

    scheduler = ScriptScheduler(run_saved_script)
    atexit.register(scheduler.shutdown)

    @mcp.tool()
    def run_scripts_batch(request: RunScriptsBatchRequest) -> list[dict[str, object]]:
        """Runs multiple saved scripts sequentially and returns per-script results."""
        results: list[dict[str, object]] = []
        for script_args in request.scripts:
            try:
                payload = run_saved_script(script_args)
                results.append(
                    {
                        "name": script_args.name,
                        "success": True,
                        "result": payload,
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("run_scripts_batch failure name=%s", script_args.name)
                results.append(
                    {
                        "name": script_args.name,
                        "success": False,
                        "error": str(exc),
                    }
                )
        return results

    @mcp.tool()
    def schedule_saved_script(args: ScheduleScriptArgs) -> dict[str, object]:
        """Schedules a saved script for future (and optionally recurring) execution."""
        return scheduler.schedule(args)

    @mcp.tool()
    def list_scheduled_scripts() -> list[dict[str, object]]:
        """Returns metadata for all scheduled script runs."""
        return scheduler.list()

    @mcp.tool()
    def cancel_scheduled_script(job_id: Annotated[str, Field(description="Identifier returned by schedule_saved_script")]) -> dict[str, object]:
        """Cancels a scheduled script run by its identifier."""
        return scheduler.cancel(job_id)

    @mcp.tool()
    def delete_saved_script(
        name: Annotated[
            str,
            Field(
                description="Имя или слаг скрипта, полученный через list_saved_scripts"
            ),
        ],
    ) -> dict[str, str]:
        """Deletes a saved script by name or slug.

        Example:
            delete_saved_script(name="analyze-logs")
        """
        try:
            meta = store.delete(name)
            logger.info(
                "delete_saved_script success name=%s path=%s", name, meta.get("path")
            )
            return meta
        except KeyError as exc:
            logger.warning("delete_saved_script not_found name=%s", name)
            raise ValueError(str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("delete_saved_script error name=%s", name)
            raise ValueError(f"Failed to delete script '{name}': {exc}") from exc

    @mcp.tool()
    def delete_saved_scripts(
        names: Annotated[
            list[str],
            Field(
                description="List of saved script names/slugs to delete",
                min_length=1,
                examples=[["analyze-logs", "daily-report"]],
            ),
        ]
    ) -> list[dict[str, object]]:
        """Deletes multiple saved scripts and returns per-script status."""
        results: list[dict[str, object]] = []
        for name in names:
            try:
                meta = store.delete(name)
                logger.info("delete_saved_scripts success name=%s", name)
                results.append(
                    {"name": name, "success": True, "metadata": meta}
                )
            except KeyError:
                logger.warning("delete_saved_scripts not_found name=%s", name)
                results.append(
                    {"name": name, "success": False, "error": "Script not found"}
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("delete_saved_scripts error name=%s", name)
                results.append(
                    {"name": name, "success": False, "error": str(exc)}
                )
        return results

    @mcp.tool()
    def index_existing_scripts() -> dict[str, object]:
        """Indexes script files on disk that are missing from the ScriptStore.

        Use this tool to capture scripts created outside the MCP workflow.
        """
        try:
            indexed_files = store.index_existing_files()
            logger.info("index_existing_scripts success count=%d", len(indexed_files))
            return {
                "success": True,
                "indexed_count": len(indexed_files),
                "indexed_files": indexed_files,
                "message": f"Successfully indexed {len(indexed_files)} files"
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("index_existing_scripts error")
            return {
                "success": False,
                "indexed_count": 0,
                "indexed_files": [],
                "message": f"Failed to index files: {exc}"
            }

    @mcp.tool()
    def save_temp_script(name: str, language: str, code: str) -> dict[str, object]:
        """Stores a temporary script that expires after three days.

        Args:
            name: Script name for identification.
            language: Runtime language (python, bash, node).
            code: Script contents to persist.
        """
        try:
            meta = store.save_temp(name, language, code)
            logger.info("save_temp_script success name=%s language=%s", name, language)
            return {
                "success": True,
                "script": meta,
                "message": f"Temporary script '{name}' saved successfully"
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("save_temp_script error name=%s", name)
            return {
                "success": False,
                "script": None,
                "message": f"Failed to save temporary script '{name}': {exc}"
            }

    @mcp.tool()
    def list_temp_scripts() -> list[dict[str, str]]:
        """Lists currently saved temporary scripts."""
        try:
            temp_scripts = store.list_temp()
            logger.info("list_temp_scripts success count=%d", len(temp_scripts))
            return temp_scripts
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("list_temp_scripts error")
            raise ValueError(f"Failed to list temporary scripts: {exc}") from exc

    @mcp.tool()
    def promote_temp_script(slug: str) -> dict[str, object]:
        """Promotes a temporary script to a permanent entry.

        Args:
            slug: Temporary script slug selected from ``list_temp_scripts``.
        """
        try:
            meta = store.promote_temp_to_permanent(slug)
            logger.info("promote_temp_script success slug=%s", slug)
            return {
                "success": True,
                "script": meta,
                "message": f"Temporary script '{slug}' promoted to permanent"
            }
        except KeyError as exc:
            logger.warning("promote_temp_script not_found slug=%s", slug)
            return {
                "success": False,
                "script": None,
                "message": f"Temporary script '{slug}' not found"
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("promote_temp_script error slug=%s", slug)
            return {
                "success": False,
                "script": None,
                "message": f"Failed to promote temporary script '{slug}': {exc}"
            }

    @mcp.tool()
    def promote_temp_scripts(
        slugs: Annotated[
            list[str],
            Field(
                description="List of temporary script slugs to promote",
                min_length=1,
                examples=[["tmp-analyze", "tmp-cleanup"]],
            ),
        ]
    ) -> list[dict[str, object]]:
        """Promotes multiple temporary scripts to permanent entries."""
        results: list[dict[str, object]] = []
        for slug in slugs:
            try:
                meta = store.promote_temp_to_permanent(slug)
                logger.info("promote_temp_scripts success slug=%s", slug)
                results.append(
                    {"slug": slug, "success": True, "script": meta}
                )
            except KeyError:
                logger.warning("promote_temp_scripts not_found slug=%s", slug)
                results.append(
                    {"slug": slug, "success": False, "message": "Temporary script not found"}
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("promote_temp_scripts error slug=%s", slug)
                results.append(
                    {"slug": slug, "success": False, "message": str(exc)}
                )
        return results

    @mcp.tool()
    def cleanup_old_temp_scripts() -> dict[str, object]:
        """Deletes temporary scripts older than three days."""
        try:
            cleaned_scripts = store.cleanup_old_temp_scripts()
            logger.info("cleanup_old_temp_scripts success count=%d", len(cleaned_scripts))
            return {
                "success": True,
                "cleaned_count": len(cleaned_scripts),
                "cleaned_scripts": cleaned_scripts,
                "message": f"Cleaned {len(cleaned_scripts)} old temporary scripts"
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("cleanup_old_temp_scripts error")
            return {
                "success": False,
                "cleaned_count": 0,
                "cleaned_scripts": [],
                "message": f"Failed to cleanup old temporary scripts: {exc}"
            }
