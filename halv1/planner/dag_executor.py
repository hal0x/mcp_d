from __future__ import annotations

import base64
import hashlib
import inspect
import json
import shutil
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Protocol,
    Tuple,
    TypedDict,
    cast,
)

import jmespath

from executor import ExecutionResult, ToolPolicy
from planner import ConditionExpr, Plan, PlanStep, lint_plan
from tools import ToolRegistry
from utils.artifacts import write_artifact_files


def cache_key(step: PlanStep, ctx: dict[str, Any]) -> str:
    """Return a stable hash for ``step`` within ``ctx``.

    The hash incorporates the tool, inputs, execution policy and the handler's
    version.  Additional execution details like container image and environment
    variables are included when present to ensure cached results are invalidated
    if the execution context changes. ``ctx`` is used solely to look up the
    handler's version via the tool registry.
    """

    registry = ctx.get("registry")
    version = "0"
    if isinstance(registry, ToolRegistry):
        handler = registry.try_get(step.tool)
        if handler is not None:
            version = getattr(handler, "__version__", "0")

    policy: Any
    if step.policy is None:
        policy = None
    elif isinstance(step.policy, Enum):
        policy = step.policy.value
    elif is_dataclass(step.policy):
        policy = asdict(step.policy)
    else:
        policy = step.policy

    payload: dict[str, Any] = {
        "tool": step.tool.value,
        "inputs": {"content": step.content, **step.inputs},
        "version": version,
        "policy": policy,
    }

    image = getattr(step, "image", None)
    if image is not None:
        payload["image"] = image

    env = getattr(step, "env", None) or getattr(step, "environment", None)
    if env is not None:
        payload["env"] = env
    data = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


class Artifact(TypedDict, total=False):
    stdout: str
    stderr: str
    files: dict[str, str]


class CacheBackend(Protocol):
    """Backend interface for artifact caching."""

    def load(self, key: str) -> Artifact | None:  # pragma: no cover - interface
        ...

    def save(
        self, key: str, artifact: Artifact
    ) -> None:  # pragma: no cover - interface
        ...

    def cleanup(self, ttl: timedelta) -> None:  # pragma: no cover - interface
        ...


@dataclass
class FileCache(CacheBackend):
    """File-based cache backend."""

    root: Path = Path("db/artifacts")

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> Artifact | None:
        artifact_dir = self.root / key
        meta = artifact_dir / "artifact.json"
        if not meta.exists():
            return None
        with meta.open() as f:
            artifact = cast(Artifact, json.load(f))
        files_dir = artifact_dir / "files"
        files: dict[str, str] = {}
        for name in artifact.get("files", []):
            file_path = files_dir / name
            if file_path.exists():
                files[name] = base64.b64encode(file_path.read_bytes()).decode()
        artifact["files"] = files
        return artifact

    def save(self, key: str, artifact: Artifact) -> None:
        artifact_dir = self.root / key
        files_dir = artifact_dir / "files"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        files_dir.mkdir(parents=True, exist_ok=True)
        files = artifact.get("files", {})
        names: list[str] = []
        for name, content in files.items():
            file_path = files_dir / name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(base64.b64decode(content))
            names.append(name)
        meta = {k: v for k, v in artifact.items() if k != "files"}
        meta["files"] = names
        with (artifact_dir / "artifact.json").open("w") as f:
            json.dump(meta, f)

    def cleanup(self, ttl: timedelta) -> None:
        cutoff = datetime.now() - ttl
        for path in self.root.glob("*"):
            if not path.is_dir():
                continue
            meta = path / "artifact.json"
            if meta.exists() and meta.stat().st_mtime < cutoff.timestamp():
                shutil.rmtree(path)


@dataclass
class ArtifactCache:
    """Cache interface delegating to a :class:`CacheBackend`."""

    backend: CacheBackend = field(default_factory=FileCache)

    def load(self, key: str) -> Artifact | None:
        return self.backend.load(key)

    def save(self, key: str, artifact: Artifact) -> None:
        self.backend.save(key, artifact)

    def cleanup(self, ttl: timedelta) -> None:
        self.backend.cleanup(ttl)


Condition = str | ConditionExpr | Callable[[dict[str, Any]], bool]


def _iter_output_strings(outputs: Artifact) -> Iterable[str]:
    if not isinstance(outputs, dict):
        return []
    parts: List[str] = []
    stdout = outputs.get("stdout")
    stderr = outputs.get("stderr")
    files: dict[str, str] = outputs.get("files") or {}
    if stdout:
        parts.append(str(stdout))
    if stderr:
        parts.append(str(stderr))
    parts.extend(map(str, files.values()))
    return parts


def check_conditions(conditions: Iterable[Condition], ctx: dict[str, Any]) -> bool:
    """Return ``True`` if all ``conditions`` are satisfied within ``ctx``."""

    # Build a view that overlays latest outputs onto ctx for evaluation
    view: dict[str, Any] = dict(ctx)
    outputs = ctx.get("outputs")
    if isinstance(outputs, dict):
        view.update(outputs)

    for cond in conditions:
        if isinstance(cond, str):
            if view.get(cond):
                continue
            if isinstance(outputs, dict) and any(
                cond in s for s in _iter_output_strings(cast(Artifact, outputs))
            ):
                ctx[cond] = True
                continue
            return False
        if isinstance(cond, ConditionExpr):
            engine = cond.engine.lower()
            if engine == "jmespath":
                try:
                    if jmespath.search(cond.expr, view):
                        continue
                except Exception:
                    return False
            else:  # pragma: no cover - unsupported engines
                raise ValueError(f"Unknown condition engine {cond.engine}")
            return False
        if not cond(view):
            return False
    return True


async def run_step(
    step: PlanStep, ctx: dict[str, Any], cache: ArtifactCache
) -> Artifact:
    """Execute a single ``step`` using caching."""

    registry = ctx.get("registry")
    if registry is None:
        raise ValueError("ctx must include a 'registry'")
    registry = cast(ToolRegistry, registry)

    if not check_conditions(step.preconditions, ctx):
        raise RuntimeError("Preconditions not met for step")

    key = cache_key(step, ctx)
    cached = cache.load(key)
    if cached is not None:
        write_artifact_files(
            {
                name: base64.b64decode(content)
                for name, content in cached.get("files", {}).items()
            }
        )
        ctx["outputs"] = cached
        if not check_conditions(step.postconditions, ctx):
            ctx.pop("outputs", None)
            raise RuntimeError("Postconditions failed for step")
        ctx.pop("outputs", None)
        return cached

    handler = registry.try_get(step.tool)
    if handler is None:
        raise RuntimeError(f"No handler for tool {step.tool.value}")

    policy_engine = ctx.get("policy_engine")
    if policy_engine is not None and not isinstance(step.policy, ToolPolicy):
        policy = policy_engine.get_policy(step.tool)
        step = replace(step, policy=policy)

    # Strict schema validation of step arguments against registered models
    model = registry.get_model(step.tool)
    if model is not None:
        payload: dict[str, Any]
        try:
            if step.tool.value == "code":
                payload = {"code": step.content}
            elif step.tool.value == "search":
                payload = {"query": step.content}
            elif step.tool.value == "file_io":
                first, _, rest = step.content.partition("\n")
                first = first.strip()
                if first.startswith("read"):
                    path = first[4:].strip()
                    payload = {"operation": "read", "path": path, "content": None}
                elif first.startswith("write"):
                    path = first[5:].strip()
                    payload = {"operation": "write", "path": path, "content": rest}
                else:
                    raise ValueError("Unknown FILE_IO operation")
            elif step.tool.value == "shell":
                payload = {"command": step.content}
            elif step.tool.value == "http":
                first, _, body = step.content.partition("\n")
                method, _, url = first.strip().partition(" ")
                payload = {"method": method.upper(), "url": url.strip(), "body": body}
            else:
                payload = {"content": step.content}
            # Pydantic v1/v2 compatibility
            if hasattr(model, "model_validate"):
                model.model_validate(payload)  # type: ignore[attr-defined]
            else:
                model.parse_obj(payload)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - validation failure
            raise ValueError(f"Invalid arguments for tool {step.tool.value}: {exc}")

    try:
        maybe = handler(step)
        step_results: Artifact | ExecutionResult | dict[str, Any] | None
        step_results = await maybe if inspect.isawaitable(maybe) else maybe
        if step_results is None:
            step_results = {}
        if isinstance(step_results, ExecutionResult):
            files_map = write_artifact_files(step_results.files)
            step_results = {
                "stdout": step_results.stdout,
                "stderr": step_results.stderr,
                "files": files_map,
            }
        elif isinstance(step_results, dict):
            files = step_results.get("files")
            if isinstance(files, list):
                new_files: dict[str, str] = {}
                for name in files:
                    file_path = Path(name)
                    if file_path.exists():
                        new_files[name] = base64.b64encode(
                            file_path.read_bytes()
                        ).decode()
                    else:
                        new_files[name] = ""
                step_results["files"] = new_files
            elif files is None:
                step_results["files"] = {}
        artifact = cast(Artifact, step_results)
        ctx["outputs"] = artifact
        if not check_conditions(step.postconditions, ctx):
            ctx.pop("outputs", None)
            raise RuntimeError("Postconditions failed for step")
        ctx.pop("outputs", None)
        cache.save(key, artifact)
        return artifact
    except Exception as exc:
        raise exc


async def run_plan(
    plan: Plan,
    ctx: dict[str, Any],
    cache: ArtifactCache,
    *,
    continue_on_error: bool = False,
    cleanup_ttl: timedelta | None = None,
) -> Tuple[List[Tuple[int, Artifact]], List[Tuple[int, Exception]]]:
    """Execute ``plan`` respecting dependencies and caching results.

    Parameters
    ----------
    plan:
        The plan to execute.
    ctx:
        Shared context passed to each tool handler.
    cache:
        Cache instance used for artifact storage.
    continue_on_error:
        If ``True``, execution continues for independent steps after a failure.
    cleanup_ttl:
        Remove artifacts older than this ``timedelta`` before execution when
        provided.
    """

    # Validate references before execution
    lint_plan(plan)
    order: List[int] = _topological_order(plan.steps)
    results: List[Tuple[int, Artifact]] = []
    errors: List[Tuple[int, Exception]] = []
    failed: set[int] = set()

    if cleanup_ttl is not None:
        cache.cleanup(cleanup_ttl)

    for idx in order:
        step = plan.steps[idx]
        if any(_coerce_dep(dep) in failed for dep in step.depends_on):
            err = RuntimeError("Dependency failed")
            errors.append((idx, err))
            failed.add(idx)
            if not continue_on_error:
                break
            continue
        try:
            step_results = await run_step(step, ctx, cache)
        except Exception as exc:
            errors.append((idx, exc))
            failed.add(idx)
            if not continue_on_error:
                break
            continue

        # Expose step outputs at the top-level ctx for subsequent conditions
        for k, v in step_results.items():
            if k not in {"stdout", "stderr", "files"}:
                ctx[k] = v
        ctx[step.id or str(idx)] = step_results
        results.append((idx, step_results))

    return results, errors


async def run_partial(
    plan: Plan,
    start_from_ids: Iterable[str | int],
    ctx: dict[str, Any],
    cache: ArtifactCache,
) -> Tuple[List[Tuple[int, Artifact]], List[Tuple[int, Exception]]]:
    """Execute a subset of ``plan`` starting from ``start_from_ids``.

    ``start_from_ids`` may contain step ``id`` values or numeric indices. All
    descendants of the specified steps are executed while assuming that their
    ancestors were previously run.
    """

    # Validate references before partial execution as well
    lint_plan(plan)
    id_to_idx = {step.id or str(i): i for i, step in enumerate(plan.steps)}
    start_indices: list[int] = []
    for sid in start_from_ids:
        if isinstance(sid, int):
            start_indices.append(sid)
        else:
            if sid not in id_to_idx:
                raise KeyError(f"Unknown step id {sid}")
            start_indices.append(id_to_idx[sid])

    graph: dict[int, set[int]] = {i: set() for i in range(len(plan.steps))}
    for i, step in enumerate(plan.steps):
        for dep in step.depends_on:
            graph[_coerce_dep(dep)].add(i)

    needed: set[int] = set()
    stack = list(start_indices)
    while stack:
        current = stack.pop()
        if current in needed:
            continue
        needed.add(current)
        stack.extend(graph[current])

    order = _topological_order(plan.steps)
    results: List[Tuple[int, Artifact]] = []
    errors: List[Tuple[int, Exception]] = []
    failed: set[int] = set()

    for idx in order:
        if idx not in needed:
            continue
        step = plan.steps[idx]
        if any(_coerce_dep(dep) in failed for dep in step.depends_on):
            err = RuntimeError("Dependency failed")
            errors.append((idx, err))
            failed.add(idx)
            break
        try:
            step_results = await run_step(step, ctx, cache)
        except Exception as exc:
            errors.append((idx, exc))
            failed.add(idx)
            break

        for k, v in step_results.items():
            if k not in {"stdout", "stderr", "files"}:
                ctx[k] = v
        ctx[step.id or str(idx)] = step_results
        results.append((idx, step_results))

    return results, errors


def _coerce_dep(dep: int | str) -> int:
    if isinstance(dep, int):
        return dep
    if isinstance(dep, str):
        s = dep.strip()
        if s.isdigit():
            return int(s)
    raise ValueError(f"Invalid dependency index: {dep}")


def _topological_order(steps: Iterable[PlanStep]) -> List[int]:
    indegree: dict[int, int] = {}
    graph: dict[int, set[int]] = {}
    steps_list = list(steps)
    # Normalize dependencies to integers
    norm_deps: dict[int, list[int]] = {}
    for i, step in enumerate(steps_list):
        deps = [_coerce_dep(d) for d in step.depends_on]
        norm_deps[i] = deps
        indegree[i] = len(deps)
        graph[i] = set()
    for i, _step in enumerate(steps_list):
        for dep in norm_deps[i]:
            graph[dep].add(i)
    queue: List[int] = [i for i, deg in indegree.items() if deg == 0]
    order: List[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    if len(order) != len(steps_list):
        # Вместо ошибки возвращаем fallback план
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Cycle detected in plan steps, using fallback plan")
        # Возвращаем простой линейный план
        return list(range(len(steps_list)))
    return order
