"""Multi-step execution mixin for AgentCore.

Contains logic for executing multi-step plans (optionally via a Docker-based
executor) and continuation heuristics based on `is_final` and expected outputs.
"""

from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from planner import Plan, PlanStep, Tool
from tools.registry import ArtifactDict
from utils.artifacts import write_artifact_files

# Try to import optional DAG executor utilities; provide fallbacks to keep tests light
try:  # pragma: no cover - import-time flexibility
    from planner.dag_executor import run_plan
except Exception:  # noqa: BLE001
    async def run_plan(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError("planner.dag_executor is not available in this environment")


logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

_COMPLETION_SYNONYM_GROUPS: tuple[tuple[str, set[str]], ...] = (
    (
        "complete",
        {
            "complete",
            "completed",
            "completes",
            "finis",
            "finish",
            "finished",
            "done",
            "accomplished",
        },
    ),
    (
        "success",
        {
            "success",
            "successful",
            "successfully",
        },
    ),
    (
        "create",
        {
            "create",
            "created",
            "creating",
            "make",
            "made",
            "write",
            "writes",
            "written",
            "generate",
            "generated",
            "produced",
            "produce",
            "build",
            "built",
        },
    ),
    (
        "read",
        {
            "read",
            "reads",
            "loaded",
            "load",
            "loads",
            "retrieve",
            "retrieved",
            "fetch",
            "fetched",
            "open",
            "opened",
        },
    ),
    (
        "sum",
        {
            "sum",
            "total",
            "aggregate",
        },
    ),
)

_SYNONYM_LOOKUP: dict[str, str] = {}
for canonical, synonyms in _COMPLETION_SYNONYM_GROUPS:
    for synonym in synonyms:
        _SYNONYM_LOOKUP[synonym] = canonical


class MultiStepPlanMixin:
    async def _execute_multi_step_plan(self, plan: Plan) -> tuple[list[tuple[int, ArtifactDict]], list[tuple[int, str]]]:
        """Execute multi-step plans using Docker when available.

        Falls back to the DAG ``run_plan`` executor when no ``docker_executor`` is
        attached to the instance.
        """
        trace_start = perf_counter()
        logger.info(
            "multi_step_executor_start",
            extra={"steps": len(plan.steps), "has_docker": hasattr(self, "docker_executor")},
        )

        ctx = {
            "registry": getattr(self, "registry", None),
            "policy_engine": getattr(self, "policy_engine", None),
        }
        if hasattr(self, "docker_executor"):
            has_non_convertible_tool = any(
                step.tool not in {Tool.CODE, Tool.FILE_IO} for step in plan.steps
            )
            if has_non_convertible_tool:
                executed_raw, errors = await run_plan(
                    plan,
                    ctx,
                    getattr(self, "cache", None),
                    cleanup_ttl=getattr(self, "artifact_ttl", None),
                )
                executed = cast(list[tuple[int, ArtifactDict]], executed_raw)
                duration = perf_counter() - trace_start
                logger.info(
                    "multi_step_executor_complete",
                    extra={
                        "steps": len(plan.steps),
                        "duration_ms": int(duration * 1000),
                        "errors": len(errors),
                        "mode": "dag_fallback",
                    },
                )
                return executed, [(idx, str(err)) for idx, err in errors]
            step_codes: list[str] = []
            for step in plan.steps:
                logger.info(
                    "processing_step",
                    extra={
                        "step_tool": step.tool.value,
                        "step_content_preview": step.content[:100] + "..." if len(step.content) > 100 else step.content,
                    },
                )
                if step.tool == Tool.CODE:
                    step_codes.append(step.content)
                elif step.tool == Tool.FILE_IO:
                    step_codes.append(self._convert_file_io_to_code(step))
                else:
                    step_codes.append(
                        f"# Tool: {step.tool.value}\n# Content: {step.content}\nprint('Tool executed')"
                    )
            result = self.docker_executor.execute_multi_step(step_codes)  # type: ignore[attr-defined]
            outputs = json.loads(result.stdout) if result.stdout else []
            stderr_list = json.loads(result.stderr) if result.stderr else []
            executed: list[tuple[int, ArtifactDict]] = []
            errors: list[tuple[int, str]] = []
            files_map = write_artifact_files(result.files) if getattr(result, "files", None) else {}
            files_per_step: list[dict[str, str]] = [{} for _ in plan.steps]
            if files_map and plan.steps:
                candidate_indices = [
                    idx for idx, step in enumerate(plan.steps) if getattr(step, "is_final", False)
                ]
                if not candidate_indices:
                    candidate_indices = [len(plan.steps) - 1]
                remaining_files = files_map.copy()
                for idx in sorted(set(candidate_indices), reverse=True):
                    if not remaining_files:
                        break
                    files_per_step[idx] = remaining_files.copy()
                    remaining_files.clear()
            for i, _ in enumerate(plan.steps):
                stderr_entry = stderr_list[i] if i < len(stderr_list) else ""
                if i < len(outputs):
                    artifact = {
                        "stdout": outputs[i],
                        "stderr": stderr_entry,
                        "files": files_per_step[i],
                    }
                    executed.append((i, artifact))
                if stderr_entry:
                    errors.append((i, stderr_entry))
                elif i >= len(outputs):
                    errors.append((i, "Step output not found"))
            for i in range(len(plan.steps), len(stderr_list)):
                stderr_entry = stderr_list[i]
                if stderr_entry:
                    errors.append((i, stderr_entry))
            duration = perf_counter() - trace_start
            logger.info(
                "multi_step_executor_complete",
                extra={
                    "steps": len(plan.steps),
                    "duration_ms": int(duration * 1000),
                    "errors": len(errors),
                    "mode": "docker",
                },
            )
            return executed, errors

        executed_raw, errors = await run_plan(
            plan, ctx, getattr(self, "cache", None), cleanup_ttl=getattr(self, "artifact_ttl", None)
        )
        executed = cast(list[tuple[int, ArtifactDict]], executed_raw)
        duration = perf_counter() - trace_start
        logger.info(
            "multi_step_executor_complete",
            extra={
                "steps": len(plan.steps),
                "duration_ms": int(duration * 1000),
                "errors": len(errors),
                "mode": "async_fallback",
            },
        )
        return executed, [(idx, str(err)) for idx, err in errors]

    def _needs_continuation(self, results: list[str], plan: Plan) -> bool:
        """Decide whether to continue based on final steps and results.

        - If no results or no steps: do not continue.
        - If final steps exist: continue until ALL finals are completed.
        - If no finals: allow a single continuation on first iteration.
        - Always check LLM assessment for task completion.
        """
        trace_start = perf_counter()
        logger.debug(
            "needs_continuation_start",
            extra={"results_count": len(results), "plan_steps": len(plan.steps)},
        )
        if not results or not plan.steps:
            duration = perf_counter() - trace_start
            logger.debug(
                "needs_continuation_finish",
                extra={
                    "decision": False,
                    "duration_ms": int(duration * 1000),
                    "reason": "no_results_or_steps",
                },
            )
            return False

        # Проверяем мнение LLM о завершенности задачи
        llm_assessment = self._check_llm_completion_assessment(results)
        if llm_assessment is not None:
            logger.info(
                "continuation_decision",
                extra={
                    "continuation_needed": not llm_assessment,
                    "reason": "llm_assessment",
                    "llm_says_complete": llm_assessment,
                    "results": results,
                },
            )
            duration = perf_counter() - trace_start
            logger.debug(
                "needs_continuation_finish",
                extra={
                    "decision": not llm_assessment,
                    "duration_ms": int(duration * 1000),
                    "reason": "llm_assessment",
                },
            )
            return not llm_assessment

        final_steps = [s for s in plan.steps if getattr(s, "is_final", False)]

        if final_steps:
            incomplete = [s.id for s in final_steps if not self._is_step_completed(s, results)]
            need = len(incomplete) > 0
            logger.info(
                "continuation_decision",
                extra={
                    "continuation_needed": need,
                    "reason": ("some_final_incomplete" if need else "all_finals_completed"),
                    "incomplete_final_steps": incomplete,
                    "final_steps_total": len(final_steps),
                },
            )
            duration = perf_counter() - trace_start
            logger.debug(
                "needs_continuation_finish",
                extra={
                    "decision": need,
                    "duration_ms": int(duration * 1000),
                    "reason": (
                        "some_final_incomplete" if need else "all_finals_completed"
                    ),
                },
            )
            return need

        need = bool(results) and getattr(self, "iterations", 0) == 0
        logger.info(
            "continuation_decision",
            extra={
                "continuation_needed": need,
                "reason": "no_final_steps_in_plan_once",
                "plan_steps": len(plan.steps),
            },
        )
        duration = perf_counter() - trace_start
        logger.debug(
            "needs_continuation_finish",
            extra={
                "decision": need,
                "duration_ms": int(duration * 1000),
                "reason": "no_final_steps_in_plan_once",
            },
        )
        return need

    def _check_llm_completion_assessment(self, results: list[str]) -> bool | None:
        """Check LLM assessment of task completion.
        
        Returns:
            True if LLM says task is complete
            False if LLM says task is incomplete  
            None if assessment failed or not available
        """
        try:
            # Получаем LLM клиент из code_generator
            cg_client = getattr(self, "code_generator", None)
            if cg_client is None:
                return None
                
            client = getattr(cg_client, "client", None)
            if client is None:
                return None
                
            # Формируем промпт для проверки завершенности
            goal = getattr(self, "goal", "")
            if not goal:
                return None
                
            results_text = "\n".join(str(r) for r in results if r and str(r).strip())
            if not results_text:
                return None
                
            prompt = (
                "Determine if the objective is complete given current results.\n"
                f"Objective: {goal}\n"
                f"Results:\n{results_text}\n\n"
                "Respond with 'COMPLETE' if the objective is fully achieved, "
                "or 'INCOMPLETE' if more work is needed."
            )
            
            response = client.generate(prompt)
            response_text, new_history = self._extract_llm_response_parts(response)

            if new_history is not None:
                try:
                    setattr(cg_client, "conversation_history", new_history)
                except Exception:  # noqa: BLE001 - best effort persistence only
                    logger.debug("failed_to_store_llm_history", exc_info=True)

            if not response_text:
                return None

            response_lower = response_text.lower().strip()
            if "complete" in response_lower and "incomplete" not in response_lower:
                return True
            elif "incomplete" in response_lower:
                return False
            else:
                # Если ответ неясный, считаем задачу не завершенной для безопасности
                return False
                
        except Exception as exc:
            logger.warning(
                "llm_completion_assessment_failed",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            return None

    @staticmethod
    def _extract_llm_response_parts(response: Any) -> tuple[str | None, Any | None]:
        """Extract text and optional history from an LLM response container."""

        if isinstance(response, str):
            return response, None

        history: Any | None = None
        text: str | None = None

        if isinstance(response, Mapping):
            for key in ("text", "response", "content", "completion", "message"):
                if key in response and response[key] is not None:
                    text = str(response[key])
                    break
            history = response.get("history") or response.get("context")
        elif isinstance(response, Sequence) and not isinstance(
            response, (str, bytes, bytearray)
        ):
            if response:
                first = response[0]
                if first is not None:
                    text = str(first)
            if len(response) > 1:
                history = response[1]
        elif response is not None:
            text = str(response)

        text = text if text and text.strip() else None
        return text, history

    def _is_step_completed(self, step: PlanStep, results: list[str]) -> bool:
        """Check completion of a step using expected_output heuristic."""
        exp = getattr(step, "expected_output", "")
        if not isinstance(exp, str) or not exp.strip():
            return True

        expected_tokens = self._normalize_expected_output(exp)
        if not expected_tokens:
            return True

        normalized_results = [self._normalize_expected_output(res) for res in results if isinstance(res, str)]
        for tokens in normalized_results:
            if expected_tokens.issubset(tokens):
                return True

        combined_tokens: set[str] = set()
        for tokens in normalized_results:
            combined_tokens.update(tokens)
        return expected_tokens.issubset(combined_tokens)

    @staticmethod
    def _normalize_expected_output(text: str) -> set[str]:
        """Normalize text for completion comparison.

        The normalization removes punctuation, lowercases text, maps common synonyms
        to a shared canonical form, and highlights numeric tokens so that small
        formatting differences do not prevent a completion match.
        """

        if not isinstance(text, str):
            return set()

        lowered = text.casefold()
        numbers = set(re.findall(r"\d+(?:\.\d+)?", lowered))
        tokens = re.findall(r"[a-z0-9]+", lowered)

        normalized_tokens: list[str] = []
        for token in tokens:
            if token in numbers:
                normalized_tokens.append(token)
                continue
            canonical = _SYNONYM_LOOKUP.get(token, token)
            if canonical not in _STOPWORDS:
                normalized_tokens.append(canonical)

        if not normalized_tokens and tokens:
            normalized_tokens = [_SYNONYM_LOOKUP.get(token, token) for token in tokens]

        normalized_tokens.extend(numbers)
        return set(normalized_tokens)
