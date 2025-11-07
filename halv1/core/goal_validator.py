from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Tuple, cast

from tools.registry import ArtifactDict

Rule = Callable[[ArtifactDict], Tuple[bool, str]]


@dataclass
class ValidationRule:
    """Описание правила валидации цели."""

    name: str
    validator: Rule
    priority: int = 0
    description: str = ""
    keywords: list[str] = field(default_factory=list)

    def matches(self, goal: str) -> bool:
        if not self.keywords:
            return False
        goal_lower = goal.lower()
        return any(keyword in goal_lower for keyword in self.keywords)


class GoalValidator:
    """Validate goal completion using execution artifacts.

    Объединяет исходную реализацию и расширенные возможности v2.
    """

    def __init__(
        self,
        rules: Dict[str, ValidationRule | Rule] | None = None,
    ) -> None:
        self._default_generic_rule = ValidationRule(
            name="GenericTask",
            validator=self._validate_generic_task,
            priority=10,
            description="Универсальная валидация задач",
        )
        self.rules: Dict[str, ValidationRule] = self._initialise_rules(rules)
        self.rules.setdefault("GenericTask", self._default_generic_rule)

    def _initialise_rules(
        self, rules: Dict[str, ValidationRule | Rule] | None
    ) -> Dict[str, ValidationRule]:
        if rules is None:
            return self._create_default_rules()

        return {
            name: self._ensure_validation_rule(name, rule)
            for name, rule in rules.items()
        }

    def _ensure_validation_rule(
        self, name: str, rule: ValidationRule | Rule
    ) -> ValidationRule:
        if isinstance(rule, ValidationRule):
            return rule
        return ValidationRule(name=name, validator=rule)

    def _create_default_rules(self) -> Dict[str, ValidationRule]:
        return {
            "SumNumbers": ValidationRule(
                name="SumNumbers",
                validator=self._validate_sum_numbers,
                priority=100,
                description="Валидация суммирования чисел",
                keywords=["сумма", "sum", "сложить", "add"],
            ),
            "DailySummary": ValidationRule(
                name="DailySummary",
                validator=self._validate_daily_summary,
                priority=90,
                description="Валидация ежедневной сводки",
                keywords=["саммари", "summary", "сводка", "итог", "ежедневная"],
            ),
            "DateTask": ValidationRule(
                name="DateTask",
                validator=self._validate_date_task,
                priority=80,
                description="Валидация задач с датами",
                keywords=["дата", "date", "время", "time", "сегодня", "today"],
            ),
            "CodeExecution": ValidationRule(
                name="CodeExecution",
                validator=self._validate_code_execution,
                priority=70,
                description="Валидация выполнения кода",
                keywords=["код", "code", "программа", "скрипт", "выполнить"],
            ),
            "FileOperation": ValidationRule(
                name="FileOperation",
                validator=self._validate_file_operation,
                priority=60,
                description="Валидация файловых операций",
                keywords=["файл", "file", "создать", "create", "записать", "write"],
            ),
            "GenericTask": self._default_generic_rule,
        }

    def validate(self, goal: str, artifacts: ArtifactDict | None) -> Tuple[bool, str]:
        """Validate goal execution outcome."""
        if artifacts is None:
            return False, "missing artifacts"

        rule = self._find_best_rule(goal)
        try:
            return rule.validator(artifacts)
        except Exception as exc:  # pragma: no cover - defensive guard
            return False, f"validation error: {exc}"

    def _find_best_rule(self, goal: str) -> ValidationRule:
        if goal in self.rules:
            return self.rules[goal]

        matches = [rule for rule in self.rules.values() if rule.matches(goal)]
        if matches:
            return max(matches, key=lambda r: r.priority)

        return self.rules.get("GenericTask", self._default_generic_rule)

    def add_rule(self, rule: ValidationRule) -> None:
        self.rules[rule.name] = rule

    def remove_rule(self, name: str) -> bool:
        if name in self.rules:
            del self.rules[name]
            return True
        return False

    def get_rule_info(self, name: str) -> Dict[str, Any] | None:
        rule = self.rules.get(name)
        if rule is None:
            return None
        return {
            "name": rule.name,
            "priority": rule.priority,
            "description": rule.description,
            "keywords": rule.keywords,
        }

    def list_rules(self) -> list[Dict[str, Any]]:
        infos: list[Dict[str, Any]] = []
        for name in self.rules:
            info = self.get_rule_info(name)
            if info is not None:
                infos.append(info)
        return infos

    # --- built-in validators -------------------------------------------------
    def _validate_sum_numbers(self, artifacts: ArtifactDict) -> Tuple[bool, str]:
        data = cast("dict[str, Any] | None", artifacts.get("sum"))
        if data is None:
            return False, "missing sum data"

        try:
            total = data["total"]
            inputs = data["inputs"]
            provided_hash = data["hash"]
        except KeyError as exc:
            return False, f"missing {exc.args[0]}"

        if not isinstance(inputs, list) or any(not isinstance(x, str) for x in inputs):
            return False, "inputs must be list of strings"

        payload = "".join(inputs).encode()
        expected = hashlib.sha256(payload).hexdigest()

        if provided_hash != expected:
            return False, "hash mismatch"

        if total != sum(int(x) for x in inputs):
            return False, "incorrect total"

        return True, "sum validation passed"

    def _validate_daily_summary(self, artifacts: ArtifactDict) -> Tuple[bool, str]:
        stdout = artifacts.get("stdout", "")
        if not stdout:
            return False, "missing stdout"

        summary_keywords = ["сводка", "summary", "итог", "результат", "вывод"]
        if not any(keyword in stdout.lower() for keyword in summary_keywords):
            return False, "no summary keywords found"

        if len(stdout.strip()) < 50:
            return False, "summary too short"

        return True, "daily summary validated"

    def _validate_date_task(self, artifacts: ArtifactDict) -> Tuple[bool, str]:
        stdout = artifacts.get("stdout", "")
        if not stdout:
            return False, "missing stdout"

        date_patterns = [
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{2}\.\d{2}\.\d{4})",
        ]

        found_dates: list[str] = []
        for pattern in date_patterns:
            found_dates.extend(re.findall(pattern, stdout))

        if not found_dates:
            return False, "no dates found in output"

        try:
            now = datetime.now(timezone.utc)
            for date_str in found_dates:
                if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                elif re.match(r"\d{2}\.\d{2}\.\d{4}", date_str):
                    parsed_date = datetime.strptime(date_str, "%d.%m.%Y")
                else:  # pragma: no cover - формат не совпал с regex
                    continue
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                if abs((now - parsed_date).total_seconds()) > 86400 * 2:
                    return False, f"date {date_str} too far from current date"
        except Exception as exc:  # pragma: no cover - защитный блок
            return False, f"date parsing error: {exc}"

        return True, "date task validated"

    def _validate_code_execution(self, artifacts: ArtifactDict) -> Tuple[bool, str]:
        stdout = artifacts.get("stdout", "")
        stderr = artifacts.get("stderr", "")

        if stderr and any(
            keyword in stderr.lower() for keyword in ["error", "exception", "failed", "traceback"]
        ):
            return False, f"code execution error: {stderr[:200]}"

        if not stdout:
            return False, "no output produced"

        if "error" in stdout.lower() or "exception" in stdout.lower():
            return False, "code execution error in output"

        return True, "code executed successfully"

    def _validate_file_operation(self, artifacts: ArtifactDict) -> Tuple[bool, str]:
        files = artifacts.get("files", {})
        if not files:
            return False, "no files created"

        for filename, content in files.items():
            if not content:
                return False, f"file {filename} is empty"

        return True, "files created successfully"

    def _validate_generic_task(self, artifacts: ArtifactDict) -> Tuple[bool, str]:
        stdout = artifacts.get("stdout", "")
        stderr = artifacts.get("stderr", "")

        if stderr and any(
            keyword in stderr.lower() for keyword in ["error", "exception", "failed", "traceback"]
        ):
            return False, f"execution error: {stderr[:200]}"

        if not stdout:
            return False, "no output produced"

        return True, "generic task completed"


def _validate_sum_numbers(artifacts: ArtifactDict) -> Tuple[bool, str]:
    """Backwards-compatible helper."""
    return GoalValidator()._validate_sum_numbers(artifacts)


def _validate_daily_summary(artifacts: ArtifactDict) -> Tuple[bool, str]:
    """Backwards-compatible helper."""
    return GoalValidator()._validate_daily_summary(artifacts)


def _validate_date_task(artifacts: ArtifactDict) -> Tuple[bool, str]:
    """Backwards-compatible helper."""
    return GoalValidator()._validate_date_task(artifacts)


def _validate_generic_task(goal: str, artifacts: ArtifactDict) -> Tuple[bool, str]:
    """Backwards-compatible helper preserving старый API."""
    return GoalValidator()._find_best_rule(goal).validator(artifacts)
