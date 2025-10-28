"""Alerts service for managing alert rules and notifications."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, AsyncContextManager, Callable, Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Alert, AlertRule
from ..models.orm import ActiveAlertORM, AlertRuleORM

SessionProvider = Callable[[], AsyncContextManager[AsyncSession]]


class AlertsService:
    """Service for managing alert rules and notifications."""

    def __init__(self, session_provider: SessionProvider):
        self._session_provider = session_provider

    async def create_alert_rule(self, rule: AlertRule) -> None:
        """Create or update an alert rule."""
        async with self._session_provider() as session:
            existing = await session.scalar(
                select(AlertRuleORM).where(AlertRuleORM.rule_id == rule.id)
            )
            if existing:
                existing.name = rule.name
                existing.condition = rule.condition
                existing.severity = rule.severity
                existing.enabled = rule.enabled
                existing.cooldown_minutes = rule.cooldown_minutes
                existing.actions = list(rule.actions)
            else:
                session.add(
                    AlertRuleORM(
                        rule_id=rule.id,
                        name=rule.name,
                        condition=rule.condition,
                        severity=rule.severity,
                        enabled=rule.enabled,
                        cooldown_minutes=rule.cooldown_minutes,
                        actions=list(rule.actions),
                    )
                )
            await session.commit()

    async def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        async with self._session_provider() as session:
            result = await session.execute(select(AlertRuleORM).order_by(AlertRuleORM.name.asc()))
            records = result.scalars().all()
        return [self._rule_from_record(record) for record in records]

    async def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get specific alert rule."""
        async with self._session_provider() as session:
            record = await session.scalar(select(AlertRuleORM).where(AlertRuleORM.rule_id == rule_id))
        return self._rule_from_record(record) if record else None

    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        async with self._session_provider() as session:
            result = await session.execute(delete(AlertRuleORM).where(AlertRuleORM.rule_id == rule_id))
            await session.execute(delete(ActiveAlertORM).where(ActiveAlertORM.rule_id == rule_id))
            await session.commit()
        return result.rowcount > 0

    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        async with self._session_provider() as session:
            result = await session.execute(select(ActiveAlertORM).order_by(ActiveAlertORM.triggered_at.desc()))
            records = result.scalars().all()
        return [self._alert_from_record(record) for record in records]

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        async with self._session_provider() as session:
            record = await session.scalar(select(ActiveAlertORM).where(ActiveAlertORM.alert_id == alert_id))
            if not record:
                return False
            record.acknowledged = True
            record.acknowledged_at = datetime.utcnow()
            record.acknowledged_by = acknowledged_by
            await session.commit()
            return True

    async def evaluate_alerts(self, metrics_data: Dict[str, Any]) -> List[Alert]:
        """Evaluate alert rules against current metrics."""
        async with self._session_provider() as session:
            rules_result = await session.execute(
                select(AlertRuleORM).where(AlertRuleORM.enabled.is_(True))
            )
            rules = rules_result.scalars().all()

            new_alert_records: List[ActiveAlertORM] = []
            now = datetime.utcnow()

            for rule in rules:
                if not self._evaluate_condition(rule.condition, metrics_data):
                    continue

                in_cooldown = await self._is_in_cooldown(session, rule, now)
                if in_cooldown:
                    continue

                alert_record = ActiveAlertORM(
                    alert_id=f"{rule.rule_id}-{int(now.timestamp())}",
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    message=f"Alert triggered: {rule.name}",
                    triggered_at=now,
                    acknowledged=False,
                )
                session.add(alert_record)
                new_alert_records.append(alert_record)

            if new_alert_records:
                await session.commit()

        return [self._alert_from_record(record) for record in new_alert_records]

    async def _is_in_cooldown(self, session: AsyncSession, rule: AlertRuleORM, now: datetime) -> bool:
        cooldown_time = now - timedelta(minutes=rule.cooldown_minutes)
        result = await session.execute(
            select(ActiveAlertORM).where(
                ActiveAlertORM.rule_id == rule.rule_id,
                ActiveAlertORM.triggered_at >= cooldown_time,
            )
        )
        return result.first() is not None

    def _evaluate_condition(self, condition: str, metrics_data: Dict[str, Any]) -> bool:
        """Evaluate alert condition (simplified implementation)."""
        try:
            expr = condition
            for key, value in metrics_data.items():
                expr = expr.replace(key, str(value))
            return bool(eval(expr))
        except Exception:  # pragma: no cover - defensive
            return False

    def _rule_from_record(self, record: AlertRuleORM) -> AlertRule:
        return AlertRule(
            id=record.rule_id,
            name=record.name,
            condition=record.condition,
            severity=record.severity,
            enabled=record.enabled,
            cooldown_minutes=record.cooldown_minutes,
            actions=list(record.actions or []),
        )

    def _alert_from_record(self, record: ActiveAlertORM) -> Alert:
        return Alert(
            id=record.alert_id,
            rule_id=record.rule_id,
            severity=record.severity,
            message=record.message,
            triggered_at=record.triggered_at,
            acknowledged=record.acknowledged,
            acknowledged_at=record.acknowledged_at,
            acknowledged_by=record.acknowledged_by,
        )
