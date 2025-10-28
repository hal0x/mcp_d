"""Data models for Policy MCP."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Return timezone-aware current time."""
    return datetime.now(tz=timezone.utc)


class DecisionProfile(BaseModel):
    """Decision profile for policy enforcement."""

    id: str
    name: str
    description: str
    active: bool = True
    version: str = "1.0.0"
    rules: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ProfileVersion(BaseModel):
    """Historical snapshot of a profile."""

    profile_id: str
    version: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class PolicyRule(BaseModel):
    """A single policy rule."""

    id: str
    profile_id: str
    name: str
    condition: str
    action: str
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecisionRequest(BaseModel):
    """Request for a decision under a policy."""

    profile_id: str
    context: Dict[str, Any]
    actor: Optional[str] = None
    correlation_id: Optional[str] = None


class DecisionResponse(BaseModel):
    """Response from policy decision."""

    profile_id: str
    decision: str  # allow, deny, review
    reason: str
    rules_applied: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
