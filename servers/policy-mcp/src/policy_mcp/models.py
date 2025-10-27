"""Data models for Policy MCP."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionProfile(BaseModel):
    """Decision profile for policy enforcement."""
    id: str
    name: str
    description: str
    active: bool = True
    version: str = "1.0.0"
    rules: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


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

