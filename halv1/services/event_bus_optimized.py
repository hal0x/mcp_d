"""Shim module re-exporting optimized event bus from unified implementation."""

from __future__ import annotations

from .event_bus_impl import (
    AsyncEventBusOptimized,
    EventPriority,
    PrioritizedEvent,
)

from .event_bus_common import LRUCache

