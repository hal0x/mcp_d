#!/usr/bin/env python3
"""
Основные модули для session_summarizer
"""

from .batch import summarize_batch_sessions
from .canonical import build_canonical_summary as build_canonical_summary_internal
from .generation import (
    apply_addon_metadata_to_claim,
    generate_actions,
    generate_claims,
    generate_discussion,
    generate_risks,
    generate_topics,
)
from .quality import refresh_quality
from .refinement import run_structural_pass

__all__ = [
    "summarize_batch_sessions",
    "build_canonical_summary_internal",
    "generate_topics",
    "generate_claims",
    "apply_addon_metadata_to_claim",
    "generate_discussion",
    "generate_actions",
    "generate_risks",
    "refresh_quality",
    "run_structural_pass",
]

