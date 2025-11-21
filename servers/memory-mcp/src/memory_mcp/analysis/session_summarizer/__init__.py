#!/usr/bin/env python3
"""
Пакет для структурной саммаризации сессий
"""

from .api import summarize_chat_sessions
from .summarizer import SessionSummarizer

__all__ = ["SessionSummarizer", "summarize_chat_sessions"]

