from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Query(BaseModel):
    query_id: str
    text: str
    query_type: str = Field(..., pattern="^(factual|contextual|analytical|custom)$")
    chat_name: str
    entity: str | None = None
    timeframe: str | None = None
    generated_at: datetime


class SearchResult(BaseModel):
    result_id: str
    query_id: str
    content: str
    score: float
    chat: str | None = None
    metadata: dict = Field(default_factory=dict)


class RelevanceScore(BaseModel):
    query_id: str
    result_id: str | None = None
    overall_score: float
    individual_scores: list[float] = Field(default_factory=list)
    problems: dict = Field(default_factory=dict)
    explanation: str = ""
    recommendations: list[str] = Field(default_factory=list)


class Problem(BaseModel):
    problem_id: str
    problem_type: str
    severity: str
    description: str
    examples: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    title: str
    description: str
    suggestions: list[str] = Field(default_factory=list)
    priority: str = "medium"


class QualityMetrics(BaseModel):
    average_score: float = 0.0
    median_score: float = 0.0
    success_rate: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    details: dict = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    analysis_id: str
    chat_name: str
    timestamp: datetime
    queries: list[Query] = Field(default_factory=list)
    search_results: list[SearchResult] = Field(default_factory=list)
    relevance_scores: list[RelevanceScore] = Field(default_factory=list)
    metrics: QualityMetrics
    problems: list[Problem] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
