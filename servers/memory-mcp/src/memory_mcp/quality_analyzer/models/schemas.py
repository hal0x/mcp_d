from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class QuerySchema(BaseModel):
    id: str
    query: str
    type: str = Field(pattern="^(factual|contextual|analytical|custom)$")
    chat_name: str
    entity: str | None = None
    timeframe: str | None = None
    generated_at: datetime


class SearchResultSchema(BaseModel):
    id: str
    text: str
    score: float
    query_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class RelevanceAnalysisSchema(BaseModel):
    overall_score: float
    individual_scores: list[float] = Field(default_factory=list)
    problems: dict = Field(default_factory=dict)
    explanation: str = ""
    recommendations: list[str] = Field(default_factory=list)


class BatchResultSchema(BaseModel):
    query: QuerySchema
    search_results: list[SearchResultSchema] = Field(default_factory=list)
    relevance_analysis: RelevanceAnalysisSchema
    timestamp: datetime
    error: str | None = None
