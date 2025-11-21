"""Analysis functionality for Telegram Dump Manager."""

from .aggregation import (
    BatchSessionProcessor,
    LargeContextProcessor,
    RollingWindowAggregator,
    SmartRollingAggregator,
)
from .context import (
    ContextAwareProcessor,
    ContextManager,
    IncrementalContextManager,
)
from .entities import EntityDictionary, EntityExtractor
from .rendering import (
    Insight,
    InsightGraphResult,
    MarkdownRenderer,
    SummaryInsightAnalyzer,
    build_insight_graph,
)
from .segmentation import (
    AdaptiveMessageGrouper,
    DayGroupingSegmenter,
    SemanticRegrouper,
    SessionClusterer,
    SessionSegmenter,
)
from .summarization.session import SessionSummarizer, summarize_chat_sessions
from .summarization import (
    ClusterSummarizer,
    IterativeRefiner,
    LangChainSummarizationChain,
    QualityEvaluator,
)
from .utils import InstructionManager, MessageFilter, TimeProcessor

__all__ = [
    # Segmentation
    "SessionSegmenter",
    "DayGroupingSegmenter",
    "SessionClusterer",
    "AdaptiveMessageGrouper",
    "SemanticRegrouper",
    # Session Summarizer
    "SessionSummarizer",
    "summarize_chat_sessions",
    # Entities
    "EntityExtractor",
    "EntityDictionary",
    # Rendering
    "MarkdownRenderer",
    "SummaryInsightAnalyzer",
    "InsightGraphResult",
    "Insight",
    "build_insight_graph",
    # Aggregation
    "LargeContextProcessor",
    "ContextAwareProcessor",
    "BatchSessionProcessor",
    "RollingWindowAggregator",
    "SmartRollingAggregator",
    # Context
    "ContextManager",
    "IncrementalContextManager",
    # Summarization
    "ClusterSummarizer",
    "LangChainSummarizationChain",
    "QualityEvaluator",
    "IterativeRefiner",
    # Utils
    "MessageFilter",
    "InstructionManager",
    "TimeProcessor",
]
