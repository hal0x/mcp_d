# metrics/registry.py
from prometheus_client import Counter, Gauge, Histogram

# Гистограмма латентности LLM по модели
LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM response latency",
    ["model", "phase"]  # phase: "gen"|"embed"|"rerank"
)

# Активные токены (в полёте) — удобно для следящих панелей
LLM_TOKENS_INFLIGHT = Gauge(
    "llm_tokens_inflight",
    "Active tokens in flight",
    ["direction"]  # "in"|"out"
)

# Назначения в A/B по промту/варианту
AB_ASSIGN = Counter(
    "ab_assignment_total",
    "AB assignments",
    ["experiment", "variant"]
)

# Решения координатора по политике
COORDINATOR_DECISION = Counter(
    "coordinator_decision_total",
    "Module decisions",
    ["policy", "module"]
)

# Ошибки по компонентам/типам
ERRORS = Counter(
    "errors_total",
    "Errors by component",
    ["component", "etype"]  # etype: Timeout|HTTP|Parse|Tool|Unknown
)
