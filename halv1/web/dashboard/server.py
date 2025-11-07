# web/dashboard/server.py
import logging
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Импортируем метрики для их регистрации
from metrics import LLM_LATENCY, LLM_TOKENS_INFLIGHT, AB_ASSIGN, COORDINATOR_DECISION, ERRORS
from bot.telegram_bot import BroadcastExecutor
from storage import (
    init_trading_storage,
    close_trading_storage,
    insert_trading_alert,
    insert_trading_feedback,
    fetch_trading_alert,
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

static_dir = BASE_DIR / "static"


from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PositiveInt


logger = logging.getLogger(__name__)


_broadcast_executor: BroadcastExecutor | None = None


class TradingAlertPayload(BaseModel):
    id: Optional[int] = Field(default=None, description="Internal alert id")
    external_id: Optional[str] = Field(default=None, description="External source id")
    symbol: str
    timeframe: str
    direction: str
    entry: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr: Optional[float] = None
    leverage: float | None = None
    confidence: PositiveInt | None = None
    reasons: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI()

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def on_startup():
    global _broadcast_executor
    try:
        from bot.telegram_bot import create_broadcast_executor
    except ImportError:  # pragma: no cover
        logger.warning("Broadcast executor not available")
        return
    await init_trading_storage()
    _broadcast_executor = await create_broadcast_executor()


@app.on_event("shutdown")
async def on_shutdown():
    await close_trading_storage()


@app.post("/api/trading-feedback")
async def receive_trading_feedback(feedback: dict):
    """Receive trading signal feedback from Telegram bot."""
    logger.info(
        "Received trading feedback",
        extra={
            "topic": "trading_feedback",
            "payload": feedback,
        },
    )
    signal_id = feedback.get("signal_id")
    action = feedback.get("action")
    if signal_id is None or action is None:
        raise HTTPException(status_code=400, detail="signal_id and action are required")
    try:
        await insert_trading_feedback(int(signal_id), str(action))
    except Exception as exc:
        logger.exception("feedback_persist_failed", signal_id=signal_id, error=str(exc))
    return {"status": "ok"}


@app.post("/api/trading-alert")
async def receive_trading_alert(alert: TradingAlertPayload):
    payload = alert.model_dump()
    logger.info(
        "Received trading alert",
        extra={
            "topic": "trading_alert",
            "payload": payload,
        },
    )
    stored = None
    try:
        stored = await insert_trading_alert(payload)
    except Exception as exc:
        logger.exception("alert_persist_failed", error=str(exc))
    alert_id = None
    if stored:
        alert_id = stored.get("id")
        payload["id"] = alert_id
        payload.setdefault("created_at", stored.get("created_at"))
    if alert.external_id and alert_id is None:
        payload.setdefault("id", alert.external_id)
    if _broadcast_executor is None:
        raise HTTPException(status_code=503, detail="Notification executor unavailable")
    await _broadcast_executor.broadcast_trade_signal(payload)
    return {"status": "ok", "alert_id": alert_id}



@app.get("/api/trading-alert/{alert_id}")
async def get_trading_alert(alert_id: int):
    data = await fetch_trading_alert(alert_id)
    if not data:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"payload": data}

def _collect_metric_metadata(metric):
    """Получить информацию о метрике для отображения на панели."""

    metric_type = metric._type  # type: ignore[attr-defined]
    metric_doc = getattr(metric, "_documentation", "")
    metric_name = getattr(metric, "_name", "")
    return {
        "name": metric_name,
        "documentation": metric_doc,
        "type": metric_type,
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    metrics_summary = [
        _collect_metric_metadata(LLM_LATENCY),
        _collect_metric_metadata(LLM_TOKENS_INFLIGHT),
        _collect_metric_metadata(AB_ASSIGN),
        _collect_metric_metadata(COORDINATOR_DECISION),
        _collect_metric_metadata(ERRORS),
    ]

    documentation_links = [
        {"label": "Healthcheck", "href": "/health"},
        {"label": "Prometheus metrics", "href": "/metrics"},
        {"label": "Interactive API docs", "href": "/docs"},
        {"label": "OpenAPI schema", "href": "/openapi.json"},
    ]

    context = {
        "request": request,
        "health_status": "OK",
        "metrics": metrics_summary,
        "documentation_links": documentation_links,
    }

    return templates.TemplateResponse(request, "dashboard.html", context)
