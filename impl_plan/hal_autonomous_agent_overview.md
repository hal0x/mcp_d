# 📚 Обзор идей: «Доведение HAL до полноценного автономного AI‑агента»

> Этот обзор консолидирует решения и спецификации, разработанные **после** `HAL_Implementation_Instructions.md`. Документ служит картой реализации: что строим, как связываем модули и какие артефакты/контракты считаем «стандартом».

---

## 1) Executive Summary

**Текущий прогресс:** ~45-50% (28 октября 2025)

- **✅ Завершено**: `supervisor-mcp` базовые API (~50%), `learning-mcp` offline training (~35%), `policy-mcp` Decision Profiles (~25%), `orchestrator-mcp` базовая структура (~20%)
- **🟡 В процессе**: Интеграция HALv1 ↔ Supervisor, завершение Ingest/Query API, стабилизация EventBus
- **❌ Планируется**: Decision MCP, World‑Model MCP, полная автономность

**Фокус приоритета**: `supervisor-mcp` (реестр/метрики/факты/алерты) + `learning-mcp` (обучение на агрегатах) как «кровеносная система» автономности.
- **Упрощённая оркестрация**: «тонкий Orchestrator» (линейные планы) вместо сложных графов; всё остальное — приращениями.
- **SSOT памяти**: единый `memory-mcp` + L0‑кэш в HALv1; «факты» и «кристаллы» знаний как ключевые артефакты.
- **World‑Model MCP**: lenses (точки взгляда), personas (модели поведения), intent→plan, prospective memory, причинность и уверенность.
- **Интеграционный слой**: единый RPC/схемы, circuit breakers, health‑агрегатор; стандартизованные DTO.
- **Безопасность по умолчанию**: Policy>Decision>Signal, dry‑run+confirm, канареечные изменения (A/B ≤1% бюджета).

---

## 2) Архитектурная карта модулей

### Базовые компоненты
- **HALv1 Core** — планирование/исполнение, Telegram‑UX, L0‑кэш.
- **supervisor‑mcp** — реестр MCP (discovery), health/status, сбор метрик и `Fact:*`, агрегаты 7/30d, алерты, рекомендации действий.
- **learning‑mcp** — оффлайн обучение на агрегатах supervisor, предложение `DecisionProfile` (веса/пороги) + A/B рамка + guardrails.
- **memory‑mcp (SSOT)** — хранение фактов (`Fact:Signal|Decision|Trade|Outcome|PolicyHit`), артефактов (PMI, Post‑mortems, Skills, Concepts).
- **policy/config** — единая точка политик/флагов/профилей решений, ETag и версии.
- **(тонкий) orchestrator‑mcp** — линейные планы: analyze → decide → dry‑run → confirm → execute → log.

### Расширяющие компоненты
- **World‑Model MCP** — lenses (market_regime/liquidity/tech), personas (conservative/balanced/opportunistic), intent→plan, causality+confidence, встроенная Prospective Memory.
- **Prospective‑Memory (встроено в World‑Model)** — планирование воспоминаний (PMI): триггеры «время/цена/режим», ресёрчи, алерты, жизненный цикл short‑term→dormant→archived с деградацией/кристаллизацией в навыки.

### Интеграция и устойчивость
- **MCP Integration Layer** — единые клиенты, JSON Schema Registry, нормализация `Ok/Err/Meta`, retry/backoff, circuit‑breaker, health‑агрегатор, классы надёжности `critical|standard|optional`.
- **HALv1 EventBus Fixes** — таймауты, cancel tokens, lock‑free очереди, backpressure, воркеры, watchdog, идемпотентность шагов.

---

## 3) Канонические артефакты и DTO

### Fact (событие‑факт)
```json
{
  "ts": "2025-10-19T12:05:00Z",
  "kind": "Fact:Outcome|Fact:Trade|Fact:Decision|Fact:Signal|Fact:PolicyHit|Fact:PMI*",
  "actor": "halv1",
  "correlation_id": "tsk_abcd1234",
  "payload": { /* детали */ }
}
```

### Metric
```json
{"ts":"2025-10-19T12:00:00Z","name":"rpc_latency_p95","value":173,
 "tags":{"svc":"binance-mcp","op":"create_order"}}
```

### DecisionProfile
```json
{
  "profile_id": "default_v1.3",
  "weights": {"squeeze": 0.38, "trend_up": 0.24, "funding_neg": 0.16, "low_corr": 0.22},
  "thresholds": {"score_trade": 0.67, "score_full": 0.80},
  "caps": {"max_risk_pct": 0.8, "max_change_per_epoch": 0.05},
  "valid_until": "2025-11-01T00:00:00Z"
}
```

### PMI (Prospective Memory Intent)
```json
{
  "pmi_id": "pmi_2025_10_19_TON_watch_6.05",
  "title": "Присмотреться к покупке TON при ~6.05",
  "triggers": {"time":{"review_cron":"0 */6 * * *"},
    "market":{"price_near":{"symbol":"TONUSDT","target":6.05,"tolerance_pct":0.5}},
    "regime":{"require":["trend_up"],"forbid":["crash"]}},
  "actions":[{"type":"research.scan","args":{"scanner":"bollinger_squeeze"}},
             {"type":"alert.init","args":{"channel":"telegram","ttl_min":60}}],
  "priority":0.62,"confidence":0.55,"status":"short_term"
}
```

### World‑Model Lenses / Inference
```json
{
  "lens": "market_regime",
  "view": {"regime":"trend_up","confidence":0.74,"features":{"vola":"normal","breadth":"wide"}}
}
```

### Унифицированный RPC ответ
```json
{"ok": {"data": {}}}
{"err": {"code":"UNAVAILABLE","msg":"binance-mcp timeout","details":{}}}
{"meta": {"latency_ms": 142}}
```

---

## 4) supervisor‑mcp (итоговая роль и API)
**Роль:** единый «надсмотрщик» экосистемы.
- **Discovery/Registry**: `GET /mcp/registry`, `GET /mcp/{name}/capabilities`
- **Health**: `GET /health`, `GET /health/{name}`
- **Ingest**: `POST /ingest/metric`, `POST /ingest/event`
- **Query**: `GET /query/metrics`, `GET /query/agg?kind=business&window=7d`, `GET /query/facts`
- **Alerts**: правила и активные алерты; действия‑рекомендации (Pause/Reduce/CloseWorst)

**Хранилище:** Postgres (metrics, facts), материализованные представления для окон 7/30d, опционально Timescale.

---

## 5) learning‑mcp (офлайн‑обучение и A/B)
- EMA‑коррекция весов и порогов; guardrails: `max_change_per_epoch`, минимальная уверенность.
- API: `/train/offline`, `/propose`, `/ab/start|stop`, `/promote`, `/profiles/latest`.
- Критерии промо: `expectancy_B − expectancy_A ≥ +0.05` и `maxDD_B ≤ maxDD_A + 0.5%` при бюджете ≤1%.

---

## 6) World‑Model MCP (включая Prospective Memory)
- **Lenses**: market_regime, liquidity_basic, tech_snapshot — возвращают value+confidence.
- **Personas**: conservative/balanced/opportunistic (yaml профили ценностей/лимитов/стиля).
- **Intent→Plan**: конструктор линейных планов (без side‑effects).
- **Prospective Memory**: PMI create/list/evaluate/snooze/archive, price_near/cron/режим триггеры.
- **What‑If/Simulation**: лёгкие стресс‑сценарии.
- **Confidence & Causality** в ответах; калибровка (Brier/ACE) → настройка риска.

---

## 7) Упрощения и оптимизации (QA‑резюме)
- Объединить logging→в supervisor; config→в policy; отложить online‑learning.
- Один внутренний контракт RPC (JSON‑RPC over HTTP), для stdio — адаптеры.
- 3 Telegram‑мастера: Анализ монеты / Портфель / Память.
- Мини‑метрики v1: `rpc_success`, `rpc_latency_p95`, `error_rate`, `queue_depth`, `portfolio_drawdown`, `decision_confidence_avg`.
- Blue/Green деплой критичных; фича‑флаги и тёмные запуски.

---

## 8) Ошибки, отказоустойчивость, безопасность
- Шаблон вызова: `retry(backoff) → circuit → fallback(cache/noop) → escalate`.
- Edge‑cases: `MCP down`, `partial down`, `network split`, `timeout` — один runbook на кейс.
- Политики: **dry‑run + confirm** для side‑effects, 2‑key > X USD, роли `view|train|promote|trade`.
- Secrets в Vault/ENV, ноль ключей в логах; audit‑trail через Facts + Memory‑артефакты.

---

## 9) Дорожная карта (после Instructions)
- **Rα (1–2 нед.)**: Supervisor v1 (ingest/query/health), HALv1 отправляет метрики/факты.
- **Rβ (3–4 нед.)**: Learning v1 (train/propose), Policy: профили с ETag; HALv1 читает профиль.
- **Rγ (5–6 нед.)**: A/B рамка, алерты Supervisor, канареечные изменения ≤1%, TG‑карточки.
- **Rδ (7–10 нед.)**: World‑Model v0.1 (lenses/personas/intent→plan/PMI), What‑If.
- **Rε (11–14 нед.)**: Калибровка уверенности, причинность, интеграция с Learning‑loop.

> Полная программа (с буферами и стабилизацией HALv1) — 24–45 недель.

---

## 10) Критерии успеха (DoD)
- Supervisor надёжен (ingest≥99%, query p95<200 мс), health отражает реальные состояния.
- Learning ежедневно предлагает профиль; A/B и промо по критериям; guardrails соблюдаются.
- HALv1 стабилен (48 ч без зависаний), все e2e «анализ→решение→dry‑run→confirm→исполнение→лог» зелёные.
- World‑Model даёт осмысленные lenses/personas/PMI; объяснения (confidence+provenance) понятны.
- Память — SSOT: факты/артефакты воспроизводят решения; есть post‑mortems и skills‑кристаллы.

---

## 11) Следующие шаги
1. Завести репо `supervisor-mcp` и `learning-mcp` с минимальными OpenAPI/SQL (из спеки).  
2. Добавить в HALv1 клиента для `/ingest/*` и генерацию Facts.  
3. Подготовить YAML‑профили personas и начальный DecisionProfile.  
4. Срез «TONUSDT»: полный цикл (анализ→решение→dry‑run→confirm→исполнение→Fact/метрики→предложение профиля).  
5. Начать реализацию World‑Model v0.1 (3 lenses + PMI).

