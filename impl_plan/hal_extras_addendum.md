# 📎 Дополнение: Что не вошло в «HAL_Autonomous_Agent_Overview.md»

> Этот аддендум собирает полезные детали и заготовки, которые остались за рамками обзорного документа: проверки готовности, расширенные практики эксплуатации, шаблоны OpenAPI/SQL, матрицы тестов, модели угроз, управление схемами/версиями, бюджет/ёмкость и пр.

**Текущий статус (28 октября 2025):**
- **Общий прогресс:** ~45-50%
- **Supervisor MCP:** ~50% (базовые API готовы, требуются интеграция и доработка)
- **Learning MCP:** ~35% (offline training работает, требуется интеграция)
- **Policy MCP:** ~25% (Decision Profiles готовы)
- **Orchestrator MCP:** ~20% (базовая структура)

---

## 1) RACI и роли (минимум)
| Область | Responsible | Accountable | Consulted | Informed |
|---|---|---|---|---|
| Supervisor ingest/query | DevOps, Backend | Tech Lead | Data | All |
| Learning train/propose | ML | Tech Lead | Risk | All |
| Policy/Profiles промо | Risk | Product | ML, Backend | All |
| A/B rollout | Backend | Tech Lead | ML, Product | All |
| Incidents P0 | On-call | Tech Lead | Security, Product | All |

---

## 2) Runbooks (шаблоны)
**RB‑001: MCP недоступен**
1) Проверить `/health/{name}` в supervisor; 2) если `down` → перезапуск; 3) включить fallback (cache/noop); 4) открыть инцидент P1; 5) ретест; 6) post‑mortem факт `Fact:Incident`.

**RB‑002: Скачок задержек**
1) Снять p99 трасс supervisor; 2) включить деградацию (уменьшить частоту сканеров); 3) проверить GC/IO; 4) алерт Risk если влияет на трейдинг; 5) откат последнего деплоя при корреляции.

**RB‑003: Негативный expectancy 3 дня**
1) Остановить auto‑promote; 2) понизить риск‑кап; 3) включить более строгую persona; 4) инициировать review ML профиля.

---

## 3) OpenAPI скелеты (сверхкратко)
### 3.1. Supervisor (фрагменты)
```yaml
openapi: 3.0.3
info: {title: supervisor-mcp, version: 1.0.0}
paths:
  /ingest/metric:
    post:
      requestBody: {required: true, content: {application/json: {schema: {oneOf: [{$ref: '#/components/schemas/Metric'}, {type: array, items: {$ref: '#/components/schemas/Metric'}}]}}}}
      responses: {"200": {description: ok}}
  /ingest/event:
    post: {responses: {"200": {description: ok}}}
  /query/agg:
    get:
      parameters: [{name: kind, in: query, schema: {enum: [business, tech]}}, {name: window, in: query, schema: {type: string}}]
      responses: {"200": {content: {application/json: {schema: {type: object}}}}}
components:
  schemas:
    Metric:
      type: object
      properties: {ts: {type: string, format: date-time}, name: {type: string}, value: {type: number}, tags: {type: object}}
```

### 3.2. Learning (фрагменты)
```yaml
openapi: 3.0.3
info: {title: learning-mcp, version: 1.0.0}
paths:
  /train/offline:
    post: {parameters: [{name: window, in: query, schema: {type: string, example: '30d'}}], responses: {"200": {description: ok}}}
  /propose:
    post: {responses: {"200": {content: {application/json: {schema: {type: object}}}}}}
  /ab/start:
    post: {responses: {"200": {description: ok}}}
```

### 3.3. World‑Model (фрагменты)
```yaml
openapi: 3.0.3
info: {title: world-model-mcp, version: 0.1.0}
paths:
  /state/snapshot: {get: {responses: {"200": {description: ok}}}}
  /lens/appraise: {post: {responses: {"200": {description: ok}}}}
  /intent/construct: {post: {responses: {"200": {description: ok}}}}
  /pmi/create: {post: {responses: {"200": {description: ok}}}}
```

---

## 4) SQL заготовки
```sql
-- Supervisor
CREATE TABLE metrics (
  ts timestamptz NOT NULL,
  name text NOT NULL,
  value double precision NOT NULL,
  tags jsonb NOT NULL,
  PRIMARY KEY (ts, name, (tags->>'svc'))
);
CREATE TABLE facts (
  ts timestamptz NOT NULL,
  kind text NOT NULL,
  actor text NOT NULL,
  correlation_id text,
  payload jsonb NOT NULL
);
CREATE INDEX ON metrics (name, ts DESC);
CREATE INDEX ON facts (kind, ts DESC);

-- Learning
CREATE TABLE decision_profiles (
  profile_id text PRIMARY KEY,
  body jsonb NOT NULL,
  created_at timestamptz DEFAULT now()
);
```

---

## 5) Матрица тестирования (минимум)
| Категория | Кейс | Ожидаемое |
|---|---|---|
| Ingest | Потеря сети при батче | ретраи, без дубликатов |
| Query | Окно 7d/30d | корректные агрегаты |
| Learning | propose на синтетике | валидные дельты, caps соблюдены |
| A/B | бюджет 1% | не превышен, телеметрия пишется |
| Persona | смена профиля | план пересчитан |
| PMI | price_near с tol | срабатывание + ресёрч |

---

## 6) Модель угроз (TMM) и меры
- **Ключи/API**: угрозы утечки → Vault, KMS, короткие TTL, ротация.
- **Инъекции**: в ingest/query → строгие схемы и лимиты размера.
- **Replay**: идемпотентность по ключам событий.
- **DoS**: rate‑limit, backpressure, circuit‑breaker.
- **Supply‑chain**: pin зависимостей, SLSA‑уровень, подпись артефактов.

---

## 7) Управление схемами и версиями
- **Schema Registry** (JSON Schema): версии с `compat: backward`.
- **Контракты**: автоген клиентов; тест «золотых» ответов.
- **Фича‑флаги**: `dark launch → canary → 100%`.

---

## 8) Ёмкость и бюджет (оценка)
- Supervisor: 1×vCPU/1–2GB RAM (ingest до 1k eps), Postgres 2–4GB RAM.
- Learning: CPU‑достаточно (EMA), ежедневные джобы < 5 мин; GPU не требуется на v1.
- Хранилище: metrics ~ 5–10GB/мес (при 1k eps, gzip), facts зависит от частоты сделок; S3 для архивов.

---

## 9) Chaos/Resilience план
- Еженедельный `MCP down` game day.
- Ежемесячный `network split` тест.
- Квартальный disaster‑recovery: восстановление из snapshot (RTO<30 мин, RPO<15 мин).

---

## 10) UX нюансы Telegram
- Таймауты на callback 60–90с, идемпотентность по `cb_id+hash`.
- Короткие карточки TL;DR + кнопки: «Dry‑run», «Подробнее», «Snooze».
- Логику мастеров хранить в Redis (TTL по сессии).

---

## 11) Глоссарий (коротко)
- **PMI** — Prospective Memory Intent, планируемое воспоминание.
- **Persona** — профиль ценностей/ограничений поведения агента.
- **Lens** — точка взгляда/перспектива модели мира.
- **DecisionProfile** — веса/пороги для принятия решений.
- **SSOT** — Single Source of Truth (единый источник истины).

---

## 12) Бэклог «удалить/объединить/отложить»
- Отложить online‑learning до появления статистически значимого выигрыша.
- Объединить logging → в supervisor, config‑флаги → в policy.
- Убрать сложные графы оркестрации до стабилизации линейных планов.

---

## 13) Контрольные чек‑листы релизов
**Rα:** ingest/query/health ok, HAL отправляет метрики/факты.  
**Rβ:** train/propose ok, профили через ETag, HAL читает профиль.  
**Rγ:** A/B с бюджетом≤1%, алерты supervisor, TG‑карточки.  
**Rδ:** World‑Model v0.1 (3 lenses + PMI), snapshot и intent→plan работают.

