# Memory 2.0: типизированная граф‑память + политики извлечения

## Краткий вывод
Переходим от «кучи текстов + эмбеддингов» к типизированному графу знаний (TMG) с контрактами «событие‑сущности‑связи‑версии». Это снижает шум в RAG и повышает объяснимость.

---

## 1. Требования
- Релевантность при длинной истории (диверсификация и квоты).
- Свежесть (time‑decay) и надёжность (authority).
- Прозрачность происхождения (provenance).
- Лёгкая эксплуатация на Mac (SQLite + FAISS).

---

## 2. Схема типизированного графа
### 2.1. Типы узлов
- Entity — персоны/проекты/термины (name, aliases, source, conf).
- Event — действия/факты во времени (timestamp, actors, summary).
- DocChunk — фрагменты документа/чата (text, doc_id, span).
- Topic — тематические кластеры/лейблы.
- ToolCall — вызовы инструментов (функция, args, статус).

### 2.2. Типы рёбер
- relates_to(Entity↔Entity|Event|DocChunk) — семантическая связь.
- causes(Event→Event) — причинность.
- mentions(DocChunk→Entity|Event) — упоминания.
- has_topic(*→Topic) — тематическая привязка.
- authored_by(DocChunk|Event→Entity) — авторство.

### 2.3. Свойства
- Узлы: source, confidence ∈ [0,1], version, created_at, updated_at.
- Рёбра: weight, evidence (ссылки на DocChunk.id).

---

## 3. Хранилище и индексы
- SQLite таблицы nodes/edges (props JSON), FTS5 для DocChunk.
- FAISS (HNSW/IVF‑PQ) для эмбеддингов, мапа chunk_id → vector_id.

```sql
CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT NOT NULL, props JSON NOT NULL, created_at INTEGER, updated_at INTEGER);
CREATE TABLE edges (id TEXT PRIMARY KEY, src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL, props JSON NOT NULL, created_at INTEGER, updated_at INTEGER);
CREATE VIRTUAL TABLE doc_chunks_fts USING fts5(content, doc_id, chunk_id, tokenize='porter');
```

---

## 4. Инжест и канонизация
Парсинг → DocChunk; NER/Linking → Entity; merge‑эвристики; Event‑выделение; Topic‑кластеры; provenance.

---

## 5. Политики извлечения
### 5.1. Кандидаты
- FAISS top‑200; FTS5 BM25 top‑200; fuse(alpha).

### 5.2. Диверсификация/веса
- MMR/α‑nDCG по topic/source/type; time‑decay; authority.

### 5.3. Квоты
- Event:10, DocChunk:10, Entity:4; лимит по токенам в PromptComposer.

```python
def retrieve(query, k=24):
    C_vec = faiss_search(query, top=200)
    C_bm25 = fts5_search(query, top=200)
    C = fuse(C_vec, C_bm25, alpha=0.6)
    C = diversify(C, by=["topic","source","type"], strength=0.5)
    score = 0.7*C.sim + 0.2*time_decay(C.ts, half_life=48*3600) + 0.1*authority(C.source)
    return quota_topk(C, score, k=k, quotas={"Event":10,"DocChunk":10,"Entity":4})
```

---

## 6. Компакция и архив
L2→L∞: суммари старых DocChunk с evidence; OPQ/PQ; WAL‑лог изменений.

---

## 7. API
```python
class MemoryGraph:
    def upsert_node(self, type: str, props: dict) -> str: ...
    def upsert_edge(self, src: str, dst: str, type: str, props: dict) -> str: ...
    def retrieve(self, query: str, k: int = 24) -> list: ...
```

---

## 8. Интеграция
ModuleCoordinator получает сбалансированный контекст; RAG использует evidence для faithfulness.

---

## 9. Тесты и метрики
- Quality: nDCG@k, Recall@k, % цитат; Perf: p95 retrieve < 80 ms при N≈50k.

---

## 10. Миграция
Импорт текущих чанков → DocChunk; синтез Entity/Event; построение связей; пересбор индексов; флаг fallback.

---

## 11. План внедрения
- Спринт 1: ядро (DDL, retrieve, индексы). DoD: p95<80ms, интеграция с RAG.
- Спринт 2: сущности/события, рёбра, цитаты. DoD: рост nDCG vs baseline.
- Спринт 3: компактор, OPQ/PQ, WAL. DoD: восстановление без потерь ссылок.

---

## 12. Риски
| Риск | Влияние | Митигация |
|---|---|---|
| Ошибки Linking | Искажения графа | Консервативные пороги, ручное исправление |
| Рост кардинальности | Падение perf | Квоты/сэмплирование, PQ, батч‑обновления |
| Дрейф тем | Потеря релевантности | Переобучение кластеров по расписанию |

---

## 📚 Связанные документы

- [OPTIMIZATION_AND_MEMORY_STRATEGY.md](OPTIMIZATION_AND_MEMORY_STRATEGY.md) — объединенная стратегия оптимизации и памяти
- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) — детальный план улучшений проекта
- [ARCHITECTURE.md](ARCHITECTURE.md) — архитектура системы
