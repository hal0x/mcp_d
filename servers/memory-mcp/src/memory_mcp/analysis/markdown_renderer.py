#!/usr/bin/env python3
"""Модуль для рендеринга Markdown отчётов."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.datetime_utils import format_datetime_display
from ..utils.naming import slugify
from ..utils.paths import find_project_root

logger = logging.getLogger(__name__)


class MarkdownRenderer:
    """Класс для рендеринга Markdown отчётов"""

    def __init__(
        self,
        output_dir: Path = Path("artifacts/reports"),
        chat_links_path: Optional[Path] = None,
    ):
        """Инициализация рендерера."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        if chat_links_path is None:
            project_root = find_project_root(Path(__file__).parent)
            chat_links_path = project_root / "config" / "chat_links.json"
        self.chat_links = self._load_chat_links(chat_links_path)

    def render_session_summary(
        self,
        summary: Dict[str, Any],
        chat_links: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Path]:
        """Создаёт Markdown и JSON артефакты сессии по канонической схеме."""

        meta = summary.get("meta", {})
        chat_name = meta.get("chat_name", "Unknown chat")
        chat_id = summary.get("chat_id") or self._safe_name(chat_name)
        session_id = summary.get("session_id", "session")
        profile = meta.get("profile", "group-project")
        quality = summary.get("quality", {})
        quality_status = quality.get("status", "accepted")

        sessions_dir = self.output_dir / chat_id / "sessions"
        try:
            sessions_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Ошибка прав доступа при создании директории {sessions_dir}: {e}")
            # Пытаемся создать в альтернативном месте или просто логируем ошибку
            raise

        json_path = sessions_dir / f"{session_id}.json"
        md_filename = (
            f"{session_id}-needs-review.md"
            if quality_status == "needs_review"
            else f"{session_id}.md"
        )
        md_path = sessions_dir / md_filename

        if not force and json_path.exists() and md_path.exists():
            logger.info(f"Артефакты сессии уже существуют: {md_path}, {json_path}")
            return {"markdown": md_path, "json": json_path}

        try:
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, ensure_ascii=False, indent=2)
        except PermissionError as e:
            logger.error(f"Ошибка прав доступа при сохранении {json_path}: {e}")
            raise

        effective_chat_links = chat_links if chat_links is not None else self.chat_links
        
        if profile == "broadcast":
            content = self._render_broadcast_markdown(summary)
        else:
            content = self._render_group_markdown(summary, effective_chat_links)

        if quality_status == "needs_review":
            banner = "> ⚠️ **Этот отчёт помечен как needs_review.** Проверить данные вручную перед использованием.\n\n"
            content = banner + content

        try:
            with open(md_path, "w", encoding="utf-8") as fp:
                fp.write(content)
        except PermissionError as e:
            logger.error(f"Ошибка прав доступа при сохранении {md_path}: {e}")
            raise

        logger.info(f"Созданы артефакты сессии: {md_path}, {json_path}")
        return {"markdown": md_path, "json": json_path}

    def render_chat_index(
        self, chat: str, sessions: List[Dict[str, Any]], force: bool = False, has_new_data: bool = False
    ) -> Path:
        """Создаёт JSON индекс сессий для чата."""
        chat_id = self._safe_name(chat)
        chat_dir = self.output_dir / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)
        index_path = chat_dir / "index.json"

        if not force and not has_new_data and index_path.exists():
            logger.info(f"Индекс чата уже существует: {index_path}")
            return index_path

        entries: List[Dict[str, Any]] = []
        for session in sessions:
            session_id = session.get("session_id", "")
            meta = session.get("meta", {})
            quality = session.get("quality", {})
            quality_status = quality.get("status", "accepted")
            kpi = quality.get("kpi", {})
            flags = quality.get("flags", {})
            md_filename = (
                f"{session_id}-needs-review.md"
                if quality_status == "needs_review"
                else f"{session_id}.md"
            )
            entry = {
                "session_id": session_id,
                "time_span": meta.get("time_span", ""),
                "messages_total": meta.get("messages_total", 0),
                "profile": meta.get("profile", ""),
                "addons": meta.get("addons", []),
                "policy_flags": meta.get("policy_flags", []),
                "quality": {
                    "score": quality.get("score", 0),
                    "status": quality_status,
                    "kpi": {
                        "coverage": kpi.get("coverage"),
                        "claims_coverage": kpi.get("claims_coverage"),
                        "topics": kpi.get("topics"),
                        "actions": kpi.get("actions"),
                        "risks": kpi.get("risks"),
                        "threads": kpi.get("threads"),
                    },
                    "flags": flags,
                },
                "counts": {
                    "topics": len(session.get("topics", [])),
                    "claims": len(session.get("claims", [])),
                    "discussion": len(session.get("discussion", [])),
                    "actions": len(session.get("actions", [])),
                    "risks": len(session.get("risks", [])),
                },
                "paths": {
                    "markdown": f"sessions/{md_filename}",
                    "json": f"sessions/{session_id}.json",
                },
            }
            entries.append(entry)

        payload = {
            "chat_id": chat_id,
            "chat_name": chat,
            "updated_at": datetime.now().isoformat(),
            "sessions_total": len(entries),
            "sessions": entries,
        }

        with open(index_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        logger.info(f"Создан индекс чата: {index_path}")
        return index_path

    def _render_broadcast_markdown(self, summary: Dict[str, Any]) -> str:
        meta = summary.get("meta", {})
        quality = summary.get("quality", {})
        topics = summary.get("topics", [])
        claims = summary.get("claims", [])
        discussion = summary.get("discussion", [])
        uncertainties = summary.get("uncertainties", [])
        attachments = summary.get("attachments", [])

        lines = []
        lines.append(
            f"# {meta.get('chat_name', 'Чат')} — {summary.get('session_id', '')}"
        )
        lines.append("")
        lines.append(
            f"Период: {meta.get('time_span', 'N/A')} · Профиль: {meta.get('profile', 'broadcast')} · "
            f"Сообщений: {meta.get('messages_total', 0)} · Уверенность: {self._format_float(meta.get('confidence'))} · "
            f"Quality: {quality.get('score', 0)}"
        )
        lines.append("")

        lines.append("## Topics")
        if topics:
            for topic in topics:
                lines.append(
                    f"- **{topic.get('title', 'Тема')}** • {topic.get('time_span', '')}"
                )
                lines.append(f"  {topic.get('summary', '')}")
        else:
            lines.append("- (Темы не определены)")
        lines.append("")

        lines.append("## Claims")
        if claims:
            lines.append(self._format_claims_table(claims))
        else:
            lines.append("- (Утверждений не зафиксировано)")
        lines.append("")

        lines.append("## Timeline")
        if discussion:
            for item in discussion:
                lines.append(self._format_timeline_item(item))
        else:
            lines.append("- (Цитаты не подобраны)")
        lines.append("")

        lines.append("## Uncertainties")
        if uncertainties:
            for entry in uncertainties:
                lines.append(f"- {entry}")
        else:
            lines.append("- (Неопределённости не выделены)")
        lines.append("")

        lines.append("## Attachments")
        lines.extend(self._render_attachments_section(attachments))
        lines.append("")

        lines.append(f"## Rationale\n{summary.get('rationale', 'no_risks_detected')}")

        return "\n".join(lines)

    def _render_group_markdown(
        self, summary: Dict[str, Any], chat_links: Optional[Dict[str, Any]] = None
    ) -> str:
        meta = summary.get("meta", {})
        quality = summary.get("quality", {})
        topics = summary.get("topics", [])
        discussion = summary.get("discussion", [])
        actions = summary.get("actions", [])
        risks = summary.get("risks", [])
        uncertainties = summary.get("uncertainties", [])
        attachments = summary.get("attachments", [])

        lines = []
        lines.append(
            f"# {meta.get('chat_name', 'Чат')} — {summary.get('session_id', '')}"
        )
        lines.append("")
        participant_str = ", ".join(meta.get("participants", []))
        lines.append(
            f"Период: {meta.get('time_span', 'N/A')} · Профиль: {meta.get('profile', 'group-project')} · "
            f"Сообщений: {meta.get('messages_total', 0)} · Участники: {participant_str or '—'} · "
            f"Quality: {quality.get('score', 0)}"
        )
        lines.append("")

        lines.append("## Topics")
        if topics:
            for topic in topics:
                lines.append(
                    f"- **{topic.get('title', 'Тема')}** • {topic.get('time_span', '')}"
                )
                lines.append(f"  {topic.get('summary', '')}")
        else:
            lines.append("- (Темы не определены)")
        lines.append("")

        lines.append("## Discussion")
        if discussion:
            for item in discussion:
                lines.append(self._format_timeline_item(item))
        else:
            lines.append("- (Ключевые цитаты не выделены)")
        lines.append("")

        lines.append("## Actions")
        if actions:
            for action in actions:
                lines.append(
                    self._format_action_item(
                        action, meta.get("chat_name", ""), chat_links
                    )
                )
        else:
            lines.append("- (Действия не зафиксированы)")
        lines.append("")

        lines.append("## Risks")
        if risks:
            for risk in risks:
                lines.append(self._format_risk_item(risk))
        else:
            lines.append("- (Рисков не выявлено)")
        lines.append("")

        lines.append("## Uncertainties")
        if uncertainties:
            for entry in uncertainties:
                lines.append(f"- {entry}")
        else:
            lines.append("- (Открытых вопросов нет)")
        lines.append("")

        lines.append("## Attachments")
        lines.extend(self._render_attachments_section(attachments))
        lines.append("")

        lines.append(
            f"## Rationale\n{summary.get('rationale', 'project_session_with_actions')}"
        )

        return "\n".join(lines)

    def _format_claims_table(self, claims: List[Dict[str, Any]]) -> str:
        header = "| Time | Source | Credibility | Entities | Summary |\n| --- | --- | --- | --- | --- |"
        rows = []
        for claim in claims:
            ts = self._format_time(claim.get("ts"))
            source = claim.get("source", "")
            credibility = claim.get("credibility", "")
            entities = ", ".join(claim.get("entities", []))
            summary = claim.get("summary", "")
            rows.append(f"| {ts} | {source} | {credibility} | {entities} | {summary} |")
        return "\n".join([header] + rows)

    def _format_timeline_item(self, item: Dict[str, Any]) -> str:
        ts = self._format_time(item.get("ts"))
        author = item.get("author", "")
        quote = item.get("quote", "")
        msg_id = item.get("msg_id")
        suffix = f" (msg: {msg_id})" if msg_id else ""
        return f"- [{ts}] {author} · «{quote}»{suffix}"

    def _format_action_item(
        self, action: Dict[str, Any], chat: str, chat_links: Optional[Dict[str, Any]]
    ) -> str:
        checkbox = "- [ ]"
        text = action.get("text", "")
        owner = action.get("owner") or ""
        due_raw = action.get("due_raw") or action.get("due") or ""
        priority = action.get("priority", "normal")
        msg_id = action.get("msg_id")

        owner_part = f" — owner: {owner}" if owner else ""
        due_part = f" — due: {due_raw}" if due_raw else ""
        priority_part = f" — pri: {priority}"

        deeplink = (
            self._generate_deeplink(chat, {"msg_id": msg_id}, chat_links)
            if msg_id
            else None
        )
        link_part = f" ↗ {deeplink}" if deeplink else ""

        fallback = self._generate_fallback(chat, {"msg_id": msg_id}) if msg_id else ""
        fallback_part = f" ({fallback})" if fallback else ""

        return f"{checkbox} {text}{owner_part}{due_part}{priority_part}{link_part}{fallback_part}"

    def _format_risk_item(self, risk: Dict[str, Any]) -> str:
        text = risk.get("text", "")
        likelihood = risk.get("likelihood") or "—"
        impact = risk.get("impact") or "—"
        mitigation = risk.get("mitigation")
        msg_id = risk.get("msg_id")
        detail = f" (L:{likelihood}, I:{impact})"
        if mitigation:
            detail += f" · Mitigation: {mitigation}"
        if msg_id:
            detail += f" · msg: {msg_id}"
        return f"- {text}{detail}"

    def _render_attachments_section(self, attachments: List[str]) -> List[str]:
        if not attachments:
            return ["- (Артефактов не найдено)"]
        lines = []
        for attachment in attachments[:20]:
            if ":" in attachment:
                kind, value = attachment.split(":", 1)
                lines.append(f"- **{kind}**: {value}")
            else:
                lines.append(f"- {attachment}")
        return lines

    def _format_time(self, iso: Optional[str]) -> str:
        """Форматирует ISO строку времени для отображения."""
        from ..utils.datetime_utils import format_datetime_display

        return format_datetime_display(iso, format_type="datetime", fallback=iso or "—")

    def _format_float(self, value: Optional[float]) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.2f}"
        except Exception:
            return str(value)

    def render_chat_summary(
        self,
        chat: str,
        sessions: List[Dict[str, Any]],
        top_sessions: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
        has_new_data: bool = False,
    ) -> Path:
        """Генерирует сводный отчёт по чату в Markdown."""
        chat_id = self._safe_name(chat)
        file_path = self.output_dir / f"{chat_id}.md"

        if not force and not has_new_data and file_path.exists():
            logger.info(f"Сводка чата уже существует: {file_path}")
            return file_path

        participants = set()
        start_times = []
        end_times = []
        for session in sessions:
            meta = session.get("meta", {})
            participants.update(meta.get("participants", []))
            if meta.get("start_time_utc"):
                start_times.append(meta["start_time_utc"])
            if meta.get("end_time_utc"):
                end_times.append(meta["end_time_utc"])

        if start_times and end_times:
            time_range = f"{min(start_times)[:10]} — {max(end_times)[:10]}"
        else:
            time_range = "N/A"

        lines = []
        lines.append(f"# Сводка чата {chat}")
        lines.append("")
        lines.append(
            f"Всего сессий: {len(sessions)} · Период: {time_range} · Обновлено: {format_datetime_display(datetime.now(timezone.utc), format_type='date')}"
        )
        lines.append(
            f"Участники: {', '.join(sorted(participants)) if participants else '—'}"
        )
        lines.append("")

        lines.append("## 📌 Актуально за 30 дней")
        if top_sessions:
            for i, session in enumerate(top_sessions[:10], 1):
                meta = session.get("meta", {})
                session_id = session.get("session_id", "N/A")
                span = meta.get("time_span", "")
                score = session.get("quality", {}).get("score", 0)
                context = " ".join(
                    topic.get("summary", "") for topic in session.get("topics", [])[:1]
                )
                lines.append(
                    f"{i}. **[{session_id}](sessions/{session_id}.md)** ({span}) · Score: {score} · {context}"
                )
        else:
            lines.append("_(Нет данных)_")
        lines.append("")

        lines.append("## 📝 Последние сессии")
        for session in reversed(sessions[-10:]):
            meta = session.get("meta", {})
            session_id = session.get("session_id", "N/A")
            span = meta.get("time_span", "N/A")
            score = session.get("quality", {}).get("score", 0)
            topic = session.get("topics", [{}])[0]
            summary = topic.get("summary", "")
            lines.append(
                f"- **[{session_id}](sessions/{session_id}.md)** · {span} · Score: {score} · {summary}"
            )
        lines.append("")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Создан файл сводки чата: {file_path}")
        return file_path

    def render_snippets(self, session: Dict[str, Any], force: bool = False) -> Path:
        """Рендеринг ключевых сниппетов сессии."""
        session_id = session["session_id"]
        chat = session["chat"]

        snippets_dir = self.output_dir / self._safe_name(chat) / "snippets"
        snippets_dir.mkdir(parents=True, exist_ok=True)

        file_path = snippets_dir / f"{session_id}.jsonl"

        if not force and file_path.exists():
            logger.info(f"Файл сниппетов уже существует: {file_path}")
            return file_path

        messages = session.get("messages", [])
        key_messages = self._select_key_messages(messages)

        with open(file_path, "w", encoding="utf-8") as f:
            for msg in key_messages:
                snippet = {
                    "msg_id": msg.get("id", ""),
                    "text": msg.get("text", "")[:220],
                    "date": msg.get("date_utc", ""),
                    "from": msg.get("from", {}),
                }
                f.write(json.dumps(snippet, ensure_ascii=False) + "\n")

        logger.info(f"Создан файл сниппетов: {file_path}")
        return file_path

    def _select_key_messages(
        self, messages: List[Dict[str, Any]], max_count: int = 5
    ) -> List[Dict[str, Any]]:
        """Отбор ключевых сообщений из сессии."""
        if len(messages) <= max_count:
            return messages

        key_messages = [messages[0]]
        
        sorted_by_length = sorted(
            messages[1:-1], key=lambda x: len(x.get("text", "")), reverse=True
        )
        key_messages.extend(sorted_by_length[: max_count - 2])
        key_messages.append(messages[-1])
        
        key_messages.sort(key=lambda x: x.get("date_utc", ""))

        return key_messages[:max_count]

    def _group_sessions_by_month(
        self, sessions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for session in sessions:
            start_time = session.get("meta", {}).get("start_time_utc")
            if not start_time:
                continue
            month_key = start_time[:7]
            grouped.setdefault(month_key, []).append(session)
        return grouped

    def _generate_deeplink(
        self,
        chat: str,
        decision: Dict[str, Any],
        chat_links: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Генерация Telegram deeplink."""
        if not chat_links:
            return None

        chats_config = chat_links.get("chats", {})
        chat_config = chats_config.get(chat, {})

        if chat_config.get("type") == "public":
            domain = chat_config.get("domain")
            msg_id = decision.get("msg_id")

            if domain and msg_id:
                return f"tg://resolve?domain={domain}&message_id={msg_id}"

        return None

    def _generate_fallback(self, chat: str, decision: Dict[str, Any]) -> str:
        """Генерация fallback ссылки."""
        msg_id = decision.get("msg_id", "")
        date = decision.get("date", "")[:10] if decision.get("date") else "unknown"

        return f"/chats/{chat}/{date}.json#msg={msg_id}"

    def _safe_name(self, name: str) -> str:
        """Создаёт безопасный slug для директорий/файлов."""
        return slugify(name)

    def _load_chat_links(self, chat_links_path: Path) -> Optional[Dict[str, Any]]:
        """Загружает конфигурацию chat_links из JSON файла."""
        if not chat_links_path.exists():
            logger.debug(f"Файл chat_links не найден: {chat_links_path}")
            return None

        try:
            with open(chat_links_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.debug(f"Загружена конфигурация chat_links из {chat_links_path}")
                return data
        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка парсинга chat_links.json: {e}")
            return None
        except Exception as e:
            logger.warning(f"Ошибка загрузки chat_links.json: {e}")
            return None

    def render_cumulative_context(
        self, chat: str, sessions: List[Dict[str, Any]], force: bool = False, has_new_data: bool = False
    ) -> Path:
        """Создаёт файл с накопленным контекстом чата."""
        chat_id = self._safe_name(chat)
        chat_dir = self.output_dir / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)

        file_path = chat_dir / f"{chat_id}_context.md"

        if not force and not has_new_data and file_path.exists():
            logger.info(f"Файл контекста уже существует: {file_path}")
            return file_path

        lines = []
        lines.append(f"# Накапливающийся контекст чата {chat}")
        lines.append("")

        lines.append(
            f"**Обновлено:** {format_datetime_display(datetime.now(timezone.utc), format_type='%Y-%m-%d %H:%M', timezone_name='Asia/Bangkok')} BKK"
        )
        lines.append(f"**Всего сессий:** {len(sessions)}")
        lines.append("")

        for index, session in enumerate(sessions, 1):
            meta = session.get("meta", {})
            session_id = session.get("session_id", "unknown")
            span = meta.get("time_span", "")
            topics = session.get("topics", [])
            summary = " ".join(topic.get("summary", "") for topic in topics[:2])
            if not summary:
                continue
            lines.append(f"## {index}. {session_id}")
            if span:
                lines.append(f"**Период:** {span}")
            lines.append("")
            lines.append(summary)
            lines.append("---")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Создан файл накапливающегося контекста: {file_path}")
        return file_path


if __name__ == "__main__":
    test_summary = {
        "version": "1.0.0",
        "chat_id": "TestChat",
        "session_id": "TestChat-D0001",
        "meta": {
            "chat_name": "TestChat",
            "profile": "group-project",
            "time_span": "2025-10-01 10:00 – 12:00 BKK",
            "messages_total": 24,
            "confidence": 0.92,
            "participants": ["alice", "bob"],
            "dominant_language": "ru",
            "chat_mode": "group",
            "start_time_utc": "2025-10-01T03:00:00+00:00",
            "end_time_utc": "2025-10-01T05:00:00+00:00",
        },
        "topics": [
            {
                "title": "План работ на неделю",
                "time_span": "2025-10-01 10:00 – 11:00 BKK",
                "message_ids": ["1", "2"],
                "summary": "Обсудили приоритеты и ответственных за задачи.",
            }
        ],
        "claims": [
            {
                "ts": "2025-10-01T10:05:00+07:00",
                "source": "internal",
                "modality": "internal",
                "credibility": "medium",
                "entities": ["alice"],
                "summary": "Команда согласовала список задач на неделю.",
                "msg_id": "1",
                "topic_title": "План работ на неделю",
            }
        ],
        "discussion": [
            {
                "ts": "2025-10-01T10:15:00+07:00",
                "author": "alice",
                "msg_id": "1",
                "quote": "Акцент на завершении отчёта по продукту.",
            }
        ],
        "actions": [
            {
                "text": "Подготовить отчёт к пятнице",
                "owner": "@alice",
                "due_raw": "2025-10-03",
                "due": "2025-10-03",
                "priority": "high",
                "status": "open",
                "msg_id": "1",
                "topic_title": "План работ на неделю",
            }
        ],
        "risks": [
            {
                "text": "Возможна задержка из-за отсутствия данных",
                "likelihood": "medium",
                "impact": "medium",
                "mitigation": None,
                "msg_id": "2",
                "topic_title": "План работ на неделю",
            }
        ],
        "uncertainties": [],
        "entities": ["alice", "bob"],
        "attachments": ["link:https://example.com/doc"],
        "rationale": "project_session_with_actions",
        "quality": {
            "score": 90,
            "status": "accepted",
            "kpi": {
                "coverage": 0.8,
                "claims_coverage": 0.6,
                "topics": 1,
                "actions": 1,
                "risks": 1,
                "threads": 0,
            },
            "details": {},
        },
        "raw_summary": "Draft text",
        "fallback_used": False,
        "_legacy": {},
    }

    renderer = MarkdownRenderer()
    paths = renderer.render_session_summary(test_summary, force=True)
    print(f"Markdown: {paths['markdown']}")
    print(f"JSON: {paths['json']}")
