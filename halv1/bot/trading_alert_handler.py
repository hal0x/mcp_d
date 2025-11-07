"""Trading Alert Handler with interactive Telegram buttons."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import ContextTypes
except ImportError:
    # Fallback для случаев когда python-telegram-bot не установлен
    InlineKeyboardButton = None
    InlineKeyboardMarkup = None
    Update = None
    ContextTypes = None

from .telegram_bot import TelegramBot

logger = logging.getLogger(__name__)

BINANCE_NATIVE_URL = os.getenv("HAL_TRADING_BINANCE_URL", "http://localhost:8000")
HAL_API_URL = os.getenv("HAL_TRADING_API_URL", "http://localhost:8080")


class TradingAlertHandler:
    """Обработчик торговых алертов с интерактивными кнопками."""

    def __init__(self, bot: TelegramBot):
        self.bot = bot

    async def send_alert(self, chat_id: int, signal: Dict[str, Any]) -> None:
        """Отправка алерта с интерактивными кнопками."""

        enriched_signal = dict(signal)
        await self._enrich_signal(enriched_signal)

        if InlineKeyboardMarkup is None:
            logger.warning("python-telegram-bot not available, sending simple alert")
            await self._send_simple_alert(chat_id, enriched_signal)
            return

        msg = self._format_signal(enriched_signal)
        keyboard = self._create_keyboard(enriched_signal)

        try:
            await self.bot.app.bot.send_message(
                chat_id=chat_id,
                text=msg,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML"
            )
            logger.info("trading_alert_sent", chat_id=chat_id, symbol=enriched_signal.get("symbol"))
        except Exception as e:
            logger.error("Failed to send trading alert: %s", e)
            await self._send_simple_alert(chat_id, enriched_signal)

    async def _send_simple_alert(self, chat_id: int, signal: Dict[str, Any]) -> None:
        """Отправка простого алерта без кнопок."""
        msg = self._format_signal(signal)
        try:
            await self.bot.app.bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Failed to send simple alert: {e}")

    def _format_signal(self, signal: Dict[str, Any]) -> str:
        """Форматирование сигнала в читаемое сообщение."""

        symbol = signal.get("symbol", "N/A")
        direction = str(signal.get("direction", "N/A"))
        entry = signal.get("entry", 0.0)
        confidence = signal.get("confidence")
        timeframe = signal.get("timeframe", "N/A")
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        leverage = signal.get("leverage")

        direction_emoji = "🟢" if direction.lower() == "long" else "🔴"
        entry_str = f"{entry:.4f}" if isinstance(entry, (int, float)) else str(entry)

        lines: list[str] = [
            f"{direction_emoji} <b>Торговый сигнал</b>",
            "",
            f"📊 <b>Символ:</b> {symbol}",
            f"⏰ <b>Таймфрейм:</b> {timeframe}",
            f"📈 <b>Направление:</b> {direction.upper()}",
            f"💰 <b>Вход:</b> {entry_str}",
        ]
        if isinstance(stop_loss, (int, float)):
            lines.append(f"🛑 <b>Stop-loss:</b> {stop_loss:.4f}")
        if isinstance(take_profit, (int, float)):
            lines.append(f"🎯 <b>Take-profit:</b> {take_profit:.4f}")
        if leverage:
            lines.append(f"⚖️ <b>Плечо:</b> {leverage}x")
        if confidence is not None:
            lines.append(f"🔥 <b>Уверенность:</b> {confidence}%")

        risk_info = signal.get("risk") or {}
        if risk_info:
            rr = risk_info.get("adjusted_rr") or risk_info.get("rr")
            adjusted_tp = risk_info.get("adjusted_tp")
            if rr:
                lines.append(f"📐 <b>Risk/Reward:</b> {rr:.2f}")
            if adjusted_tp and adjusted_tp != take_profit:
                lines.append(f"🔧 <b>Реком. TP:</b> {adjusted_tp:.4f}")

        metrics = signal.get("metrics") or {}
        if metrics:
            highlights: list[str] = []
            if metrics.get("winrate_pct") is not None:
                highlights.append(f"Winrate {metrics['winrate_pct']:.1f}%")
            if metrics.get("profit_factor") is not None:
                highlights.append(f"PF {metrics['profit_factor']:.2f}")
            if metrics.get("avg_trade_return_pct") is not None:
                highlights.append(f"Avg {metrics['avg_trade_return_pct']:.2f}%")
            if highlights:
                lines.append("📊 <b>Статистика:</b> " + " · ".join(highlights))

        reasons = signal.get("reasons")
        if reasons:
            if isinstance(reasons, list):
                reasons_str = ", ".join(str(item) for item in reasons[:3])
            else:
                reasons_str = str(reasons)
            lines.append("")
            lines.append(f"🔍 <b>Причины:</b> {reasons_str}")

        lines.append("")
        lines.append("<i>Выберите действие:</i>")
        return "\n".join(lines)

    def _create_keyboard(self, signal: Dict[str, Any]) -> list[list[Any]]:
        """Создание клавиатуры с кнопками."""
        
        if InlineKeyboardButton is None:
            return []
        
        signal_id = signal.get("id", "unknown")
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Взять", callback_data=f"take_{signal_id}"),
                InlineKeyboardButton("❌ Пропустить", callback_data=f"skip_{signal_id}")
            ],
            [InlineKeyboardButton("📊 Детали", callback_data=f"details_{signal_id}")]
        ]
        
        return keyboard

    async def _enrich_signal(self, signal: Dict[str, Any]) -> None:
        """Дополнительное обогащение сигнала перед отправкой."""
        signal.setdefault("id", signal.get("id") or signal.get("external_id") or "unknown")
        entry = signal.get("entry")
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        atr = signal.get("atr")
        side = signal.get("direction")
        if signal.get("risk") or None in (entry, stop_loss, take_profit, side) or not atr:
            return
        try:
            payload = {
                "entry": float(entry),
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "atr": float(atr),
                "side": str(side),
            }
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(f"{BINANCE_NATIVE_URL.rstrip('/')}/risk/evaluate", json=payload)
                response.raise_for_status()
            signal["risk"] = response.json()
        except Exception:
            logger.debug("risk_enrichment_failed", exc_info=True)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка нажатий на кнопки."""
        
        if Update is None or ContextTypes is None:
            logger.warning("python-telegram-bot not available for callback handling")
            return
        
        query = update.callback_query
        if not query:
            return
            
        await query.answer()
        
        try:
            action, signal_id = query.data.split("_", 1)
            
            if action == "take":
                await self._handle_take_action(query, signal_id)
            elif action == "skip":
                await self._handle_skip_action(query, signal_id)
            elif action == "details":
                await self._handle_details_action(query, signal_id)
            else:
                await query.edit_message_text("❌ Неизвестное действие")
                
        except ValueError:
            await query.edit_message_text("❌ Ошибка обработки действия")
        except Exception as e:
            logger.error(f"Error handling callback: {e}")
            await query.edit_message_text("❌ Произошла ошибка")

    async def _handle_take_action(self, query, signal_id: str) -> None:
        """Обработка действия 'Взять'."""
        
        # Сохраняем feedback в PostgreSQL через HALv1 API
        try:
            await self._save_feedback(signal_id, "take")
            
            # Обновляем сообщение
            await query.edit_message_text(
                query.message.text + "\n\n✅ <b>Сигнал принят!</b>",
                parse_mode="HTML"
            )
            
            logger.info(f"Signal {signal_id} taken by user")
            
        except Exception as e:
            logger.error(f"Failed to save take feedback: {e}")
            await query.edit_message_text(
                query.message.text + "\n\n⚠️ Ошибка сохранения действия",
                parse_mode="HTML"
            )

    async def _handle_skip_action(self, query, signal_id: str) -> None:
        """Обработка действия 'Пропустить'."""
        
        try:
            await self._save_feedback(signal_id, "skip")
            
            # Обновляем сообщение
            await query.edit_message_text(
                query.message.text + "\n\n❌ <b>Сигнал пропущен</b>",
                parse_mode="HTML"
            )
            
            logger.info(f"Signal {signal_id} skipped by user")
            
        except Exception as e:
            logger.error(f"Failed to save skip feedback: {e}")
            await query.edit_message_text(
                query.message.text + "\n\n⚠️ Ошибка сохранения действия",
                parse_mode="HTML"
            )

    async def _handle_details_action(self, query, signal_id: str) -> None:
        """Обработка действия 'Детали'."""
        
        try:
            # Получаем детальную информацию о сигнале
            details = await self._get_signal_details(signal_id)
            
            # Отправляем детали как новое сообщение
            await query.message.reply_text(
                f"📊 <b>Детали сигнала #{signal_id}</b>\n\n{details}",
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error(f"Failed to get signal details: {e}")
            await query.edit_message_text(
                query.message.text + "\n\n⚠️ Ошибка получения деталей",
                parse_mode="HTML"
            )

    async def _save_feedback(self, signal_id: str, action: str) -> None:
        """Сохранение обратной связи через HALv1 API."""
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{HAL_API_URL.rstrip('/')}/api/trading-feedback",
                    json={"signal_id": int(signal_id), "action": action},
                )
                response.raise_for_status()
                
            logger.info(f"Feedback saved successfully: signal_id={signal_id}, action={action}")
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            raise

    async def _get_signal_details(self, signal_id: str) -> str:
        """Получение детальной информации о сигнале."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{HAL_API_URL.rstrip('/')}/api/trading-alert/{signal_id}")
                response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("failed_to_fetch_signal_details", signal_id=signal_id, error=str(exc))
            return f"❌ Не удалось загрузить детали сигнала #{signal_id}"

        metadata = data.get("payload") or data
        created_at = data.get("created_at")
        metrics = metadata.get("metrics", {})
        risk = metadata.get("risk", {})

        lines = [
            f"🆔 ID сигнала: {signal_id}",
        ]
        if created_at:
            lines.append(f"📅 Время создания: {created_at}")
        if metrics:
            winrate = metrics.get("winrate_pct")
            profit_factor = metrics.get("profit_factor")
            if winrate is not None or profit_factor is not None:
                lines.append("📊 Метрики:")
                if winrate is not None:
                    lines.append(f"• Winrate: {winrate:.1f}%")
                if profit_factor is not None:
                    lines.append(f"• Profit factor: {profit_factor:.2f}")
        if risk:
            rr = risk.get("rr") or risk.get("adjusted_rr")
            if rr:
                lines.append(f"📐 Risk/Reward: {rr:.2f}")
            if risk.get("adjusted_tp"):
                lines.append(f"🎯 Реком. TP: {risk['adjusted_tp']:.4f}")
        return "\n".join(lines)


__all__ = ["TradingAlertHandler"]
