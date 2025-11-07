"""Утилиты для извлечения расширенной структуры сообщений из Telegram API."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from telethon.tl.types import (
    Message,
    User,
    Channel,
    Chat,
    MessageReactions,
    MessageFwdHeader,
    MessageEntityUrl,
    MessageEntityTextUrl,
)


def detect_language(text: str) -> Optional[str]:
    """Простое определение языка по кириллице."""
    if not text:
        return None
    
    # Подсчитываем кириллические символы
    cyrillic_count = len(re.findall(r'[а-яё]', text.lower()))
    latin_count = len(re.findall(r'[a-z]', text.lower()))
    
    if cyrillic_count > latin_count:
        return "ru"
    elif latin_count > 0:
        return "en"
    else:
        return None


def extract_reactions(reactions: Optional[MessageReactions]) -> List[Union[str, Dict[str, Any]]]:
    """Извлекает реакции из сообщения."""
    if not reactions or not reactions.results:
        return []
    
    result = []
    for reaction in reactions.results:
        # Используем hasattr для проверки атрибутов
        if hasattr(reaction, 'reaction') and hasattr(reaction, 'count'):
            emoji = reaction.reaction
            count = reaction.count
            if count > 0:
                # Преобразуем emoji в строку для JSON сериализации
                emoji_str = str(emoji) if emoji else ""
                result.append({"emoji": emoji_str, "count": count})
    
    return result


def extract_attachments(message: Message) -> List[Dict[str, Any]]:
    """Извлекает вложения из сообщения."""
    attachments = []
    
    # Медиафайлы
    if message.photo:
        attachments.append({
            "type": "photo",
            "file": f"photo_{message.id}.jpg"
        })
    
    if message.document:
        file_ext = "bin"
        if message.document.mime_type:
            if "pdf" in message.document.mime_type:
                file_ext = "pdf"
            elif "image" in message.document.mime_type:
                file_ext = "jpg"
            elif "video" in message.document.mime_type:
                file_ext = "mp4"
            elif "audio" in message.document.mime_type:
                file_ext = "mp3"
            elif "text" in message.document.mime_type:
                file_ext = "txt"
        
        attachments.append({
            "type": "document",
            "file": f"doc_{message.id}.{file_ext}"
        })
    
    if message.video:
        attachments.append({
            "type": "video",
            "file": f"video_{message.id}.mp4"
        })
    
    if message.voice:
        attachments.append({
            "type": "voice",
            "file": f"voice_{message.id}.ogg"
        })
    
    if message.sticker:
        attachments.append({
            "type": "sticker",
            "file": f"sticker_{message.id}.webp"
        })
    
    if message.contact:
        attachments.append({
            "type": "contact",
            "file": f"contact_{message.id}.vcf"
        })
    
    if message.geo:
        attachments.append({
            "type": "location",
            "file": f"location_{message.id}.json"
        })
    
    if message.venue:
        attachments.append({
            "type": "venue",
            "file": f"venue_{message.id}.json"
        })
    
    if message.poll:
        attachments.append({
            "type": "poll",
            "file": f"poll_{message.id}.json"
        })
    
    if message.dice:
        attachments.append({
            "type": "dice",
            "file": f"dice_{message.id}.json"
        })
    
    # URL из entities
    if message.entities:
        for entity in message.entities:
            if isinstance(entity, (MessageEntityUrl, MessageEntityTextUrl)):
                if isinstance(entity, MessageEntityTextUrl):
                    url = entity.url
                else:
                    url = message.text[entity.offset:entity.offset + entity.length]
                
                attachments.append({
                    "type": "url",
                    "href": url
                })
    
    return attachments


def extract_forward_info(fwd_from: Optional[MessageFwdHeader]) -> Optional[Dict[str, Any]]:
    """Извлекает информацию о пересылке."""
    if not fwd_from:
        return None
    
    result = {}
    
    if fwd_from.from_id:
        if hasattr(fwd_from.from_id, 'user_id'):
            result["user_id"] = fwd_from.from_id.user_id
        elif hasattr(fwd_from.from_id, 'channel_id'):
            result["channel_id"] = fwd_from.from_id.channel_id
    
    if fwd_from.from_name:
        result["from_name"] = fwd_from.from_name
    
    if fwd_from.date:
        result["date"] = fwd_from.date.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    if fwd_from.post_author:
        result["post_author"] = fwd_from.post_author
    
    return result if result else None


def extract_user_info(user: Optional[User]) -> Optional[Dict[str, Any]]:
    """Извлекает информацию о пользователе."""
    if not user:
        return None
    
    display_name = ""
    if user.first_name:
        display_name += user.first_name
    if user.last_name:
        display_name += " " + user.last_name
    display_name = display_name.strip()
    
    return {
        "id": user.id,
        "username": user.username,
        "display": display_name or f"User {user.id}"
    }


def extract_chat_info(chat: Any) -> str:
    """Извлекает название чата."""
    if not chat:
        return "unknown"
    
    if hasattr(chat, 'title'):
        return chat.title
    elif hasattr(chat, 'first_name'):
        return chat.first_name
    else:
        return "unknown"


def extract_message_data(message: Message) -> Dict[str, Any]:
    """Извлекает расширенную структуру данных из сообщения Telegram."""
    
    # Информация об отправителе
    from_info = None
    if message.from_id:
        if hasattr(message.from_id, 'user_id'):
            # Прямой ID пользователя
            user_id = message.from_id.user_id
            username = None
            display_name = f"User {user_id}"
            
            # Пытаемся получить дополнительную информацию из sender
            if hasattr(message, 'sender') and message.sender:
                if isinstance(message.sender, User):
                    username = message.sender.username
                    display_name = ""
                    if message.sender.first_name:
                        display_name += message.sender.first_name
                    if message.sender.last_name:
                        display_name += " " + message.sender.last_name
                    display_name = display_name.strip() or f"User {user_id}"
            
            from_info = {
                "id": user_id,
                "username": username,
                "display": display_name
            }
        else:
            # Канал или группа
            from_info = {
                "id": getattr(message.from_id, 'channel_id', getattr(message.from_id, 'chat_id', None)),
                "username": None,
                "display": f"Channel {getattr(message.from_id, 'channel_id', 'Unknown')}"
            }
    
    # Структура в соответствии с желаемым форматом
    data = {
        "id": str(message.id),
        "date_utc": message.date.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "from": from_info,
        "text": message.text or "",
        "reply_to": str(message.reply_to.reply_to_msg_id) if message.reply_to else None,
        "reactions": extract_reactions(message.reactions),
        "attachments": extract_attachments(message),
        "forwarded_from": extract_forward_info(message.fwd_from),
        "language": detect_language(message.text) if message.text else None,
        "chat": extract_chat_info(getattr(message, 'chat', None))
    }
    
    # Дата редактирования (если есть)
    if message.edit_date:
        data["edited_utc"] = message.edit_date.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    return data
