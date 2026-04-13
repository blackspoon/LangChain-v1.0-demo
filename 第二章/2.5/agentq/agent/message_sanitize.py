"""将消息 content 规范为字符串，兼容 DeepSeek 等要求 content 不能为块列表的接口。"""

from __future__ import annotations

import json
from typing import Any, List, Sequence

from langchain_core.messages import BaseMessage


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                text = getattr(block, "text", None)
                parts.append(str(text) if text is not None else str(block))
        return "\n".join(parts)
    return str(content)


def stringify_message(msg: BaseMessage) -> BaseMessage:
    if isinstance(msg.content, str):
        return msg
    return msg.model_copy(update={"content": stringify_content(msg.content)})


def stringify_dialog(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    return [stringify_message(m) for m in messages]
