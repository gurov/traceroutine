from __future__ import annotations

from pathlib import Path

from .base import Adapter
from .claude_code import ClaudeCodeAdapter
from .messages import MessagesAdapter
from .otlp import OtlpAdapter

REGISTRY: dict[str, Adapter] = {a.name: a for a in (OtlpAdapter(), ClaudeCodeAdapter(), MessagesAdapter())}


def pick(path: Path, name: str | None = None) -> Adapter:
    if name:
        if name not in REGISTRY:
            raise SystemExit(f"неизвестный адаптер {name!r}; есть: {', '.join(REGISTRY)}")
        return REGISTRY[name]
    for adapter in REGISTRY.values():
        if adapter.detect(path):
            return adapter
    raise SystemExit(f"не удалось определить формат {path}; укажите --adapter")


__all__ = ["Adapter", "ClaudeCodeAdapter", "MessagesAdapter", "OtlpAdapter", "REGISTRY", "pick"]
