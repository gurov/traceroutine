"""Контракт адаптера намеренно минимален.

Длинный хвост адаптеров — главный моат проекта, поэтому порог написания нового
должен быть максимально низким: два метода и итератор.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from ..model import RawSpan


@runtime_checkable
class Adapter(Protocol):
    name: str

    def detect(self, path: Path) -> bool:
        """Быстрая проверка «это мой формат?» — по возможности без полного чтения."""
        ...

    def read(self, path: Path) -> Iterator[RawSpan]:
        ...
