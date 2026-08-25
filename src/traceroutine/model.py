"""Канонический Event — единственная структура, которую видят все слои после Ingest."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Порядок колонок фиксирован: на него опираются store и mine.
COLUMNS = [
    "case_id", "event_id", "activity_raw", "activity",
    "ts_start", "ts_end", "parent_id", "agent", "resource",
    "tokens_in", "tokens_cached", "tokens_cache_write", "tokens_out", "cost_usd",
    "status", "error_type", "attrs",
]


@dataclass(slots=True)
class RawSpan:
    """То, что отдаёт адаптер. Ещё не событие: case_id не выбран, стоимость не посчитана."""
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    ts_start: float          # unix seconds, float
    ts_end: float | None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"       # ok | error | timeout
    error_type: str | None = None
    resource_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Event:
    """Канонический элемент event log."""
    case_id: str
    event_id: str
    activity_raw: str
    activity: str            # до слоя Abstract равен activity_raw
    ts_start: float
    ts_end: float | None
    parent_id: str | None
    agent: str | None
    resource: str | None
    tokens_in: int
    tokens_cached: int       # ОТДЕЛЬНО от tokens_in: чтение кеша стоит кратно дешевле
    tokens_cache_write: int  # запись в кеш — НАОБОРОТ дороже input (1.25× или 2×)
    tokens_out: int
    cost_usd: float
    status: str
    error_type: str | None
    attrs: dict[str, Any]

    def as_row(self) -> tuple:
        d = asdict(self)
        return tuple(d[c] for c in COLUMNS)

    @property
    def duration_s(self) -> float:
        return (self.ts_end - self.ts_start) if self.ts_end else 0.0
