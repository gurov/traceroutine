"""Адаптер OTLP/JSON (OpenTelemetry) с поддержкой GenAI semantic conventions.

Реальность, ради которой этот файл длиннее, чем хотелось бы:

* OTLP/JSON заворачивает каждое значение атрибута в тип: {"stringValue": "x"},
  {"intValue": "42"} — причём intValue приходит СТРОКОЙ (protobuf int64 → JSON string).
* GenAI semconv переименовывались на ходу: prompt_tokens → input_tokens,
  completion_tokens → output_tokens. В логах 2025-2026 встречается и то и другое,
  иногда в одном файле от разных инструментов.
* Кеш-токены не стандартизированы вообще: каждый вендор пишет по-своему —
  словарь псевдонимов вынесен в ../semconv.py, общий для всех адаптеров.
* Файл бывает как одним JSON-объектом, так и JSONL (по строке на батч экспорта).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..model import RawSpan
from ..semconv import extract

NANOS = 1_000_000_000

# OTLP status.code: 0 UNSET, 1 OK, 2 ERROR
_STATUS = {0: "ok", 1: "ok", 2: "error"}


def _unwrap(v: dict[str, Any]) -> Any:
    """{"intValue": "42"} → 42. Разворачивает и вложенные массивы/kvlist."""
    if not isinstance(v, dict):
        return v
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        try:
            return int(v["intValue"])          # приходит строкой из protobuf
        except (TypeError, ValueError):
            return None
    if "doubleValue" in v:
        return v["doubleValue"]
    if "boolValue" in v:
        return v["boolValue"]
    if "arrayValue" in v:
        return [_unwrap(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return _attrs(v["kvlistValue"].get("values", []))
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs(items: list[dict] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kv in items or []:
        k = kv.get("key")
        if k is not None:
            out[k] = _unwrap(kv.get("value", {}))
    return out

def _ts(v: Any) -> float | None:
    """startTimeUnixNano приходит строкой. Пустое/0 считаем отсутствующим."""
    if v in (None, "", 0, "0"):
        return None
    try:
        return int(v) / NANOS
    except (TypeError, ValueError):
        return None

def _iter_payloads(path: Path) -> Iterator[dict]:
    """Файл может быть одним JSON-объектом, массивом или JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] == "[":
        for obj in json.loads(text):
            yield obj
        return
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():          # JSONL
        line = line.strip()
        if line:
            yield json.loads(line)


class OtlpAdapter:
    name = "otlp"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() not in (".json", ".jsonl", ".ndjson"):
            return False
        try:
            with path.open(encoding="utf-8") as fh:
                head = fh.read(4096)
        except OSError:
            return False
        return "resourceSpans" in head or "resource_spans" in head

    def read(self, path: Path) -> Iterator[RawSpan]:
        for payload in _iter_payloads(path):
            rspans = payload.get("resourceSpans") or payload.get("resource_spans") or []
            for rs in rspans:
                res_attrs = _attrs((rs.get("resource") or {}).get("attributes"))
                sspans = rs.get("scopeSpans") or rs.get("scope_spans") or []
                for ss in sspans:
                    for span in ss.get("spans") or []:
                        ev = self._span(span, res_attrs)
                        if ev is not None:
                            yield ev

    def _span(self, span: dict, res_attrs: dict[str, Any]) -> RawSpan | None:
        start = _ts(span.get("startTimeUnixNano") or span.get("start_time_unix_nano"))
        if start is None:
            return None                      # спан без начала бесполезен для процесса
        attrs = _attrs(span.get("attributes"))
        status = span.get("status") or {}
        code = status.get("code", 0)
        if isinstance(code, str):            # некоторые экспортёры шлют "STATUS_CODE_ERROR"
            code = 2 if "ERROR" in code.upper() else 0
        parent = span.get("parentSpanId") or span.get("parent_span_id") or None
        return RawSpan(
            span_id=span.get("spanId") or span.get("span_id") or "",
            trace_id=span.get("traceId") or span.get("trace_id") or "",
            parent_id=parent or None,
            name=span.get("name", ""),
            ts_start=start,
            ts_end=_ts(span.get("endTimeUnixNano") or span.get("end_time_unix_nano")),
            attrs=attrs,
            status=_STATUS.get(code, "ok"),
            error_type=status.get("message") if code == 2 else None,
            resource_attrs=res_attrs,
        )

    # --- используется слоем normalize ---
    # Оставлено делегатом: OTLP-спаны уже несут semconv-атрибуты как есть, разбирать
    # тут нечего. Настоящая реализация — в semconv.extract, общая для всех адаптеров.
    extract = staticmethod(extract)
