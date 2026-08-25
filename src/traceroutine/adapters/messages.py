"""Адаптер транскриптов в формате чат-сообщений OpenAI.

Это самая распространённая форма, в которой агентные фреймворки выкладывают логи:
массив сессий, в каждой — список сообщений с `role`, `content` и `tool_calls`.
В таком виде опубликованы, например, открытые корпуса траекторий SWE-agent,
mini-swe-agent и OpenHands на Hugging Face.

Отличие от OTLP и Claude Code, ради которого адаптер и написан: **здесь нет ни
таймстемпов, ни счётчиков токенов по шагам**. Так выглядит почти любой публичный
датасет. Инструмент обязан от этого не ломаться, а честно переключаться в
структурный режим: варианты, циклы, конформанс — без денег.

Зато у таких корпусов есть то, чего нет в проде: **метка исхода**. `resolved`
говорит, решена ли задача. Это переводит вопрос из «что дорого» в «какое
поведение приводит к провалу» — а это уже про качество, а не только про счёт.
Метка кладётся в атрибут `traceroutine.outcome`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..model import RawSpan

# Синтетический шаг времени: порядок событий известен, абсолютное время — нет.
# Секунда на шаг делает последовательность однозначной и честно нулевой по
# длительности анализ не ломает.
STEP_S = 1.0


def _loads(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return v


def _sessions(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] == "[":
        yield from json.loads(text)
        return
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _tool_names(msg: dict) -> list[str]:
    calls = _loads(msg.get("tool_calls_json")) or msg.get("tool_calls") or []
    out = []
    for c in calls if isinstance(calls, list) else []:
        name = (c.get("function") or {}).get("name") or c.get("name")
        if name:
            out.append(str(name))
    return out


class MessagesAdapter:
    name = "messages"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() not in (".json", ".jsonl"):
            return False
        try:
            head = path.open(encoding="utf-8", errors="replace").read(8192)
        except OSError:
            return False
        return ('"messages_json"' in head
                or ('"messages"' in head and '"role"' in head
                    and "resourceSpans" not in head and '"parentUuid"' not in head))

    def read(self, path: Path) -> Iterator[RawSpan]:
        for i, sess in enumerate(_sessions(path)):
            yield from self._session(sess, i)

    def _session(self, sess: dict, i: int) -> Iterator[RawSpan]:
        case = str(sess.get("session_id") or sess.get("id") or sess.get("instance_id") or i)
        msgs = _loads(sess.get("messages_json")) or sess.get("messages") or []
        model = sess.get("recorded_model") or sess.get("model")
        gt = _loads(sess.get("ground_truth_meta_json")) or {}
        outcome = gt.get("resolved") if isinstance(gt, dict) else None
        if outcome is None:
            outcome = sess.get("resolved")
        common: dict[str, Any] = {
            "gen_ai.conversation.id": case,
            "traceroutine.task.id": case,
            "gen_ai.agent.name": sess.get("agent_framework") or "agent",
        }
        if outcome is not None:
            # Исход — свойство ВСЕГО прогона, поэтому висит на каждом его событии:
            # иначе не отфильтровать «покажи только провалившиеся траектории».
            common["traceroutine.outcome"] = "resolved" if outcome else "failed"

        clock = 0.0
        prev: str | None = None
        for msg in msgs if isinstance(msgs, list) else []:
            role = msg.get("role")
            if role != "assistant":
                continue
            tools = _tool_names(msg)
            span_id = f"{case}#{clock:.0f}"
            yield RawSpan(
                span_id=span_id,
                trace_id=case,
                parent_id=prev,
                name="chat",
                ts_start=clock,
                ts_end=clock + STEP_S,
                attrs={"gen_ai.operation.name": "chat",
                       "gen_ai.request.model": model, **common},
                resource_attrs={"service.name": sess.get("source_dataset") or "messages"},
            )
            prev = span_id
            clock += STEP_S
            for j, name in enumerate(tools):
                yield RawSpan(
                    span_id=f"{span_id}.{j}",
                    trace_id=case,
                    parent_id=span_id,
                    name=name,
                    ts_start=clock,
                    ts_end=clock + STEP_S,
                    attrs={"gen_ai.tool.name": name, **common},
                    resource_attrs={"service.name": sess.get("source_dataset") or "messages"},
                )
                clock += STEP_S
