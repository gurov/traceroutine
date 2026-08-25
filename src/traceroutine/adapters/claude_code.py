"""Адаптер транскриптов Claude Code (~/.claude/projects/**/*.jsonl).

Почему этот источник важен отдельно от OTLP: **инструментация не нужна вообще**.
Claude Code пишет свои сессии на диск сам, у любого пользователя они уже есть.
Ни экспортёра, ни коллектора, ни аккаунта в observability-платформе — а значит,
путь до первой ценности короче, чем у любого демо на выдуманных данных.

Формат — JSONL, по записи на строку. Значимы два типа:

* `assistant` — ход модели: `usage` со всеми четырьмя видами токенов, `model`,
  и блоки контента, среди которых `tool_use`. ВАЖНО: один ответ модели пишется
  НЕСКОЛЬКИМИ такими записями — по одной на блок контента (размышление, текст,
  вызов инструмента), и `usage` в них ПОВТОРЯЕТСЯ целиком. Склеивать их по
  `requestId` обязательно: без этого стоимость завышается вдвое (на корпусе из
  61 сессии — $1550 вместо $736);
* `user` — среди прочего несёт `tool_result` с `tool_use_id` и флагом `is_error`.

Соответствие процессной модели:

    case      = promptId — ОДИН запрос пользователя со всей работой по нему
                (`--case task`). Не сессия: сессия это рабочий день, в ней подряд
                идёт десяток независимых задач, и одинаковых сессий не бывает вовсе.
    activity  = ход модели (`chat`) либо вызов инструмента (`tool:<имя>`)
    timestamp = у хода модели — от предыдущей записи до его собственной (это и есть
                latency генерации); у инструмента — от хода модели до прихода
                tool_result

ПРИВАТНОСТЬ. Транскрипты содержат код, пути и переписку. Наружу отсюда не выходит
ничего из содержимого: ни текста сообщений, ни аргументов инструментов, ни путей к
файлам, ни команд. Только имена инструментов, модель, счётчики токенов и тайминги.
Единственное исключение — basename папки проекта как имя агента: он нужен, чтобы
различать прогоны, и не раскрывает ни структуры файлов, ни содержимого. Проверять
это надо здесь, на входе, а не в отчёте: отчётами делятся.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..model import RawSpan


def _ts(v: Any) -> float | None:
    """ISO-8601 с Z. Записи без времени бесполезны для процесса."""
    if not isinstance(v, str) or not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    return [path]


def _records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue          # обрезанный хвост живого файла — не повод падать


def _usage_attrs(usage: dict) -> dict[str, Any]:
    """usage → атрибуты semconv. Разбивку записи кеша по TTL даёт только Anthropic."""
    cc = usage.get("cache_creation") or {}
    tcw_5m = int(cc.get("ephemeral_5m_input_tokens") or 0)
    tcw_1h = int(cc.get("ephemeral_1h_input_tokens") or 0)
    total_cw = int(usage.get("cache_creation_input_tokens") or 0)
    # Разбивка бывает пустой, а суммарное поле — заполненным. Тогда весь объём
    # относим к пятиминутному TTL: это дефолт, и ошибка уходит в занижение цены.
    if not (tcw_5m or tcw_1h):
        tcw_5m = total_cw
    out: dict[str, Any] = {
        "gen_ai.usage.input_tokens": int(usage.get("input_tokens") or 0),
        "gen_ai.usage.output_tokens": int(usage.get("output_tokens") or 0),
        "gen_ai.usage.cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "anthropic.usage.cache_creation.ephemeral_5m_input_tokens": tcw_5m,
        "anthropic.usage.cache_creation.ephemeral_1h_input_tokens": tcw_1h,
        # У Anthropic input_tokens НЕ включает кеш — заявляем это явно, чтобы
        # normalize не гадал по величинам. См. semconv.input_includes_cache.
        "traceroutine.usage.input_includes_cache": False,
    }
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    if thinking:
        # Уже внутри output_tokens — НЕ добавлять к стоимости, только как признак.
        out["gen_ai.usage.reasoning_tokens"] = int(thinking)
    return out


class ClaudeCodeAdapter:
    name = "claude-code"

    def detect(self, path: Path) -> bool:
        candidates = _files(path)[:1] if path.is_dir() else [path]
        for p in candidates:
            if p.suffix.lower() != ".jsonl":
                return False
            try:
                head = p.open(encoding="utf-8", errors="replace").read(8192)
            except OSError:
                return False
            # sessionId + parentUuid вместе не встречаются больше нигде из наших форматов
            if '"sessionId"' in head and '"parentUuid"' in head:
                return True
        return False

    def read(self, path: Path) -> Iterator[RawSpan]:
        for f in _files(path):
            yield from self._file(f)

    def _file(self, path: Path) -> Iterator[RawSpan]:
        records = list(_records(path))

        # Проход 1: когда вернулся результат каждого вызова и не был ли он ошибкой.
        # Без этого у вызова инструмента нет длительности — а «сколько времени агент
        # провёл в Bash» это ровно то, что хочется знать.
        done: dict[str, tuple[float | None, bool]] = {}
        for rec in records:
            if rec.get("type") != "user":
                continue
            content = (rec.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            when = _ts(rec.get("timestamp"))
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid:
                        done[tid] = (when, bool(block.get("is_error")))

        # Имя папки проекта — это путь с заменёнными на дефис слэшами, разобрать его
        # обратно нельзя. Зато записи несут `cwd` целиком: берём оттуда basename и
        # только его — ни пути, ни имени пользователя наружу не уходит.
        project = next((Path(r["cwd"]).name for r in records if r.get("cwd")),
                       path.parent.name)
        prev_ts: float | None = None
        # promptId стоит только на user-записях; ходы модели его не несут. Тянем
        # последний увиденный вперёд — это и есть граница задачи. Без этого кейсом
        # остаётся сессия, а сессия для вариантного анализа не годится: двух
        # одинаковых рабочих дней не бывает, и каждый кейс становится своим вариантом.
        task: str | None = None

        # Проход 2: сами события. Ход модели — это ГРУППА записей с одним requestId,
        # а не отдельная запись; копим её и выпускаем, когда группа кончилась.
        group: list[dict] = []
        # Записи одного ответа не всегда идут подряд: между ними успевает лечь
        # tool_result (356 случаев на корпусе из 61 сессии). Поэтому структуру
        # строим по позиции, а usage начисляем ОДИН раз на requestId — иначе
        # фрагмент ответа второй раз оплачивает весь вызов.
        charged: set[str] = set()

        def flush(start: float | None) -> Iterator[RawSpan]:
            if not group:
                return
            first, last = group[0], group[-1]
            end = _ts(last.get("timestamp")) or start
            session = first.get("sessionId") or first.get("session_id") or path.stem
            span_id = first.get("requestId") or first.get("uuid") or ""
            msg = first.get("message") or {}
            # Субагент (Task) идёт в том же файле с isSidechain=true. Кейс тот же —
            # это часть той же работы, — но агент другой, иначе стоимость субагента
            # растворится в родительской сессии.
            side = any(r.get("isSidechain") for r in group)
            agent = f"{project}/subagent" if side else project

            blocks = [b for r in group
                      for b in ((r.get("message") or {}).get("content") or [])
                      if isinstance(b, dict)]
            attrs: dict[str, Any] = {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": msg.get("model"),
                "gen_ai.conversation.id": session,
                "traceroutine.task.id": first.get("_task") or session,
                "gen_ai.agent.name": agent,
                # usage одинаков во всех записях группы — берём один раз, иначе
                # получится двойной счёт.
                **_usage_attrs({} if span_id in charged else (msg.get("usage") or {})),
            }
            charged.add(span_id)
            if first.get("effort"):
                attrs["claude_code.effort"] = first["effort"]
            kinds = sorted({b.get("type") for b in blocks if b.get("type")})
            if kinds:
                # Ход из одного размышления без ответа — отдельное поведение,
                # признак пригодится слою abstract.
                attrs["claude_code.content_kinds"] = ",".join(kinds)

            yield RawSpan(
                span_id=span_id,
                trace_id=session,
                parent_id=first.get("parentUuid") or None,
                name="chat",
                ts_start=start if start and end and start <= end else (end or 0.0),
                ts_end=end,
                attrs=attrs,
                resource_attrs={"service.name": "claude-code"},
            )

            for block in blocks:
                if block.get("type") != "tool_use":
                    continue
                tid = block.get("id") or ""
                t_end, failed = done.get(tid, (None, False))
                yield RawSpan(
                    span_id=tid,
                    trace_id=session,
                    parent_id=span_id or None,
                    name=str(block.get("name") or "unknown"),
                    ts_start=end or 0.0,
                    ts_end=t_end,
                    attrs={
                        "gen_ai.tool.name": block.get("name"),
                        "gen_ai.conversation.id": session,
                        "traceroutine.task.id": first.get("_task") or session,
                        "gen_ai.agent.name": agent,
                    },
                    status="error" if failed else "ok",
                    # Текст ошибки — это вывод инструмента, то есть содержимое.
                    # Наружу отдаём только сам факт.
                    error_type="tool_error" if failed else None,
                    resource_attrs={"service.name": "claude-code"},
                )

        group_start: float | None = None
        for rec in records:
            ts = _ts(rec.get("timestamp"))
            key = rec.get("requestId") or rec.get("uuid")
            is_turn = rec.get("type") == "assistant" and ts is not None

            if is_turn and group and key == (group[0].get("requestId") or group[0].get("uuid")):
                group.append(rec)                      # тот же ответ модели, ещё блок
                continue

            yield from flush(group_start)
            if group:
                prev_ts = _ts(group[-1].get("timestamp")) or prev_ts
                group = []

            if is_turn:
                rec["_task"] = task
                group = [rec]
                group_start = prev_ts
            else:
                if rec.get("promptId"):
                    task = rec["promptId"]
                if ts:
                    prev_ts = ts

        yield from flush(group_start)
