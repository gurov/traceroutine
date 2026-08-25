"""Тесты адаптера транскриптов Claude Code.

Почти каждый тест здесь — регрессия на дефект, который нашёлся ТОЛЬКО при
встрече с настоящими данными и не мог найтись на синтетическом генераторе:
его писал тот же человек, что и парсер, под ту же модель мира.
"""
from __future__ import annotations

import json

import pytest

from traceroutine.adapters import pick
from traceroutine.adapters.claude_code import ClaudeCodeAdapter
from traceroutine.normalize import normalize
from traceroutine.pricing import Pricing

SESSION = "s-1"


def usage(tin=2, tout=100, cread=10_000, cw_1h=0, cw_5m=0):
    return {"input_tokens": tin, "output_tokens": tout,
            "cache_read_input_tokens": cread,
            "cache_creation_input_tokens": cw_1h + cw_5m,
            "cache_creation": {"ephemeral_1h_input_tokens": cw_1h,
                               "ephemeral_5m_input_tokens": cw_5m}}


def rec(kind, ts, uuid, *, request=None, content=None, u=None, prompt=None, parent=None):
    d = {"type": kind, "uuid": uuid, "sessionId": SESSION,
         "timestamp": f"2026-08-25T10:00:{ts:02d}.000Z", "parentUuid": parent,
         "cwd": "/home/u/proj"}
    if request:
        d["requestId"] = request
    if prompt:
        d["promptId"] = prompt
    msg = {"role": kind, "content": content if content is not None else "hi"}
    if kind == "assistant":
        msg["model"] = "claude-opus-5"
        msg["usage"] = u if u is not None else usage()
    d["message"] = msg
    return d


def write(tmp_path, records, name="sess.jsonl"):
    p = tmp_path / "-home-u-proj" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def spans(tmp_path, records):
    return list(ClaudeCodeAdapter().read(write(tmp_path, records)))


# --- главный дефект: один ответ модели пишется несколькими записями ----------

def test_one_api_call_is_one_event_even_across_several_records(tmp_path):
    """Регрессия. Claude Code пишет по записи на блок контента — размышление,
    текст, вызов инструмента, — и повторяет `usage` в каждой. Без склейки по
    requestId стоимость завышается кратно: на корпусе из 61 сессии выходило
    $1550 вместо $736."""
    recs = [
        rec("user", 0, "u1", prompt="p1"),
        rec("assistant", 1, "a1", request="req-1", content=[{"type": "thinking"}]),
        rec("assistant", 2, "a2", request="req-1", content=[{"type": "text"}]),
        rec("assistant", 3, "a3", request="req-1", content=[{"type": "text"}]),
    ]
    chats = [s for s in spans(tmp_path, recs) if s.name == "chat"]
    assert len(chats) == 1
    assert chats[0].attrs["gen_ai.usage.output_tokens"] == 100


def test_usage_charged_once_when_records_are_split_by_tool_result(tmp_path):
    """Записи одного ответа не всегда идут подряд: между ними ложится tool_result
    (356 случаев на реальном корпусе). Структуру строим по позиции, но платим
    за вызов один раз."""
    recs = [
        rec("user", 0, "u1", prompt="p1"),
        rec("assistant", 1, "a1", request="req-1",
            content=[{"type": "tool_use", "id": "t1", "name": "Bash"}]),
        rec("user", 2, "u2", content=[{"type": "tool_result", "tool_use_id": "t1"}]),
        rec("assistant", 3, "a2", request="req-1", content=[{"type": "text"}]),
    ]
    chats = [s for s in spans(tmp_path, recs) if s.name == "chat"]
    assert len(chats) == 2                       # структура сохранена
    paid = [c for c in chats if c.attrs["gen_ai.usage.output_tokens"]]
    assert len(paid) == 1                        # а заплачено один раз


# --- case notion ------------------------------------------------------------

def test_task_case_notion_splits_session_into_user_requests(tmp_path):
    """Сессия как кейс не годится: это рабочий день, в нём десяток независимых
    задач, и двух одинаковых сессий не бывает — вариантный анализ вырождается."""
    recs = [
        rec("user", 0, "u1", prompt="p1"),
        rec("assistant", 1, "a1", request="r1"),
        rec("user", 2, "u2", prompt="p2"),
        rec("assistant", 3, "a2", request="r2"),
    ]
    sp = spans(tmp_path, recs)
    by_task = {e.case_id for e in normalize(sp, case_notion="task")}
    by_session = {e.case_id for e in normalize(sp, case_notion="session")}
    assert by_task == {"p1", "p2"}
    assert by_session == {SESSION}


def test_task_id_propagates_to_tool_calls(tmp_path):
    """promptId стоит только на user-записях — без переноса вперёд ходы модели
    и вызовы инструментов уехали бы в другой кейс."""
    recs = [
        rec("user", 0, "u1", prompt="p1"),
        rec("assistant", 1, "a1", request="r1",
            content=[{"type": "tool_use", "id": "t1", "name": "Bash"}]),
    ]
    evs = list(normalize(spans(tmp_path, recs), case_notion="task"))
    assert {e.case_id for e in evs} == {"p1"}
    assert any(e.activity_raw == "tool:Bash" for e in evs)


# --- деньги -----------------------------------------------------------------

def test_anthropic_input_tokens_are_not_reduced_by_cache(tmp_path):
    """У Anthropic input_tokens НЕ включает кеш. Прежняя эвристика «кеш больше
    входа — значит включён» обнуляла вход именно на этих данных."""
    recs = [rec("user", 0, "u1", prompt="p"),
            rec("assistant", 1, "a1", request="r1", u=usage(tin=2, cread=10_000))]
    e = next(e for e in normalize(spans(tmp_path, recs)) if e.activity_raw.startswith("chat"))
    assert e.tokens_in == 2
    assert e.tokens_cached == 10_000


def test_cache_write_is_priced_above_input_not_equal(tmp_path):
    """Запись в кеш дороже обычного input: 1.25× для 5 минут, 2× для часа.
    Игнорирование множителя занижало счёт на 12% на реальном корпусе."""
    recs = [rec("user", 0, "u1", prompt="p"),
            rec("assistant", 1, "a1", request="r1",
                u=usage(tin=0, tout=0, cread=0, cw_1h=1_000_000))]
    pr = Pricing()
    e = next(e for e in normalize(spans(tmp_path, recs), pricing=pr)
             if e.activity_raw.startswith("chat"))
    rate = pr.models["claude-opus-5"]["input"]
    assert e.tokens_cache_write == 1_000_000
    assert e.cost_usd == pytest.approx(rate * pr.cache_write_1h)
    assert e.cost_usd > rate                     # именно дороже, а не «как input»


def test_cache_write_falls_back_to_5m_when_ttl_unknown(tmp_path):
    """Разбивки по TTL нет ни у кого, кроме Anthropic. Без неё считаем дешёвый
    TTL: ошибиться в сторону занижения честнее, чем завысить счёт."""
    recs = [rec("user", 0, "u1", prompt="p"),
            rec("assistant", 1, "a1", request="r1",
                u={"input_tokens": 0, "output_tokens": 0,
                   "cache_read_input_tokens": 0,
                   "cache_creation_input_tokens": 1_000_000})]
    pr = Pricing()
    e = next(e for e in normalize(spans(tmp_path, recs), pricing=pr)
             if e.activity_raw.startswith("chat"))
    assert e.cost_usd == pytest.approx(pr.models["claude-opus-5"]["input"] * pr.cache_write_5m)


# --- структура и приватность ------------------------------------------------

def test_tool_failure_becomes_error_status(tmp_path):
    recs = [
        rec("user", 0, "u1", prompt="p"),
        rec("assistant", 1, "a1", request="r1",
            content=[{"type": "tool_use", "id": "t1", "name": "Bash"}]),
        rec("user", 2, "u2", content=[{"type": "tool_result", "tool_use_id": "t1",
                                       "is_error": True, "content": "boom"}]),
    ]
    tool = next(s for s in spans(tmp_path, recs) if s.name == "Bash")
    assert tool.status == "error"
    assert tool.error_type == "tool_error"


def test_tool_duration_spans_until_its_result(tmp_path):
    recs = [
        rec("user", 0, "u1", prompt="p"),
        rec("assistant", 1, "a1", request="r1",
            content=[{"type": "tool_use", "id": "t1", "name": "Bash"}]),
        rec("user", 9, "u2", content=[{"type": "tool_result", "tool_use_id": "t1"}]),
    ]
    tool = next(s for s in spans(tmp_path, recs) if s.name == "Bash")
    assert tool.ts_end - tool.ts_start == pytest.approx(8.0)


def test_no_message_content_leaks_into_spans(tmp_path):
    """Транскрипты содержат код и переписку, а отчётами делятся. Наружу отсюда
    выходят только имена, счётчики и время — проверяем это на входе, а не в отчёте."""
    secret = "SUPER_SECRET_TOKEN_42"
    recs = [
        rec("user", 0, "u1", prompt="p", content=[{"type": "text", "text": secret}]),
        rec("assistant", 1, "a1", request="r1", content=[
            {"type": "text", "text": secret},
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": f"echo {secret}"}}]),
        rec("user", 2, "u2", content=[{"type": "tool_result", "tool_use_id": "t1",
                                       "is_error": True, "content": secret}]),
    ]
    blob = json.dumps([{"name": s.name, "attrs": s.attrs, "res": s.resource_attrs,
                        "err": s.error_type} for s in spans(tmp_path, recs)])
    assert secret not in blob
    assert "/home/u/proj" not in blob            # путей тоже нет, только basename


def test_project_name_is_basename_only(tmp_path):
    recs = [rec("user", 0, "u1", prompt="p"), rec("assistant", 1, "a1", request="r1")]
    assert spans(tmp_path, recs)[0].attrs["gen_ai.agent.name"] == "proj"


def test_subagent_is_separated_from_parent(tmp_path):
    """Иначе стоимость субагента растворяется в родительской сессии."""
    r = rec("assistant", 1, "a1", request="r1")
    r["isSidechain"] = True
    sp = spans(tmp_path, [rec("user", 0, "u1", prompt="p"), r])
    assert sp[0].attrs["gen_ai.agent.name"] == "proj/subagent"


# --- обнаружение формата ----------------------------------------------------

def test_detected_without_explicit_flag(tmp_path):
    p = write(tmp_path, [rec("user", 0, "u1", prompt="p"),
                         rec("assistant", 1, "a1", request="r1")])
    assert pick(p).name == "claude-code"
    assert pick(p.parent).name == "claude-code"   # и по папке с транскриптами


def test_directory_reads_every_transcript(tmp_path):
    base = [rec("user", 0, "u1", prompt="p"), rec("assistant", 1, "a1", request="r1")]
    write(tmp_path, base, "one.jsonl")
    write(tmp_path, base, "two.jsonl")
    assert len(list(ClaudeCodeAdapter().read((tmp_path / "-home-u-proj")))) == 2


def test_truncated_last_line_does_not_crash(tmp_path):
    """Файл живой сессии читается на ходу и обрывается на полуслове."""
    p = write(tmp_path, [rec("user", 0, "u1", prompt="p"),
                         rec("assistant", 1, "a1", request="r1")])
    p.write_text(p.read_text(encoding="utf-8") + '\n{"type":"assist',
                 encoding="utf-8")
    assert len(list(ClaudeCodeAdapter().read(p))) == 1
