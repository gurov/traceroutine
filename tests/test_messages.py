"""Тесты адаптера чат-транскриптов (публичные корпуса траекторий)."""
from __future__ import annotations

import json

from traceroutine.adapters import pick
from traceroutine.adapters.messages import MessagesAdapter
from traceroutine.mine import mine
from traceroutine.model import COLUMNS
from traceroutine.normalize import normalize


def rows(events):
    """mine работает со словарями, как их отдаёт store.read."""
    return [dict(zip(COLUMNS, e.as_row())) for e in events]


def sess(sid, turns, *, resolved=None, framework="swe-agent", model="m1"):
    """turns — список списков имён инструментов на каждый ход модели."""
    msgs = [{"role": "system", "content": ""}]
    for tools in turns:
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls_json": json.dumps(
                         [{"function": {"name": t, "arguments": "{}"}} for t in tools])
                     if tools else None})
        msgs.append({"role": "user", "content": ""})
    d = {"session_id": sid, "agent_framework": framework, "recorded_model": model,
         "messages_json": json.dumps(msgs)}
    if resolved is not None:
        d["ground_truth_meta_json"] = json.dumps({"resolved": resolved})
    return d


def write(tmp_path, sessions, name="traj.json"):
    p = tmp_path / name
    p.write_text(json.dumps(sessions), encoding="utf-8")
    return p


def test_reads_turns_and_tool_calls(tmp_path):
    p = write(tmp_path, [sess("s1", [["grep"], ["python"]])])
    acts = [e.activity_raw for e in normalize(MessagesAdapter().read(p))]
    assert acts == ["chat:m1", "tool:grep", "chat:m1", "tool:python"]


def test_works_without_any_token_counts(tmp_path):
    """Так выглядит почти любой публичный датасет: ни таймстемпов, ни токенов.
    Инструмент обязан переключиться в структурный режим, а не сломаться."""
    p = write(tmp_path, [sess("s1", [["grep"]]), sess("s2", [["grep"]])])
    m = mine(rows(normalize(MessagesAdapter().read(p))))
    assert m.total_cost == 0.0
    assert m.n_cases == 2 and len(m.variants) == 1     # структура при этом видна


def test_outcome_label_lands_on_every_event_of_the_case(tmp_path):
    """Исход — свойство прогона; без него не отфильтровать провалившиеся траектории."""
    p = write(tmp_path, [sess("ok", [["grep"]], resolved=True),
                         sess("bad", [["grep"]], resolved=False)])
    got = {e.case_id: e.attrs.get("traceroutine.outcome")
           for e in normalize(MessagesAdapter().read(p))}
    assert got == {"ok": "resolved", "bad": "failed"}


def test_ordering_is_stable_without_timestamps(tmp_path):
    """Абсолютного времени нет, но порядок известен — синтетические часы не должны
    давать одинаковых меток, иначе сортировка событий станет недетерминированной."""
    p = write(tmp_path, [sess("s1", [["a", "b"], ["c"]])])
    evs = sorted(normalize(MessagesAdapter().read(p)), key=lambda e: e.ts_start)
    assert len({e.ts_start for e in evs}) == len(evs)


def test_detects_format_and_does_not_grab_other_formats(tmp_path):
    p = write(tmp_path, [sess("s1", [["grep"]])])
    assert pick(p).name == "messages"
    otlp = tmp_path / "o.json"
    otlp.write_text(json.dumps({"resourceSpans": []}), encoding="utf-8")
    assert pick(otlp).name == "otlp"


def test_jsonl_form_is_accepted(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(sess(f"s{i}", [["grep"]])) for i in range(3)),
                 encoding="utf-8")
    assert len({e.case_id for e in normalize(MessagesAdapter().read(p))}) == 3
