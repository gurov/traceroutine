"""Синтетический OTLP-лог агента: RAG-агент с реалистичными патологиями.

Нужен не только для тестов — это будущий публичный демо-датасет (Ц5):
воспроизводимый одной командой, без чужих данных.

Заложенные патологии — то, что продукт обязан находить:
  * retry-петля search→read→search, съедающая непропорционально много денег
  * редкий, но очень дорогой путь эскалации на большую модель
  * растущий кеш на happy path (проверка раздельного учёта кеша)
  * разнообразие имён инструментов и аргументов — чтобы слой Abstract (Ц2) было
    на чём показать: без него граф превращается в кашу из десятков активностей
"""
from __future__ import annotations

import json
import random
import sys
import uuid

NANOS = 1_000_000_000


def _kv(d):
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            out.append({"key": k, "value": {"boolValue": v}})
        elif isinstance(v, int):
            out.append({"key": k, "value": {"intValue": str(v)}})   # protobuf int64 → строка
        elif isinstance(v, float):
            out.append({"key": k, "value": {"doubleValue": v}})
        else:
            out.append({"key": k, "value": {"stringValue": str(v)}})
    return out


# Разнообразие имён и аргументов: правила снимут аргументы, а слить разные имена
# в один шаг процесса может только слой Abstract.
PLAN = ["plan_steps", "decompose_task", "pick_tool"]
SEARCH = ["search(query='ceny dodavatelu')", "search(query='SLA policy')",
          "vector_search(k=8,index=docs)", "vector_search(k=20,index=tickets)",
          "bm25_lookup(corpus=kb)", "rerank(candidates=40)"]
READ = ["read_document(id=4471)", "read_document(id=9902)",
        "fetch_url(https://intranet/policy)", "get_ticket(id=SUP-2231)"]
MEMORY = ["memory_read(key=user_prefs)", "memory_write(key=summary)", "load_context(session)"]
VERIFY = ["validate_answer", "self_critique", "check_citations"]
RESPOND = ["write_answer", "format_response(md)", "emit_final"]


def main(n_cases=400, seed=7, drift=0.0):
    """drift>0 имитирует смену модели: агент чаще зацикливается и чаще эскалирует.
    Нужен, чтобы было на чём показать `traceroutine diff`."""
    rnd = random.Random(seed)
    spans, t = [], 1_768_000_000.0

    for _ in range(n_cases):
        trace = uuid.uuid4().hex
        session = f"sess-{rnd.randint(1, 60)}"
        root = uuid.uuid4().hex[:16]
        t += rnd.uniform(20, 90)
        case_t = t
        steps = []

        steps.append(("tool", rnd.choice(PLAN), 0, 0, 0))
        steps.append(("chat", "claude-sonnet-5", 1800, 0, 220))
        w = [55 - 40 * drift, 25, 15 + 20 * drift, 5 + 20 * drift]     # хвост дорогих петель
        loops = rnd.choices([0, 1, 2, 5], weights=w)[0]
        for i in range(1 + loops):
            steps.append(("tool", rnd.choice(SEARCH), 0, 0, 0))
            steps.append(("tool", rnd.choice(READ), 0, 0, 0))
            if rnd.random() < 0.3:
                steps.append(("tool", rnd.choice(MEMORY), 0, 0, 0))
            cached = 1500 if i else 0
            steps.append(("chat", "claude-sonnet-5", 900 + 400 * i, cached, 300))
            if rnd.random() < 0.4:
                steps.append(("tool", rnd.choice(VERIFY), 0, 0, 0))
        if rnd.random() < 0.08 + 0.10 * drift:        # редкая дорогая эскалация
            steps.append(("tool", "escalate", 0, 0, 0))
            steps.append(("chat", "claude-opus-5", 12000, 0, 2400))
        steps.append(("tool", rnd.choice(RESPOND), 0, 0, 0))

        for kind, name, tin, tcached, tout in steps:
            dur = rnd.uniform(0.3, 2.5) if kind == "tool" else rnd.uniform(1.5, 9.0)
            attrs = {"gen_ai.conversation.id": session}
            if kind == "tool":
                attrs |= {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
            else:
                attrs |= {
                    "gen_ai.operation.name": "chat", "gen_ai.system": "anthropic",
                    "gen_ai.request.model": name, "gen_ai.response.model": name,
                    "gen_ai.usage.input_tokens": tin,
                    "gen_ai.usage.output_tokens": tout,
                }
                if tcached:
                    attrs["gen_ai.usage.cache_read_input_tokens"] = tcached
            err = kind == "tool" and name == "search" and rnd.random() < 0.06
            spans.append({
                "traceId": trace, "spanId": uuid.uuid4().hex[:16], "parentSpanId": root,
                "name": f"{attrs['gen_ai.operation.name']} {name}",
                "kind": 3,
                "startTimeUnixNano": str(int(t * NANOS)),
                "endTimeUnixNano": str(int((t + dur) * NANOS)),
                "attributes": _kv(attrs),
                "status": {"code": 2, "message": "rate_limited"} if err else {"code": 1},
            })
            t += dur

        spans.append({                                 # корневой спан без GenAI-семантики
            "traceId": trace, "spanId": root, "name": "agent.run", "kind": 1,
            "startTimeUnixNano": str(int(case_t * NANOS)),
            "endTimeUnixNano": str(int(t * NANOS)),
            "attributes": _kv({"gen_ai.agent.name": "support-agent",
                               "gen_ai.conversation.id": session}),
            "status": {"code": 1},
        })

    payload = {"resourceSpans": [{
        "resource": {"attributes": _kv({"service.name": "support-agent"})},
        "scopeSpans": [{"scope": {"name": "demo"}, "spans": spans}],
    }]}
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/traces.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"{len(spans):,} спанов, {n_cases} кейсов → {out}")


if __name__ == "__main__":
    main(drift=float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
