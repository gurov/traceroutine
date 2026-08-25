"""Тесты слоя Abstract."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traceroutine.abstract import (SYSTEM, ActivityMap, Cluster, NoBackend, _parse, build,
                                canonical, stability)


# --- уровень 1: правила -----------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("tool:search(query='ceny 2026')", "tool:search"),
    ("tool:search(query='SLA policy')", "tool:search"),
    ("tool:read_document(id=4471)", "tool:read_document"),
    ("tool:retry_with_backoff(attempt=3)", "tool:retry_with_backoff"),
    ("tool:vector_search(k=8,index=docs)", "tool:vector_search"),
    ("tool:rerank[candidates=40]", "tool:rerank"),
    ("read_document_4471", "read_document"),
])
def test_rules_strip_arguments(raw, want):
    assert canonical(raw) == want


@pytest.mark.parametrize("raw", ["chat:claude-opus-5", "chat:claude-sonnet-5", "chat:gpt-4o"])
def test_model_is_a_resource_not_an_activity(raw):
    """Классический process mining разделяет активность («что») и ресурс («чем»).
    Модель — ресурс; информация не теряется, она в Event.resource."""
    assert canonical(raw) == "chat"


def test_rules_are_idempotent():
    for l in ["tool:search(q='x')", "chat:claude-opus-5", "plain"]:
        assert canonical(canonical(l)) == canonical(l)


# --- разбор ответа ----------------------------------------------------------

def test_parse_accepts_flat_mapping_shape():
    """РЕГРЕССИЯ: локальный бэкенд вернул {"mapping": ...} вместо {"activities": ...},
    _parse отдал пустоту, и слой молча деградировал до правил."""
    out = _parse(json.dumps({"mapping": {"a": "search", "b": "search", "c": "verify"}}))
    got = {c.name: sorted(c.labels) for c in out}
    assert got == {"search": ["a", "b"], "verify": ["c"]}


def test_parse_prefers_activities_shape():
    out = _parse(json.dumps({"activities": [{"name": "x", "labels": ["1", "2"]}]}))
    assert out == [Cluster("x", ["1", "2"])]


# --- сборка словаря ---------------------------------------------------------

class FakeBackend:
    name = "fake"

    def __init__(self, clusters, model="fake-1"):
        self.clusters, self.model, self.calls = clusters, model, 0

    def cluster(self, labels, target):
        self.calls += 1
        return self.clusters


def test_lost_labels_fall_back_to_rules():
    """Модель обязана покрыть все метки; потерянные откатываются на уровень 1,
    а не исчезают из лога."""
    be = FakeBackend([Cluster("retrieve", ["tool:search"])])
    m = build(["tool:search(q=1)", "tool:write_answer"], be, target=5)
    assert m.mapping["tool:write_answer"] == "tool:write_answer"
    assert m.coverage["missing_from_llm"] == 1


def test_invented_labels_are_dropped():
    be = FakeBackend([Cluster("retrieve", ["tool:search", "totally_made_up"])])
    m = build(["tool:search(q=1)"], be, target=5)
    assert "totally_made_up" not in m.mapping
    assert m.coverage["invented_by_llm"] == 1


def test_manual_overrides_survive_regeneration(tmp_path):
    """Словарь — артефакт пользователя. Ручная правка обязана пережить перегенерацию."""
    p = tmp_path / "m.yaml"
    first = build(["tool:search(q=1)"], FakeBackend([Cluster("retrieve", ["tool:search"])]))
    first.overrides["tool:search"] = "my_custom_name"
    first.save(p)

    loaded = ActivityMap.load(p)
    second = build(["tool:search(q=2)"],
                   FakeBackend([Cluster("something_else", ["tool:search"])]),
                   existing=loaded)
    assert second.overrides["tool:search"] == "my_custom_name"
    assert second.activity("tool:search(q=99)") == "my_custom_name"
    assert second.mapping["tool:search"] == "something_else"   # mapping перезаписан


def test_map_roundtrip(tmp_path):
    p = tmp_path / "m.yaml"
    a = ActivityMap(mapping={"x": "y"}, overrides={"q": "r"}, backend="b", generated="g")
    a.save(p)
    b = ActivityMap.load(p)
    assert (b.mapping, b.overrides, b.backend) == (a.mapping, a.overrides, a.backend)
    assert "Edit overrides, not mapping" in p.read_text(encoding="utf-8")


def test_no_backend_is_rules_only():
    m = build(["tool:search(q=1)", "tool:search(q=2)", "chat:claude-opus-5"], NoBackend())
    assert set(m.mapping.values()) == {"tool:search", "chat"}


def test_stability_is_one_for_deterministic_backend():
    be = FakeBackend([Cluster("a", ["tool:x"]), Cluster("b", ["tool:y"])])
    assert stability(["tool:x", "tool:y"], be, runs=2) == pytest.approx(1.0)


def test_stability_detects_disagreement():
    """Rand index инвариантен к именам кластеров — важно, модель вольна назвать иначе."""
    class Flaky:
        name = "flaky"
        model = "f"

        def __init__(self):
            self.n = 0

        def cluster(self, labels, target):
            self.n += 1
            return ([Cluster("same", ["a", "b"])] if self.n == 1
                    else [Cluster("x", ["a"]), Cluster("y", ["b"])])

    assert stability(["a", "b"], Flaky(), runs=2) == pytest.approx(0.0)


# --- форма запроса к Anthropic ---------------------------------------------
# Без ключа вживую не прогнать, поэтому проверяем то, что реально ломается:
# параметры запроса. budget_tokens и prefill на Opus 5 дают 400.

def test_anthropic_request_shape():
    from traceroutine.abstract import AnthropicBackend

    fake = MagicMock()
    # Метка должна быть достаточно отличимой: односимвольная проба встречается
    # внутри слов системного промпта и делает проверку кеш-префикса бессмысленной.
    LABEL = "tool:zzz_unique_probe"
    fake.content = [MagicMock(type="text",
                              text=json.dumps({"activities": [{"name": "a", "labels": [LABEL]}]}))]
    fake.usage = MagicMock(input_tokens=10, output_tokens=5,
                           cache_read_input_tokens=0, cache_creation_input_tokens=0)

    with patch("anthropic.Anthropic") as ctor:
        ctor.return_value.messages.create.return_value = fake
        be = AnthropicBackend()
        assert be.cluster([LABEL], 12) == [Cluster("a", [LABEL])]
        kw = ctor.return_value.messages.create.call_args.kwargs

    assert kw["model"] == "claude-opus-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in json.dumps(kw)          # 400 на Opus 5
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["max_tokens"] >= 8000                        # обрезка = потерянные метки
    assert kw["messages"][-1]["role"] == "user"            # prefill на Opus 5 запрещён
    # Кеш: стабильный префикс в system, переменная часть (метки) — после него.
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["system"][0]["text"] == SYSTEM
    assert LABEL in kw["messages"][0]["content"]
    assert LABEL not in kw["system"][0]["text"]


# --- форма запроса к Gemini -------------------------------------------------

def test_gemini_schema_strips_unsupported_keys():
    """responseSchema у Gemini — подмножество OpenAPI: additionalProperties даёт 400."""
    from traceroutine.abstract import _SCHEMA, _gemini_schema

    g = _gemini_schema(_SCHEMA)
    assert "additionalProperties" not in json.dumps(g)
    assert "additionalProperties" in json.dumps(_SCHEMA)      # оригинал не тронут
    assert g["propertyOrdering"] == ["activities"]
    item = g["properties"]["activities"]["items"]
    assert item["propertyOrdering"] == ["name", "labels"]
    assert "additionalProperties" not in item


def test_gemini_request_shape(monkeypatch):
    from traceroutine.abstract import SYSTEM, GeminiBackend

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = {}

    class FakeResp:
        def read(self):
            return json.dumps({
                "candidates": [{"finishReason": "STOP", "content": {
                    "parts": [{"text": json.dumps(
                        {"activities": [{"name": "retrieve", "labels": ["tool:search"]}]})}]}}],
                "usageMetadata": {"promptTokenCount": 400, "candidatesTokenCount": 200},
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data)
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    be = GeminiBackend("gemini-3.1-pro-preview")
    assert be.cluster(["tool:search"], 12) == [Cluster("retrieve", ["tool:search"])]

    assert "gemini-3.1-pro-preview:generateContent" in captured["url"]
    # Ключ уходит заголовком, а не в query string: URL попадает в логи и историю.
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert "test-key" not in captured["url"]
    cfg = captured["body"]["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert "additionalProperties" not in json.dumps(cfg["responseSchema"])
    assert captured["body"]["systemInstruction"]["parts"][0]["text"] == SYSTEM
    assert be.usage[-1]["input"] == 400


def test_gemini_truncated_response_is_an_error(monkeypatch):
    """MAX_TOKENS = обрезанный JSON = потерянные метки. Молчать нельзя."""
    from traceroutine.abstract import GeminiBackend

    monkeypatch.setenv("GEMINI_API_KEY", "k")

    class FakeResp:
        def read(self):
            return json.dumps({"candidates": [{"finishReason": "MAX_TOKENS"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda r, timeout=None: FakeResp())
    with pytest.raises(SystemExit, match="MAX_TOKENS"):
        GeminiBackend().cluster(["a"], 12)


# --- проверки качества словаря ----------------------------------------------

def _ev(label, cost):
    return {"activity_raw": label, "cost_usd": cost}


def test_audit_flags_merging_priced_step_with_free_ones():
    """Счётчик активностей такое не ловит: цель достигнута, атрибуция уничтожена."""
    from traceroutine.abstract import ActivityMap, audit

    amap = ActivityMap(mapping={"chat": "respond", "tool:write_answer": "respond"})
    warnings = audit(amap, [_ev("chat", 9.05), _ev("tool:write_answer", 0.0)])
    assert len(warnings) == 1
    assert "respond" in warnings[0] and "chat" in warnings[0]


def test_audit_respects_overrides():
    """РЕГРЕССИЯ: audit читал сырой mapping и ругался на уже починенное overrides —
    предупреждение, которое невозможно погасить, обесценивает все остальные."""
    from traceroutine.abstract import ActivityMap, audit

    amap = ActivityMap(mapping={"chat": "respond", "tool:write_answer": "respond"},
                       overrides={"chat": "generate"})
    assert audit(amap, [_ev("chat", 9.05), _ev("tool:write_answer", 0.0)]) == []


def test_audit_silent_when_activity_is_all_free():
    from traceroutine.abstract import ActivityMap, audit

    amap = ActivityMap(mapping={"tool:a": "x", "tool:b": "x"})
    assert audit(amap, [_ev("tool:a", 0.0), _ev("tool:b", 0.0)]) == []
