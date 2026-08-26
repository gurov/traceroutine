"""Тесты вертикального среза.

Половина из них — регрессии на дефекты, найденные первым же прогоном на данных.
Это и есть смысл Ц1: модель данных врёт заметно, но только под нагрузкой.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from traceroutine.adapters.otlp import OtlpAdapter, _unwrap
from traceroutine.mine import mine
from traceroutine.model import RawSpan
from traceroutine.normalize import normalize
from traceroutine.pricing import Pricing
from traceroutine.store import read, write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def events(tmp_path_factory):
    d = tmp_path_factory.mktemp("am")
    traces, parquet = d / "t.json", d / "e.parquet"
    subprocess.run([sys.executable, str(ROOT / "examples/gen_otlp.py"), str(traces)],
                   check=True, capture_output=True)
    ad = OtlpAdapter()
    write(normalize(ad.read(traces), case_notion="trace", flatten="genai"), parquet)
    return read(parquet)


# --- OTLP-специфика ---------------------------------------------------------

def test_int_values_arrive_as_strings():
    """protobuf int64 сериализуется в JSON строкой — классический источник нулей."""
    assert _unwrap({"intValue": "1500"}) == 1500
    assert _unwrap({"stringValue": "x"}) == "x"
    assert _unwrap({"boolValue": False}) is False
    assert _unwrap({"doubleValue": 1.5}) == 1.5


def test_deprecated_token_attribute_names():
    """prompt_tokens/completion_tokens устарели, но живы в проде."""
    span = RawSpan("s", "t", None, "chat", 1.0, 2.0, attrs={
        "gen_ai.operation.name": "chat", "gen_ai.request.model": "claude-opus-5",
        "gen_ai.usage.prompt_tokens": 100, "gen_ai.usage.completion_tokens": 20,
    })
    f = OtlpAdapter.extract(span)
    assert (f["tokens_in"], f["tokens_out"]) == (100, 20)


# --- регрессии на найденные дефекты -----------------------------------------

def test_container_span_is_not_a_process_step():
    """РЕГРЕССИЯ: agent.run несёт gen_ai.agent.name и проходил фильтр как шаг,
    расщепляя один процесс на лишние варианты."""
    container = RawSpan("root", "t", None, "agent.run", 1.0, 9.0,
                        attrs={"gen_ai.agent.name": "a", "gen_ai.conversation.id": "s"})
    step = RawSpan("c1", "t", "root", "chat", 1.0, 2.0,
                   attrs={"gen_ai.operation.name": "chat"})
    out = list(normalize([container, step], flatten="genai"))
    assert [e.activity for e in out] == ["chat"]


def test_variant_order_is_deterministic(events):
    """РЕГРЕССИЯ: при совпадающих ts_start порядок «плавал» и множил варианты."""
    a = mine(events)
    b = mine(list(events))
    assert {v.seq for v in a.variants} == {v.seq for v in b.variants}


def test_cached_tokens_priced_separately():
    """РЕГРЕССИЯ: кеш, сложенный с input, завышает стоимость в разы."""
    p = Pricing()
    plain = p.cost("claude-opus-5", 1000, 0, 0)
    cached = p.cost("claude-opus-5", 0, 0, 1000)
    assert cached == pytest.approx(plain * p.cache_read)
    assert cached < plain


def test_tools_do_not_trigger_unknown_model_warning():
    """РЕГРЕССИЯ: имена инструментов уходили в таблицу цен и пугали пользователя."""
    p = Pricing()
    assert p.cost("search", 0, 0, 0) == 0.0
    assert not p.unknown


def test_pareto_sorted_by_per_case_cost():
    """РЕГРЕССИЯ: сортировка по суммарной стоимости отвечала не на тот вопрос."""
    from traceroutine.mine import Model, Variant
    m = Model(n_cases=110, total_cost=200.0)
    m.variants = [Variant(("cheap",), n=100, cost=100.0),
                  Variant(("pricey",), n=10, cost=100.0)]
    assert m.cost_concentration()[0][0].seq == ("pricey",)


# --- сквозные ---------------------------------------------------------------

def test_pipeline_produces_events(events):
    assert len(events) > 1000
    assert all(e["case_id"] for e in events)


def test_known_pathologies_are_found(events):
    """Генератор закладывает retry-петлю и дорогую эскалацию — продукт обязан их видеть."""
    m = mine(events)
    assert m.rework_rate > 0.2, "петля search→read→search не обнаружена"
    assert "chat:claude-opus-5" in m.loop_cost or any(
        "opus" in (v.seq and " ".join(v.seq)) for v in m.top_variants(5)
    ), "дорогой путь эскалации не попал в топ"
    assert m.total_cost > 0


def test_reports_render(events, tmp_path):
    from traceroutine.report import render_html, render_markdown
    m = mine(events)
    md, html_ = render_markdown(m), render_html(m)
    assert "```mermaid" in md and "flowchart TD" in md
    assert md.count("-->") >= 3
    assert html_.startswith("<!doctype html") and "<svg" in html_
    assert "cdn" not in html_.lower() and "http://" not in html_, "отчёт должен быть автономным"


def test_mermaid_labels_are_escaped():
    """Кавычки и скобки в имени активности ломают mermaid-разметку."""
    from traceroutine.mine import Model
    from traceroutine.report import render_markdown
    m = Model(n_cases=1, n_events=1, total_cost=1.0)
    m.nodes = {'tool:f(a="b")': {"n": 1, "cost": 1.0, "duration": 0.0, "errors": 0, "tokens": 1}}
    out = render_markdown(m)
    graph = out.split("```mermaid")[1].split("```")[0]
    assert '(' not in graph.split("flowchart TD")[1].split("classDef")[0].replace('(["', '').replace('"])', '')


# --- one-shot: путь по умолчанию --------------------------------------------
# Три команды и два промежуточных файла стояли между человеком и первой
# находкой. Эти тесты стерегут ровно то, что их больше нет.

def _run(*args, cwd=None):
    from typer.testing import CliRunner

    from traceroutine.cli import app
    return CliRunner().invoke(app, [str(a) for a in args])


def test_bare_command_does_the_whole_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traces = tmp_path / "t.json"
    subprocess.run([sys.executable, str(ROOT / "examples/gen_otlp.py"), str(traces)],
                   check=True, capture_output=True)
    res = _run("--from", traces, "-f", "md")
    assert res.exit_code == 0, res.output
    # Все три артефакта конвейера на месте: отчёт читают, остальные два нужны
    # следующим командам (`check`, `diff`).
    for name in ("events.parquet", "activity_map.yaml", "report.md"):
        assert (tmp_path / name).exists(), name
    assert "nothing leaves this machine" in res.output


def test_bare_command_is_quiet_about_plumbing(tmp_path, monkeypatch):
    """Служебные строки подкоманд глушатся, находки — нет. Первый экран должен
    быть находками, а не отчётом о проделанной работе."""
    monkeypatch.chdir(tmp_path)
    traces = tmp_path / "t.json"
    subprocess.run([sys.executable, str(ROOT / "examples/gen_otlp.py"), str(traces)],
                   check=True, capture_output=True)
    out = _run("--from", traces, "-f", "md").output
    assert "raw labels ->" not in out
    assert "flatten=" not in out
    assert "at list prices" in out


def test_missing_source_says_what_to_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("traceroutine.cli.DEFAULT_SOURCES", [tmp_path / "nowhere"])
    res = _run()
    assert res.exit_code == 2
    assert "--from" in res.output


def test_subcommands_still_work_and_stay_verbose(tmp_path, monkeypatch):
    """One-shot глушит вывод через модульный флаг — значит он обязан сниматься,
    иначе следующая команда в том же процессе онемеет."""
    monkeypatch.chdir(tmp_path)
    traces = tmp_path / "t.json"
    subprocess.run([sys.executable, str(ROOT / "examples/gen_otlp.py"), str(traces)],
                   check=True, capture_output=True)
    assert _run("--from", traces, "-f", "md").exit_code == 0
    out = _run("ingest", traces, "-o", tmp_path / "e2.parquet").output
    assert "flatten=" in out
