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



# --- пачки вызовов ----------------------------------------------------------
# Инструменты не вызывают друг друга: 97.5% переходов в логе кодового агента —
# это одно чередование «ход модели ↔ вызов». Оставшиеся 2.3% не цепочка, а
# ПАРАЛЛЕЛИЗМ: медиана зазора между ними 0.000 с против 7.3 с у «ход → вызов».
# Раньше это шло в rework и в циклы, то есть инструмент требовал чинить ровно то,
# что агент делает правильно.

def _turn(case, req, tools, ts=0.0, cost=1.0):
    """Ход модели, выдавший пачку вызовов."""
    out = [ev(case, "chat", cost=cost, ts=ts)]
    out[0]["event_id"] = req
    for i, t in enumerate(tools):
        e = ev(case, f"tool:{t}", ts=ts + 0.001 * (i + 1))
        e["parent_id"] = req
        out.append(e)
    return out


def test_parallel_calls_are_not_rework():
    """Четыре `Edit` из одного хода — один шаг с четырьмя целями."""
    from traceroutine.mine import mine
    evs = _turn("c1", "req1", ["Edit"] * 4, ts=0.0)
    m = mine(evs)
    assert m.rework_events == 0, "пачка засчитана как повтор работы"
    assert m.parallel_events == 4 and m.parallel_batches == 1
    assert m.turns_saved == 3


def test_the_same_tool_in_two_turns_is_rework():
    """А вот повтор ЧЕРЕЗ ход модели — по-прежнему повтор.

    Сравниваем с прогоном, где второй инструмент другой: разница и есть вклад
    самого инструмента. Абсолютное число тут не годится — повторный `chat` сам
    по себе тоже повтор, и он есть в обоих прогонах."""
    from traceroutine.mine import mine
    same = mine(_turn("c1", "req1", ["Edit"], ts=0.0) + _turn("c1", "req2", ["Edit"], ts=10.0))
    other = mine(_turn("c1", "req1", ["Edit"], ts=0.0) + _turn("c1", "req2", ["Read"], ts=10.0))
    assert same.rework_events - other.rework_events == 1
    assert same.turns_saved == 0


def test_parallel_calls_are_not_a_loop():
    """РЕГРЕССИЯ: `tool:Bash → tool:Bash` 143 раза в моём логе — это пачки, а не
    самая частая петля процесса."""
    from traceroutine.analyze import find_cycles
    evs = _turn("c1", "req1", ["Bash"] * 5, ts=0.0)
    assert [c for c in find_cycles(evs) if c.pattern == ("tool:Bash",)] == []


def test_a_flat_trace_is_not_one_giant_batch():
    """РЕГРЕССИЯ: у плоского OTLP-трейса все спаны висят на trace-спане, которого
    среди событий нет. Без проверки «родитель сам есть в логе» в пачку попадал
    весь кейс, и rework обнулялся на синтетике целиком."""
    from traceroutine.mine import batched_parents, mine
    evs = []
    for i, a in enumerate(["search", "read", "search", "read"]):
        e = ev("c1", f"tool:{a}", ts=float(i))
        e["parent_id"] = "trace-root-not-in-the-log"
        evs.append(e)
    assert batched_parents(evs) == set()
    assert mine(evs).rework_events == 2


# --- flame: дерево префиксов ------------------------------------------------
# Directly-follows граф у агента вырождается в звезду с хабом: топология задана
# конструкцией и ничего не сообщает. Дерево префиксов — линза, которая на этих
# данных ещё жива, и эти тесты стерегут именно её свойства.

def ev(case, act, cost=0.0, ts=0.0):
    return {"case_id": case, "event_id": f"{case}-{ts}-{act}", "activity_raw": act,
            "activity": act, "ts_start": ts, "ts_end": ts + 1, "parent_id": None,
            "agent": None, "resource": None, "tokens_in": 0, "tokens_cached": 0,
            "tokens_cache_write": 0, "tokens_out": 0, "cost_usd": cost, "status": "ok",
            "error_type": None, "attrs": "{}"}


def _model(paths):
    """paths: {последовательность: (сколько кейсов, стоимость всех этих кейсов)}"""
    from traceroutine.mine import Model, Variant
    m = Model()
    for seq, (n, cost) in paths.items():
        m.variants.append(Variant(seq=seq, n=n, cost=cost))
        m.n_cases += n
        m.total_cost += cost
        for a in seq:
            s = m.nodes.setdefault(a, {"n": 0, "cost": 0.0, "duration": 0.0,
                                       "errors": 0, "tokens": 0})
            s["n"] += n
    return m


def test_prefix_tree_children_partition_the_parent():
    """Ширина блока — доля счёта, ушедшая в эту ветку. Если дети не делят
    родителя ровно, картинка врёт масштабом, а не подписью."""
    from traceroutine.report import _prefix_tree
    m = _model({("a", "b"): (1, 10.0), ("a", "c"): (1, 30.0), ("d",): (1, 60.0)})
    root = _prefix_tree(m, 10)
    assert root["cost"] == pytest.approx(100.0)
    assert root["kids"]["a"]["cost"] == pytest.approx(40.0)
    assert sum(k["cost"] for k in root["kids"]["a"]["kids"].values()) == pytest.approx(40.0)


def test_prefix_tree_keeps_the_run_that_ends_early():
    """Кейс, закончившийся на префиксе, обязан остаться в ширине родителя:
    иначе ветка визуально сужается там, где прогоны просто завершились."""
    from traceroutine.report import _prefix_tree
    m = _model({("a",): (1, 25.0), ("a", "b"): (1, 75.0)})
    root = _prefix_tree(m, 10)
    assert root["kids"]["a"]["cost"] == pytest.approx(100.0)
    assert root["kids"]["a"]["kids"]["b"]["cost"] == pytest.approx(75.0)


def test_hub_is_detected_and_loses_its_colour():
    """Ход модели стоит на обоих концах почти каждого перехода. Отдать ему
    самый заметный цвет — отдать всю яркость картинки константе."""
    from traceroutine.mine import mine
    from traceroutine.report import _hub, _series
    evs = []
    for c in range(4):
        for i, a in enumerate(["chat", "tool:Bash", "chat", "tool:Read", "chat"]):
            evs.append(ev(f"c{c}", a, ts=float(i)))
    m = mine(evs)
    assert _hub(m) == "chat"
    colours = _series(m)
    assert colours["chat"] == "sh"
    assert {colours["tool:Bash"], colours["tool:Read"]} == {"s1", "s2"}


def test_no_hub_when_the_graph_is_a_chain():
    """Правило про хаб выключается само: у обычного пайплайна хаба нет, и три
    цвета должны достаться трём самым частым активностям."""
    from traceroutine.mine import mine
    from traceroutine.report import _hub, _series
    evs = [ev("c1", a, ts=float(i)) for i, a in enumerate(["a", "b", "c", "d"])]
    m = mine(evs)
    assert _hub(m) is None
    assert set(_series(m).values()) == {"s1", "s2", "s3"}


def test_flame_drops_slivers_thinner_than_a_pixel():
    """Хвост из сотен уникальных путей превращает картинку в шум шириной
    в волос. Блок тоньше порога уходит вместе с поддеревом."""
    from traceroutine.report import _flame_blocks, _prefix_tree
    paths = {("a", "b"): (1, 1000.0)}
    for i in range(500):                       # каждый — 0.002% ширины
        paths[("a", f"x{i}")] = (1, 0.02)
    blocks = _flame_blocks(_prefix_tree(_model(paths), 6), 6)
    assert [b for b in blocks if b[3] == "b"]
    assert not [b for b in blocks if b[3].startswith("x")]


def test_flame_survives_a_log_with_one_run():
    from traceroutine.report import _flame_svg
    svg, cap = _flame_svg(_model({("a", "b"): (1, 1.0)}))
    assert "<rect" in svg and cap


def test_html_report_carries_the_flame_and_the_carry_chart():
    from traceroutine.analyze import ContextCost
    from traceroutine.report import render_html
    m = _model({("chat", "tool:Bash", "chat"): (3, 30.0)})
    m.total_cost = 30.0
    out = render_html(m, inflation=[ContextCost(step="tool:Bash", n=3, added_avg=1000.0,
                                                carried_tokens=3e6, est_usd=12.0)])
    assert "Where the money goes" in out and "class=flame" in out
    assert "What a step costs after itself" in out
    assert "$0.00 at the call" in out, "смысл блока — шаг, который в разбивке бесплатен"


def test_carry_chart_leaves_the_hub_out():
    """У хаба «в момент вызова» стоит вся сумма счёта, и рядом с остальными
    строками его короткий столбик читается как «модель дешёвая»."""
    from traceroutine.analyze import ContextCost
    from traceroutine.mine import mine
    from traceroutine.report import _inflation_html
    evs = []
    for c in range(3):
        for i, a in enumerate(["chat", "tool:Bash", "chat"]):
            evs.append(ev(f"c{c}", a, cost=1.0 if a == "chat" else 0.0, ts=float(i)))
    m = mine(evs)
    out = _inflation_html(m, [ContextCost(step="chat", n=6, est_usd=9.0),
                              ContextCost(step="tool:Bash", n=3, est_usd=4.0)])
    assert "tool:Bash" in out and ">chat<" not in out


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
