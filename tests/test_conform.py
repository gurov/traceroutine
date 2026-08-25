"""Тесты слоя Conformance."""
from __future__ import annotations

import pytest
import yaml

from agentmine.conform import (ANY, CheckReport, ConfigError, Process, Rule, align,
                               check, check_rule, findings, parse_flow)


def ev(case, act, cost=0.0, ts=0.0, status="ok"):
    return {"case_id": case, "event_id": f"{case}-{ts}", "activity_raw": act,
            "activity": act, "ts_start": ts, "ts_end": ts + 1, "parent_id": None,
            "agent": None, "resource": None, "tokens_in": 0, "tokens_cached": 0,
            "tokens_cache_write": 0, "tokens_out": 0, "cost_usd": cost, "status": status,
            "error_type": None, "attrs": "{}"}


def seq(case, acts, cost=1.0):
    return [ev(case, a, cost=cost, ts=float(i)) for i, a in enumerate(acts)]


def proc(**kw) -> Process:
    return Process.from_dict(kw)


# --- разбор flow ------------------------------------------------------------

def test_sequence_and_quantifiers():
    n = parse_flow("plan -> gather* -> (act -> verify)+ -> respond")
    assert n.alphabet() == {"plan", "gather", "act", "verify", "respond"}
    # Кратчайший прогон: gather* пропускается, (act→verify)+ берётся один раз.
    assert n.shortest_run() == 4


def test_alternation_and_optional():
    n = parse_flow("a -> (b | c)? -> d")
    assert align(n, ["a", "d"])[0] == 0
    assert align(n, ["a", "b", "d"])[0] == 0
    assert align(n, ["a", "c", "d"])[0] == 0
    assert align(n, ["a", "b", "c", "d"])[0] == 1


def test_any_is_a_wildcard():
    """`plan -> any* -> respond` — самое частое реальное ожидание от агента:
    середина свободна, края обязательны."""
    n = parse_flow("plan -> any* -> respond")
    assert align(n, ["plan", "чтоугодно", "и_ещё", "respond"])[0] == 0
    assert align(n, ["чтоугодно"])[0] == 2
    assert ANY not in n.alphabet()          # джокер не активность, в отчёте его быть не должно


def test_names_with_colons_and_dots():
    """В логе активности выглядят как `tool:Bash` и `gen_ai.chat` — парсер обязан их брать."""
    n = parse_flow("chat -> tool:Bash -> gen_ai.chat")
    assert n.alphabet() == {"chat", "tool:Bash", "gen_ai.chat"}


@pytest.mark.parametrize("bad", ["a -> (b", "a -> -> b", "(", "a) -> b", "a -> *"])
def test_broken_flow_is_rejected(bad):
    with pytest.raises(ConfigError):
        parse_flow(bad)


# --- выравнивание -----------------------------------------------------------

def test_perfect_trace_costs_nothing():
    n = parse_flow("plan -> act -> respond")
    cost, moves = align(n, ["plan", "act", "respond"])
    assert cost == 0
    assert all(m.kind == "sync" for m in moves)


def test_missing_step_is_a_model_move():
    n = parse_flow("plan -> act -> verify -> respond")
    cost, moves = align(n, ["plan", "act", "respond"])
    assert cost == 1
    assert [(m.kind, m.activity) for m in moves if m.kind != "sync"] == [("model", "verify")]


def test_extra_step_is_a_log_move():
    n = parse_flow("plan -> respond")
    cost, moves = align(n, ["plan", "лишнее", "respond"])
    assert cost == 1
    assert [(m.kind, m.activity) for m in moves if m.kind != "sync"] == [("log", "лишнее")]


def test_empty_trace_needs_the_whole_model():
    n = parse_flow("a -> b -> c")
    assert align(n, [])[0] == 3


def test_log_move_index_points_at_the_real_event():
    """Индекс нужен, чтобы приписать отклонению стоимость ИМЕННО того события."""
    n = parse_flow("a -> c")
    _cost, moves = align(n, ["a", "b", "c"])
    log = [m for m in moves if m.kind == "log"]
    assert [(m.activity, m.index) for m in log] == [("b", 1)]


# --- правила ----------------------------------------------------------------

@pytest.mark.parametrize("rule,trace,expected", [
    (Rule("always", "respond"), ["plan", "respond"], 0),
    (Rule("always", "respond"), ["plan"], 1),
    (Rule("never", "drop_db"), ["plan", "drop_db", "drop_db"], 2),
    (Rule("never", "drop_db"), ["plan"], 0),
    (Rule("first", "plan"), ["plan", "act"], 0),
    (Rule("first", "plan"), ["act", "plan"], 1),
    (Rule("last", "respond"), ["act", "respond"], 0),
    (Rule("last", "respond"), ["respond", "act"], 1),
    (Rule("after", "act", "verify"), ["act", "verify"], 0),
    (Rule("after", "act", "verify"), ["act", "act", "verify"], 0),
    (Rule("after", "act", "verify"), ["verify", "act"], 1),
    (Rule("before", "act", "plan"), ["plan", "act"], 0),
    (Rule("before", "act", "plan"), ["act", "plan"], 1),
    (Rule("forbid", "act", "act"), ["act", "act", "verify"], 1),
    (Rule("forbid", "act", "act"), ["act", "verify", "act"], 0),
    (Rule("max", "search", n=2), ["search"] * 5, 3),
    (Rule("max", "search", n=2), ["search"] * 2, 0),
])
def test_rule_semantics(rule, trace, expected):
    n, _cost = check_rule(rule, trace, [1.0] * len(trace))
    assert n == expected


def test_only_extra_steps_carry_money():
    """Деньги приписываются УЗКО. `max 2 of search` при пяти поисках стоит три
    поиска, а не весь прогон — на переусердствовании здесь уже обжигалась
    атрибуция сбоев."""
    n, cost = check_rule(Rule("max", "search", n=2), ["search"] * 5, [1.0] * 5)
    assert (n, cost) == (3, 3.0)


def test_absent_step_gets_no_money():
    """`after act expect verify` нарушено тем, чего НЕ произошло. Приписать этому
    стоимость нечего, и выдумывать её нельзя."""
    _n, cost = check_rule(Rule("after", "act", "verify"), ["act"], [7.0])
    assert cost == 0.0


# --- разбор конфига ---------------------------------------------------------

def test_pair_forms_parse():
    p = proc(rules=[{"after": "act", "expect": "verify"},
                    {"before": "act", "expect": "plan"},
                    {"max": 3, "of": "search"}])
    assert [(r.kind, r.a, r.b, r.n) for r in p.rules] == [
        ("after", "act", "verify", 0), ("before", "act", "plan", 0), ("max", "search", "", 3)]


def test_after_forbid_is_one_rule_not_two():
    """РЕГРЕССИЯ: `{after: A, forbid: B}` содержит два слова из словаря видов.
    Определение вида «по единственному известному ключу» на этой форме падало."""
    p = proc(rules=[{"after": "respond", "forbid": "search"}])
    assert (p.rules[0].kind, p.rules[0].a, p.rules[0].b) == ("forbid", "respond", "search")


def test_unknown_top_level_key_is_rejected():
    """Опечатка в ключе иначе молча отключает проверку, а CI остаётся зелёным.
    Зелёный CI, который ничего не проверяет, — худший возможный исход."""
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        proc(flow="a", rulez=[{"always": "a"}])


def test_unknown_threshold_is_rejected():
    with pytest.raises(ConfigError, match="thresholds"):
        proc(flow="a", thresholds={"fitnes_min": 0.9})


def test_empty_process_is_rejected():
    with pytest.raises(ConfigError, match="пуст"):
        proc(name="ничего")


def test_unreadable_rule_is_rejected():
    with pytest.raises(ConfigError, match="формы"):
        proc(rules=[{"always": "a", "never": "b"}])
    with pytest.raises(ConfigError):
        proc(rules=["always: a"])


def test_shipped_example_is_valid():
    """Пример из репозитория — часть контракта: он же первое, что скопируют."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples" / "process.yaml"
    p = Process.load(path)
    assert p.nfa is not None and p.rules and p.thresholds.fitness_min


# --- check() ----------------------------------------------------------------

def log(*traces, cost=1.0):
    out = []
    for i, t in enumerate(traces):
        out += seq(f"c{i}", t, cost=cost)
    return out


def test_fitness_and_conforming_share():
    p = proc(flow="plan -> act -> respond")
    rep = check(p, log(["plan", "act", "respond"], ["plan", "respond"]))
    assert rep.fitting == 1 and rep.n_cases == 2
    # Формула pm4py: 1 − cost / (|трасса| + |кратчайший прогон модели|).
    assert rep.fitness == pytest.approx(1 - 1 / ((3 + 3) + (2 + 3)))


def test_off_model_cost_is_the_cost_of_the_extra_steps():
    """Заголовочное число conformance — не оценка, а сумма стоимости шагов,
    которых в дизайне нет."""
    evs = seq("c1", ["plan", "лишнее", "respond"], cost=2.0)
    rep = check(proc(flow="plan -> respond"), evs)
    assert rep.off_model_cost == pytest.approx(2.0)
    assert rep.off_model_share == pytest.approx(2.0 / 6.0)


def test_missing_step_costs_nothing():
    rep = check(proc(flow="plan -> verify -> respond"), seq("c1", ["plan", "respond"], cost=5.0))
    assert rep.off_model_cost == 0.0
    assert [(d.kind, d.activity) for d in rep.deviations] == [("model", "verify")]


def test_deviation_counts_events_and_cases_separately():
    p = proc(flow="a -> b")
    rep = check(p, log(["a", "x", "x", "b"], ["a", "x", "b"]))
    d = next(d for d in rep.deviations if d.activity == "x")
    assert (d.n, d.cases) == (3, 2)


def test_rules_run_without_flow():
    """Декларативная часть самодостаточна: для агента она важнее императивной,
    и требовать flow ради неё нельзя."""
    rep = check(proc(rules=[{"always": "respond"}]), log(["plan"], ["plan", "respond"]))
    assert rep.fitness is None
    assert rep.rules[0].cases == 1


def test_unseen_activity_warns():
    """Активность объявлена, но в логе её нет: почти всегда опечатка или
    несовпадение уровня абстракции — и тогда проверка зелёная просто потому,
    что ей не с чем сравнивать."""
    rep = check(proc(flow="plan -> respond", rules=[{"never": "опечатка"}]),
                seq("c1", ["plan", "respond"]))
    assert rep.unseen == ["опечатка"]
    assert any("ни разу не встретились" in w for w in rep.warnings)


def test_p95_is_not_the_mean():
    """Средняя длина прячет ровно то, ради чего порог и ставят, — убежавшие прогоны.
    Здесь 90 прогонов по шагу и 10 по полсотни: среднее ~5.9, p95 — 50."""
    rep = check(proc(rules=[{"always": "a"}]), log(*([["a"]] * 90 + [["a"] * 50] * 10)))
    assert rep.steps_p95 == 50


# --- пороги и вердикт -------------------------------------------------------

def test_thresholds_produce_failures():
    p = proc(flow="plan -> respond",
             thresholds={"fitness_min": 0.99, "usd_per_case_max": 0.5,
                         "steps_p95_max": 2, "off_model_share_max": 0.01})
    rep = check(p, seq("c1", ["plan", "лишнее", "respond"], cost=1.0))
    assert not rep.ok
    assert len(rep.failures) == 4


def test_passing_run_is_ok():
    p = proc(flow="plan -> respond", thresholds={"fitness_min": 0.9, "usd_per_case_max": 10})
    rep = check(p, seq("c1", ["plan", "respond"], cost=1.0))
    assert rep.ok and rep.failures == []


def test_allow_tolerates_imperfect_agents():
    """Агент недетерминирован по устройству: нулевой допуск на каждое правило
    сделал бы проверку невыполнимой и её бы отключили."""
    rules = [{"after": "act", "expect": "verify", "allow": 0.5}]
    good, bad = ["act", "verify"], ["act"]
    rep = check(proc(rules=rules), log(bad, bad, good, good, good))
    assert rep.rules[0].cases == 2 and rep.ok          # 40% нарушителей при допуске 50%
    strict = check(proc(rules=[{"after": "act", "expect": "verify"}]), log(bad, good))
    assert not strict.ok


def test_warn_rule_does_not_fail_the_build():
    rep = check(proc(rules=[{"never": "notify", "warn": True}]), seq("c1", ["notify"]))
    assert rep.ok
    assert rep.rule_warnings and not rep.failures


def test_baseline_catches_cost_growth():
    p = proc(rules=[{"always": "a"}], thresholds={"usd_per_case_increase_max": 0.1})
    rep = check(p, seq("c1", ["a"], cost=2.0), baseline=seq("c1", ["a"], cost=1.0))
    assert not rep.ok and "выросла" in rep.failures[0]


def test_baseline_catches_fitness_drop():
    """РЕГРЕССИЯ: порог объявлен в yaml — значит, обязан что-то проверять.
    Порог, который молча ничего не делает, хуже отсутствующего."""
    p = proc(flow="a -> b", thresholds={"fitness_drop_max": 0.05})
    rep = check(p, seq("now", ["a", "лишнее", "b"]), baseline=seq("was", ["a", "b"]))
    assert not rep.ok and "fitness просел" in rep.failures[0]


def test_baseline_catches_length_growth():
    p = proc(rules=[{"always": "a"}], thresholds={"steps_increase_max": 0.5})
    rep = check(p, seq("now", ["a"] * 4), baseline=seq("was", ["a", "a"]))
    assert not rep.ok and "длины прогона" in rep.failures[0]


def test_no_baseline_means_no_growth_checks():
    p = proc(rules=[{"always": "a"}],
             thresholds={"usd_per_case_increase_max": 0.0, "fitness_drop_max": 0.0})
    assert check(p, seq("c1", ["a"], cost=99.0)).ok


# --- находки ----------------------------------------------------------------

def test_conformance_finding_carries_the_money():
    p = proc(flow="plan -> respond")
    rep = check(p, seq("c1", ["plan", "лишнее", "respond"], cost=2.0))
    f = findings(rep)[0]
    assert f.kind == "conformance" and f.impact_usd == pytest.approx(2.0)
    assert "лишний шаг `лишнее`" in " ".join(f.evidence)


def test_no_finding_when_everything_conforms():
    rep = check(proc(flow="plan -> respond"), seq("c1", ["plan", "respond"]))
    assert findings(rep) == []


def test_rule_finding_wording_depends_on_the_rule_kind():
    """РЕГРЕССИЯ: формулировку про деньги выбирал ноль в стоимости, а не вид
    правила. `never` при бесплатном инструменте объявлялся нарушением «того,
    чего не произошло» — хотя произошло ровно то, что запрещено."""
    rep = check(proc(rules=[{"never": "notify"}]), seq("c1", ["notify"], cost=0.0))
    assert "токенов не тратят" in findings(rep)[0].detail

    rep = check(proc(rules=[{"after": "act", "expect": "verify"}]), seq("c1", ["act"], cost=5.0))
    assert "чего НЕ произошло" in findings(rep)[0].detail


def test_alternation_reports_all_branches_not_one():
    """РЕГРЕССИЯ (найдено на реальном логе). У `(a | b | c)` выравниванию всё
    равно, какую ветку объявить пропущенной — цена одинакова. Отчёт называл одну
    произвольную, и в 410 прогонах это был инструмент, которого в логе нет ни
    разу: читатель шёл чинить не то."""
    rep = check(proc(flow="start -> (tool:Read | tool:Grep | tool:Glob) -> done"),
                seq("c1", ["start", "done"]))
    missing = [d for d in rep.deviations if d.kind == "model"]
    assert len(missing) == 1
    assert missing[0].activity == "tool:Glob | tool:Grep | tool:Read"


def test_near_zero_off_model_cost_is_not_reported_as_harmless():
    """РЕГРЕССИЯ: «вне модели 0% бюджета» читается как «отклонения бесплатны».
    Вне модели почти всегда оказываются вызовы инструментов: токенов они не
    тратят, но их результат перечитывается на всех последующих ходах."""
    evs = seq("c1", ["plan", "tool:Read", "tool:Read", "respond"], cost=0.0)
    evs[0]["cost_usd"] = 5.0
    rep = check(proc(flow="plan -> respond"), evs)
    f = findings(rep)[0]
    assert "токенов они не тратят" in f.detail
    # Ярлык «до $0.00» рядом с находкой обесценивает её сильнее, чем его отсутствие.
    assert f.impact_usd == 0.0


# --- красная зона на графе --------------------------------------------------

def test_forbidden_activity_is_not_allowed():
    """РЕГРЕССИЯ: активность из `never:` упомянута в модели — и красилась как
    её часть. Её присутствие — самое сильное отклонение, какое бывает."""
    p = proc(flow="plan -> respond", rules=[{"never": "notify"}])
    assert "notify" in p.activities()
    assert "notify" not in p.allowed()
    assert p.forbidden() == {"notify"}


# --- CLI: коды возврата -----------------------------------------------------

def _events(tmp_path, traces, cost=1.0):
    """Мини-лог на диске: коды возврата проверяются только сквозь настоящий CLI."""
    from pathlib import Path

    from agentmine.model import COLUMNS, Event
    from agentmine.store import write
    evs = []
    for i, t in enumerate(traces):
        for j, a in enumerate(t):
            evs.append(Event(case_id=f"c{i}", event_id=f"c{i}-{j}", activity_raw=a,
                             activity=a, ts_start=float(j), ts_end=float(j) + 1,
                             parent_id=None, agent=None, resource=None, tokens_in=0,
                             tokens_cached=0, tokens_cache_write=0, tokens_out=0,
                             cost_usd=cost, status="ok", error_type=None, attrs={}))
    path = tmp_path / "events.parquet"
    write(evs, path)
    return path


def _run(*args):
    from typer.testing import CliRunner

    from agentmine.cli import app
    return CliRunner().invoke(app, [str(a) for a in args])


def _process(tmp_path, body: dict):
    path = tmp_path / "process.yaml"
    path.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return path


def test_cli_exits_0_when_conforming(tmp_path):
    ev_path = _events(tmp_path, [["plan", "respond"]])
    pr = _process(tmp_path, {"flow": "plan -> respond", "thresholds": {"fitness_min": 0.9}})
    assert _run("check", ev_path, "-p", pr).exit_code == 0


def test_cli_exits_1_when_violated(tmp_path):
    ev_path = _events(tmp_path, [["plan", "лишнее", "respond"]])
    pr = _process(tmp_path, {"flow": "plan -> respond", "thresholds": {"fitness_min": 0.99}})
    assert _run("check", ev_path, "-p", pr).exit_code == 1


def test_cli_exits_2_on_broken_config(tmp_path):
    """Сломанный конфиг и провалившаяся проверка требуют разной реакции, а CI
    видит только код возврата. Смешать их — значит чинить не то."""
    ev_path = _events(tmp_path, [["plan", "respond"]])
    pr = tmp_path / "process.yaml"
    pr.write_text("flow: 'plan -> (respond'\n", encoding="utf-8")
    res = _run("check", ev_path, "-p", pr)
    assert res.exit_code == 2


def test_cli_warn_only_never_fails(tmp_path):
    ev_path = _events(tmp_path, [["plan", "лишнее", "respond"]])
    pr = _process(tmp_path, {"flow": "plan -> respond", "thresholds": {"fitness_min": 0.99}})
    assert _run("check", ev_path, "-p", pr, "--warn-only").exit_code == 0


def test_cli_writes_github_outputs(tmp_path, monkeypatch):
    """Экшену не нужен ни парсинг вывода, ни знание формата отчёта."""
    ev_path = _events(tmp_path, [["plan", "respond"]])
    pr = _process(tmp_path, {"flow": "plan -> respond"})
    out, summary = tmp_path / "out.txt", tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert _run("check", ev_path, "-p", pr).exit_code == 0
    written = dict(line.split("=", 1) for line in out.read_text().strip().splitlines())
    assert written["status"] == "ok" and written["fitness"] == "1.0000"
    assert "flowchart TD" in summary.read_text()
