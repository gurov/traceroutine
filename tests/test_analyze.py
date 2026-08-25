"""Тесты слоя Analyze."""
from __future__ import annotations

import pytest

from agentmine.analyze import (RECOVERY_WINDOW, VARIANT_MIN_REUSE, _path_digest,
                               _rotation_key, context_inflation, drift, find_cycles,
                               findings)
from agentmine.mine import Model, Variant, mine


def ev(case, act, cost=0.0, status="ok", tin=0, tcached=0, ts=0.0):
    return {"case_id": case, "event_id": f"{case}-{ts}", "activity_raw": act,
            "activity": act, "ts_start": ts, "ts_end": ts + 1, "parent_id": None,
            "agent": None, "resource": None, "tokens_in": tin, "tokens_cached": tcached,
            "tokens_cache_write": 0, "tokens_out": 0, "cost_usd": cost, "status": status,
            "error_type": None, "attrs": "{}"}


def seq(case, acts, cost=1.0):
    return [ev(case, a, cost=cost, ts=float(i)) for i, a in enumerate(acts)]


# --- детекция циклов --------------------------------------------------------

def test_detects_repeating_block():
    """Цикл — повторяющийся БЛОК, а не просто повторившаяся активность."""
    evs = seq("c1", ["plan", "search", "read", "search", "read", "search", "read", "done"])
    c = find_cycles(evs)[0]
    assert c.pattern == ("read", "search") or c.pattern == ("search", "read")
    assert c.max_repeats == 3


def test_only_extra_iterations_count_as_waste():
    """Первый проход по циклу — работа, а не потери. Считаем только сверх первого."""
    evs = seq("c1", ["a", "b", "a", "b", "a", "b"], cost=1.0)
    c = find_cycles(evs)[0]
    assert c.extra_cost == pytest.approx(4.0)      # 6 событий, 2 из них — первая итерация
    assert c.extra_events == 4


def test_rotations_are_one_cycle():
    """РЕГРЕССИЯ: `a→b→b` и `b→b→a` — один цикл, пойманный с разных мест.
    Без склейки один дефект превращался в три отдельные «находки»."""
    assert _rotation_key(("generate", "retrieve", "retrieve")) == \
           _rotation_key(("retrieve", "retrieve", "generate"))
    evs = seq("c1", ["x", "a", "b", "a", "b", "y"]) + seq("c2", ["y", "b", "a", "b", "a", "x"])
    assert len({c.pattern for c in find_cycles(evs)}) == 1


def test_no_cycle_in_linear_run():
    assert find_cycles(seq("c1", ["a", "b", "c", "d"])) == []


# --- подача --------------------------------------------------------------

def test_path_digest_keeps_paths_distinguishable():
    """РЕГРЕССИЯ: свёртка середины делала разные варианты одинаковыми в отчёте —
    края у них общие, различаются они как раз в середине."""
    a = ("plan", "gen", "retrieve", "retrieve", "gen", "respond")
    b = ("plan", "gen", "retrieve", "verify", "gen", "respond")
    assert _path_digest(a) != _path_digest(b)


def test_path_digest_truncates_only_very_long():
    long = tuple(f"s{i}" for i in range(20))
    assert "ещё 8" in _path_digest(long)


# --- находки ----------------------------------------------------------------

def test_findings_are_capped_and_sorted_by_money():
    evs = seq("c1", ["a", "b"] * 6, cost=5.0)
    for i in range(20):
        evs += seq(f"x{i}", ["a", "b"], cost=0.01)
    m = mine(evs)
    out = findings(m, evs, limit=3)
    assert len(out) <= 3
    assert out == sorted(out, key=lambda f: (-f.impact_usd, -f.share))


def test_immaterial_cycles_are_not_reported():
    """Цикл на 1% бюджета — шум, вытесняющий из списка то, что чинить стоит."""
    evs = seq("big", ["expensive"], cost=1000.0)
    evs += seq("tiny", ["a", "b", "a", "b"], cost=0.5)
    kinds = {f.kind for f in findings(mine(evs), evs)}
    assert "cycle" not in kinds


def test_dead_cache_is_flagged():
    evs = [ev(f"c{i}", "chat", cost=1.0, tin=20_000, tcached=0, ts=0.0) for i in range(10)]
    assert any(f.kind == "cache" for f in findings(mine(evs), evs))


def test_working_cache_is_not_flagged():
    evs = [ev(f"c{i}", "chat", cost=1.0, tin=5_000, tcached=15_000, ts=0.0) for i in range(10)]
    assert not any(f.kind == "cache" for f in findings(mine(evs), evs))


def test_errors_finding_charges_recovery_window_not_whole_case():
    """Регрессия. Раньше на ошибку вешалась стоимость ВСЕГО прогона.

    На реальных транскриптах Claude Code это давало «сбои стоят 51% бюджета»:
    в длинном прогоне почти всегда есть хоть одна ошибка, и находка вырождалась.
    Считаем сбойный шаг плюс окно восстановления, и ни шагом больше.
    """
    ok = seq("ok1", list("abcdefgh"), cost=1.0)
    bad = seq("bad", list("abcdefgh"), cost=1.0)
    bad[1]["status"] = "error"
    m = mine(ok + bad)
    f = next(f for f in findings(m, ok + bad) if f.kind == "errors")
    assert f.impact_usd == pytest.approx(1.0 + RECOVERY_WINDOW)
    assert f.impact_usd < 8.0                      # не весь прогон


def test_error_windows_do_not_double_charge_overlap():
    """Две ошибки подряд не должны оплачивать пересечение своих окон дважды."""
    evs = seq("c1", list("abcdef"), cost=1.0)
    evs[1]["status"] = evs[2]["status"] = "error"
    m = mine(evs)
    f = next(f for f in findings(m, evs) if f.kind == "errors")
    assert f.impact_usd == pytest.approx(4.0)      # шаги 1..4, а не 3+3


# --- применимость вариантного анализа ---------------------------------------

def test_variant_findings_suppressed_when_paths_never_repeat():
    """Регрессия на главный дефект, который вскрыли реальные данные.

    Когда каждый прогон уникален, «редкие пути съедают бюджет» — тавтология
    «дорогие прогоны стоят дорого». Вместо неё инструмент обязан честно сказать,
    что линза не подходит.
    """
    import string
    evs = []
    for i in range(30):                            # 30 заведомо разных длинных путей
        acts = [string.ascii_lowercase[(i + j) % 26] for j in range(20)]
        evs += seq(f"c{i}", acts, cost=1.0)
    m = mine(evs)
    assert m.variant_reuse < VARIANT_MIN_REUSE
    kinds = {f.kind for f in findings(m, evs)}
    assert "tail" not in kinds and "concentration" not in kinds
    assert "not_applicable" in kinds


def test_variant_findings_kept_when_paths_do_repeat():
    """Обратная сторона: там, где пути повторяются, линза работает и не глушится."""
    evs = []
    for i in range(20):
        evs += seq(f"cheap{i}", ["a", "b"], cost=0.01)
    for i in range(2):
        evs += seq(f"pricey{i}", ["a", "x", "x", "b"], cost=5.0)
    m = mine(evs)
    assert m.variant_reuse >= VARIANT_MIN_REUSE
    assert {f.kind for f in findings(m, evs)} & {"tail", "concentration"}


def test_dominant_cycle_reported_as_rhythm_not_defect():
    """`chat → bash` в кодовом агенте — режим работы, а не дефект.

    Цикл, который есть в большинстве прогонов, нельзя называть «лишним
    прокручиванием»: чинить там нечего, и ложная задача вытесняет настоящую.
    """
    evs = []
    for i in range(10):
        evs += seq(f"c{i}", ["chat", "bash"] * 4, cost=1.0)
    m = mine(evs)
    f = next(f for f in findings(m, evs) if f.kind in ("cycle", "rhythm"))
    assert f.kind == "rhythm"
    assert f.impact_usd == 0.0                     # не выдаём это за экономию


# --- стоимость контекста ----------------------------------------------------

def test_context_inflation_charges_carry_not_just_delta():
    """Шаг, добавивший контекст рано, оплачивается всеми последующими ходами.

    Ровно то, что не видит атрибуция «по запросу»: сам шаг стоит $0.00.
    """
    evs = []
    ctx = 1_000
    for i in range(6):
        evs.append(ev("c1", "chat", cost=0.1, tcached=ctx, ts=float(2 * i)))
        evs.append(ev("c1", "tool:big" if i == 0 else "tool:small",
                      cost=0.0, ts=float(2 * i + 1)))
        ctx += 10_000 if i == 0 else 100
    res = {c.step: c for c in context_inflation(evs)}
    assert res["tool:big"].carried_tokens > res["tool:small"].carried_tokens * 5
    assert res["tool:big"].est_usd > 0             # хотя сам шаг стоит $0.00


def test_context_inflation_ignores_compaction():
    """Сжатие контекста — не отрицательный вклад, а просто не вклад."""
    evs = [ev("c1", "chat", cost=0.1, tcached=50_000, ts=0.0),
           ev("c1", "tool:x", ts=1.0),
           ev("c1", "chat", cost=0.1, tcached=5_000, ts=2.0)]
    assert all(c.added_avg >= 0 for c in context_inflation(evs))


# --- дрейф ------------------------------------------------------------------

def test_drift_reports_cost_and_length_change():
    a = mine(seq("c1", ["x", "y"], cost=1.0))
    b = mine(seq("c1", ["x", "y", "y", "y"], cost=1.0))
    d = drift(a, b)
    assert d.cost_change == pytest.approx(1.0)     # $2 → $4 на прогон
    assert d.len_change == pytest.approx(1.0)


def test_drift_detects_new_transition():
    a = mine(seq("c1", ["x", "y"]))
    b = mine(seq("c1", ["x", "z", "y"]))
    d = drift(a, b)
    assert ("x", "z") in {(p, q) for p, q, _ in d.new_edges}
    assert ("x", "y") in {(p, q) for p, q, _ in d.gone_edges}


def test_drift_disables_enumeration_when_variants_explode():
    """При сотнях вариантов перечисление траекторий — шум, а не сигнал."""
    m = Model(n_cases=200)
    m.variants = [Variant((f"a{i}",), n=1) for i in range(200)]
    assert drift(m, m).enumerable is False
