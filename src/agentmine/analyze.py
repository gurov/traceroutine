"""Слой Analyze: из модели процесса — findings, а не графики.

Разница принципиальная. Дашборд говорит «вот ваши расходы» — это витамин.
Finding говорит «вот этот цикл стоит вам $8k в месяц, вот как проверить» — это
обезболивающее. Поэтому отчёт открывается списком находок, отсортированных по деньгам,
а граф идёт после: он объясняет находку, а не заменяет её.

Оценки экономии сознательно консервативны: лучше недооценить и сохранить доверие,
чем назвать красивое число, которое клиент не сможет воспроизвести.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .mine import END, START, Model, Variant
from .store import cases


@dataclass
class Finding:
    kind: str
    title: str
    detail: str
    impact_usd: float = 0.0          # консервативная оценка устранимых расходов
    share: float = 0.0               # доля от общих расходов
    evidence: list[str] = field(default_factory=list)


@dataclass
class Cycle:
    pattern: tuple[str, ...]
    occurrences: int = 0             # сколько раз паттерн повторялся сверх первого раза
    cases: int = 0
    extra_cost: float = 0.0          # стоимость итераций СВЕРХ первой
    extra_events: int = 0
    max_repeats: int = 1


def _rotation_key(pattern: tuple[str, ...]) -> tuple[str, ...]:
    """Канонический ключ цикла: лексикографически наименьшая ротация.

    `generate→retrieve→retrieve` и `retrieve→retrieve→generate` — один и тот же цикл,
    пойманный с разных мест последовательности. Без склейки один дефект превращается
    в три «находки», и список перестаёт быть списком дел.
    """
    n = len(pattern)
    return min(tuple(pattern[i:] + pattern[:i]) for i in range(n)) if n else pattern


def find_cycles(events: list[dict], max_period: int = 4) -> list[Cycle]:
    """Повторяющиеся смежные подпоследовательности — настоящие циклы процесса.

    Это не то же самое, что rework rate. Rework говорит «активность встретилась
    дважды»; цикл говорит «блок `retrieve → generate` прокрутился четыре раза».
    Первое — симптом, второе — то, что можно чинить.

    Считаем только итерации СВЕРХ первой: первый проход по циклу — это работа,
    а не потери.
    """
    found: dict[tuple[str, ...], Cycle] = {}

    for case_id, evs in cases(events):
        seq = [e["activity"] for e in evs]
        costs = [e["cost_usd"] or 0.0 for e in evs]
        n = len(seq)
        i = 0
        seen_here: set[tuple[str, ...]] = set()
        while i < n:
            best: tuple[int, int] | None = None      # (период, число повторов)
            for p in range(1, min(max_period, (n - i) // 2) + 1):
                reps = 1
                while (i + (reps + 1) * p <= n
                       and seq[i:i + p] == seq[i + reps * p:i + (reps + 1) * p]):
                    reps += 1
                if reps > 1 and (best is None or reps * p > best[0] * best[1]):
                    best = (p, reps)
            if best is None:
                i += 1
                continue
            p, reps = best
            pattern = tuple(seq[i:i + p])
            key = _rotation_key(pattern)
            c = found.setdefault(key, Cycle(key))
            extra_slice = slice(i + p, i + reps * p)  # всё, кроме первой итерации
            c.occurrences += reps - 1
            c.extra_cost += sum(costs[extra_slice])
            c.extra_events += (reps - 1) * p
            c.max_repeats = max(c.max_repeats, reps)
            if key not in seen_here:
                c.cases += 1
                seen_here.add(key)
            i += reps * p
    return sorted(found.values(), key=lambda c: -c.extra_cost)


# Сколько шагов после сбоя считать «работой из-за сбоя».
RECOVERY_WINDOW = 2


def _error_cost(events: list[dict]) -> tuple[float, int, Counter]:
    """Во что обходятся сбои: не сама ошибка (она обычно бесплатна), а разбор её
    последствий.

    Раньше здесь на ошибку вешалась стоимость ВСЕГО кейса, где она случилась. На
    коротких прогонах это близко к правде, на длинных — нет: одна упавшая команда
    в прогоне из 75 шагов не делает все 75 шагов следствием сбоя. На реальных
    транскриптах такая атрибуция давала «сбои стоят 51% бюджета» — число большое,
    неверное и бесполезное, потому что хоть одна ошибка есть почти в каждом прогоне.

    Считаем консервативно: сам сбойный шаг плюс ближайшие RECOVERY_WINDOW ходов —
    то, чем агент на сбой ответил. Недооценка здесь предпочтительнее: назвать
    завышенную экономию, которую клиент не воспроизведёт, дороже, чем промолчать.
    """
    cost, n = 0.0, 0
    kinds: Counter[str] = Counter()
    for _case_id, evs in cases(events):
        evs = sorted(evs, key=lambda e: (e["ts_start"], e["event_id"]))
        idx = [i for i, e in enumerate(evs) if e["status"] != "ok"]
        if not idx:
            continue
        n += 1
        charged: set[int] = set()
        for i in idx:
            charged.update(range(i, min(i + 1 + RECOVERY_WINDOW, len(evs))))
            kinds[evs[i]["error_type"] or evs[i]["activity"]] += 1
        cost += sum(evs[i]["cost_usd"] or 0.0 for i in charged)
    return cost, n, kinds


def _tail_variants(m: Model) -> tuple[list[Variant], float, float]:
    """«Длинный хвост»: редкие пути, каждый из которых дороже медианного прогона."""
    if not m.variants or not m.n_cases:
        return [], 0.0, 0.0
    med = sorted(v.cost_avg for v in m.variants)[len(m.variants) // 2]
    tail = [v for v in m.variants
            if v.n / m.n_cases < 0.05 and v.cost_avg > med * 2]
    return (sorted(tail, key=lambda v: -v.cost)[:5],
            sum(v.cost for v in tail),
            sum(v.n for v in tail) / m.n_cases)


def _path_digest(seq: tuple[str, ...], full_upto: int = 12) -> str:
    """Путь целиком, пока он читается.

    Раньше здесь сворачивалась середина в предположении, что похожие варианты
    расходятся по краям. На деле — наоборот: края у них общие (plan…respond),
    а различаются они как раз в середине, и свёртка делала разные пути
    неотличимыми в отчёте.
    """
    if len(seq) <= full_upto:
        return " → ".join(seq)
    return f"{' → '.join(seq[:full_upto])} …ещё {len(seq) - full_upto}"


# Ниже какой доли повторяющихся путей вариантный анализ перестаёт что-либо значить.
# 0.5 — не подобранное число: при reuse ниже половины больше половины кейсов имеют
# уникальный путь, «редкий путь» становится синонимом «любой путь», и находка
# вырождается в тавтологию «дорогие прогоны стоят дорого».
VARIANT_MIN_REUSE = 0.5


def findings(m: Model, events: list[dict], limit: int = 5) -> list[Finding]:
    out: list[Finding] = []
    total = m.total_cost or 1.0
    variants_apply = m.variant_reuse >= VARIANT_MIN_REUSE

    # 1. Циклы — самая частая и самая чинимая утечка.
    # Порог существенности: цикл на 1% бюджета — не задача, а шум, и он вытесняет
    # из списка то, что чинить стоит.
    for c in find_cycles(events)[:2]:
        if c.extra_cost <= 0 or c.extra_cost / total < 0.02:
            continue
        # Блок, повторяющийся в большинстве прогонов, — это РАБОЧИЙ РИТМ агента, а
        # не дефект. У кодового агента `chat → tool:Bash → chat → tool:Bash` и есть
        # нормальная работа; назвать это «лишним прокручиванием» — соврать. Цену
        # ритма показать всё равно стоит, но как факт, а не как задачу.
        rhythm = c.cases / m.n_cases > 0.5 if m.n_cases else False
        out.append(Finding(
            kind="rhythm" if rhythm else "cycle",
            title=(f"Рабочий цикл `{' → '.join(c.pattern)}` — {c.extra_cost / total:.0%} бюджета"
                   if rhythm else
                   f"Цикл `{' → '.join(c.pattern)}` прокручивается лишний раз"),
            detail=(
                (f"Встречается в {c.cases} прогонах из {m.n_cases} — это основной режим "
                 f"работы агента, а не аномалия. Экономия здесь требует менять сам "
                 f"подход, а не чинить отдельный путь. Максимум {c.max_repeats} "
                 f"итераций подряд.") if rhythm else
                (f"В {c.cases} прогонах из {m.n_cases} этот блок повторяется; "
                 f"максимум {c.max_repeats} итераций подряд. Итерации сверх первой — "
                 f"{c.extra_events} событий.")
            ),
            impact_usd=0.0 if rhythm else c.extra_cost,
            share=c.extra_cost / total,
            evidence=[f"повторов сверх первого: {c.occurrences}"],
        ))

    # 2. Концентрация стоимости по траекториям.
    conc = m.cost_concentration()
    acc_cases = acc_cost = 0.0
    for v, share, cum in conc:
        acc_cases += share
        acc_cost = cum
        if cum >= 0.5:
            break
    if variants_apply and acc_cases and acc_cases <= 0.25:
        out.append(Finding(
            kind="concentration",
            title=f"{acc_cases:.0%} прогонов дают {acc_cost:.0%} расходов",
            detail=("Стоимость сосредоточена в узком классе траекторий. "
                    "Оптимизировать средний запрос бессмысленно — чинить надо эти пути."),
            share=acc_cost,
            evidence=[f"самый дорогой путь: {' → '.join(conc[0][0].seq[:6])}"],
        ))

    # 3. Дорогой длинный хвост.
    tail, tail_cost, tail_share = _tail_variants(m)
    if variants_apply and tail and tail_cost / total > 0.1:
        out.append(Finding(
            kind="tail",
            title=f"Редкие пути ({tail_share:.1%} прогонов) съедают {tail_cost / total:.0%} бюджета",
            detail=("Каждый встречается реже чем в 5% случаев, но стоит вдвое дороже "
                    "медианного прогона. В агрегированных дашбордах такие пути невидимы."),
            impact_usd=tail_cost,
            share=tail_cost / total,
            # Обрезка пути до N шагов делала разные варианты неотличимыми в отчёте.
            # Показываем длину и середину пути, где они и расходятся.
            evidence=[f"{v.n}× · ${v.cost_avg:.3f}/прогон · {len(v.seq)} шагов · "
                      f"{_path_digest(v.seq)}" for v in tail[:3]],
        ))

    # 4. Цена сбоев.
    err_cost, err_cases, kinds = _error_cost(events)
    if err_cases and err_cost / total > 0.05:
        top = ", ".join(f"{k} ×{n}" for k, n in kinds.most_common(3))
        out.append(Finding(
            kind="errors",
            title=f"Разбор сбоев стоит {err_cost / total:.0%} бюджета",
            detail=(f"{err_cases} прогонов из {m.n_cases} содержат хотя бы одну ошибку. "
                    f"Платится не сама ошибка, а разбор её последствий: сюда посчитан "
                    f"сбойный шаг и {RECOVERY_WINDOW} ближайших хода после него."),
            impact_usd=err_cost,
            share=err_cost / total,
            evidence=[top],
        ))

    # 5. Дорогой ресурс на редких вызовах — эффект эскалации.
    if m.resource_cost:
        calls = sum(r["n"] for r in m.resource_cost.values()) or 1
        for name, r in sorted(m.resource_cost.items(), key=lambda kv: -kv[1]["cost"]):
            cs, ss = r["n"] / calls, r["cost"] / total
            if ss > 0.2 and cs < 0.1:
                out.append(Finding(
                    kind="escalation",
                    title=f"`{name}`: {cs:.1%} вызовов, {ss:.0%} расходов",
                    detail=("Ресурс используется редко, но стоит непропорционально дорого. "
                            "Проверьте, оправдана ли эскалация на каждом из этих путей."),
                    impact_usd=r["cost"],
                    share=ss,
                    evidence=[f"{r['n']:,} вызовов, {r['tokens']:,} токенов"],
                ))
                break

    # 6. Кеш, который не работает.
    tin = sum(e["tokens_in"] or 0 for e in events)
    tcached = sum(e["tokens_cached"] or 0 for e in events)
    if tin > 100_000 and tcached / (tin + tcached) < 0.05:
        out.append(Finding(
            kind="cache",
            title="Кеш промптов практически не срабатывает",
            detail=(f"Из {tin + tcached:,} входных токенов из кеша прочитано "
                    f"{tcached:,} ({tcached / (tin + tcached):.1%}). Обычная причина — "
                    "нестабильный префикс: меняющийся набор инструментов, timestamp "
                    "или несортированный JSON в системном промпте."),
            evidence=["кеш читается по префиксу: любое изменение инвалидирует хвост"],
        ))

    # 7. Раздувание контекста. Работает там, где вариантный анализ уже бесполезен:
    # чем длиннее прогоны, тем метрика значимее.
    infl = context_inflation(events, top=3)
    if infl and infl[0].est_usd / total > 0.03:
        c = infl[0]
        out.append(Finding(
            kind="context",
            title=f"Результаты `{c.step}` тянут {c.est_usd / total:.0%} бюджета через контекст",
            detail=(
                f"Сам шаг не тратит ни одного токена и в разбивке по стоимости стоит "
                f"$0.00. Но каждый его результат добавляет в промпт ~{c.added_avg:,.0f} "
                f"токенов, и они перечитываются на КАЖДОМ последующем ходе: суммарно "
                f"{c.carried_tokens / 1e6:,.1f}M токенов. Чинится усечением вывода, "
                f"а не заменой модели."
            ),
            impact_usd=c.est_usd,
            share=c.est_usd / total,
            evidence=[f"{x.step}: {x.n:,}× · +{x.added_avg:,.0f} ток. · ≈${x.est_usd:,.2f}"
                      for x in infl],
        ))

    # 8. Честный отказ. Инструмент, который на любых данных выдаёт пять находок,
    # рано или поздно выдаёт пять выдуманных. Сказать «эта линза не подходит вашему
    # логу и вот почему» — полезнее, чем тавтология с большим числом долларов.
    if not variants_apply:
        med = sorted(len(v.seq) for v in m.variants)[len(m.variants) // 2] if m.variants else 0
        out.append(Finding(
            kind="not_applicable",
            title="Вариантный анализ к этому логу неприменим",
            detail=(
                f"Повторяющихся путей всего {m.variant_reuse:.0%}: {len(m.variants)} "
                f"путей на {m.n_cases} прогонов, медиана длины прогона — {med} шагов. "
                f"Траектории такой длины не повторяются в принципе, поэтому выводы "
                f"вида «редкие пути съедают бюджет» здесь были бы тавтологией и не "
                f"показаны. Что помогает: укрупнить активности (`agentmine abstract`), "
                f"выбрать более узкий case notion (`--case task`) — либо смотреть на "
                f"метрики, не зависящие от длины: раздувание контекста и сравнение "
                f"через `agentmine diff`."
            ),
            evidence=[f"порог применимости: повторяющихся путей ≥ {VARIANT_MIN_REUSE:.0%}"],
        ))

    # Список «что чинить» должен быть коротким, иначе это не список дел, а свалка.
    return sorted(out, key=lambda f: (-f.impact_usd, -f.share))[:limit]


# --- стоимость контекста ----------------------------------------------------

@dataclass
class ContextCost:
    """Во что обошёлся контекст, добавленный одним видом шага."""
    step: str
    n: int = 0                       # сколько раз шаг встречался
    added_avg: float = 0.0           # средний прирост промпта после него, токенов
    carried_tokens: float = 0.0      # прирост × число ходов, которые его потом несли
    est_usd: float = 0.0


def context_inflation(events: list[dict], top: int = 6) -> list[ContextCost]:
    """Кто раздувает контекст — и во что это обходится на всей оставшейся траектории.

    Метрика, ради которой стоит смотреть на агента процессно. Вызов инструмента сам
    по себе не стоит ни одного токена: в разбивке по стоимости все инструменты ровно
    по $0.00, а 100% расходов висит на `chat`. Вывод «оптимизируйте chat» бесполезен.

    На деле результат инструмента попадает в промпт и **перечитывается на каждом
    последующем ходе**. Один `Read` большого файла на десятом шаге из сорока
    оплачивается тридцать раз. Это и есть стоимость ТРАЕКТОРИИ, а не запроса:
    цена шага определяется не им самим, а тем, сколько ходов после него осталось.

    Ни одна платформа наблюдаемости так не считает — они атрибутируют расход по
    запросу и по пользователю, а не по тому, что этот расход породило.

    В отличие от вариантного анализа, метрика не вырождается на длинных прогонах:
    чем длиннее траектория, тем она ЗНАЧИМЕЕ.
    """
    by_case: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_case[e["case_id"]].append(e)

    def prompt_size(e: dict) -> int:
        return ((e.get("tokens_in") or 0) + (e.get("tokens_cached") or 0)
                + (e.get("tokens_cache_write") or 0))

    # Эффективная ставка за токен промпта, выведенная из самих данных: почти весь
    # промпт — чтение кеша, и брать за него полную цену input было бы враньём.
    paid = [e for e in events if prompt_size(e) and (e.get("cost_usd") or 0) > 0]
    tok = sum(prompt_size(e) for e in paid)
    rate = (sum(e["cost_usd"] for e in paid) / tok) if tok else 0.0

    added: dict[str, list[float]] = defaultdict(list)
    carried: dict[str, float] = defaultdict(float)

    for evs in by_case.values():
        evs = sorted(evs, key=lambda e: (e["ts_start"], e["event_id"]))
        turns = [i for i, e in enumerate(evs) if prompt_size(e)]
        for k, (i, j) in enumerate(zip(turns, turns[1:])):
            delta = prompt_size(evs[j]) - prompt_size(evs[i])
            if delta <= 0:
                continue                     # компакция или сброс контекста
            between = [evs[x]["activity"] for x in range(i + 1, j)] or [evs[i]["activity"]]
            remaining = len(turns) - k - 1
            for name in between:
                share = delta / len(between)
                added[name].append(share)
                carried[name] += share * remaining

    out = [ContextCost(step=name, n=len(vals),
                       added_avg=sum(vals) / len(vals),
                       carried_tokens=carried[name],
                       est_usd=carried[name] * rate)
           for name, vals in added.items() if vals]
    return sorted(out, key=lambda c: -c.est_usd)[:top]


# --- сравнение двух логов ---------------------------------------------------

@dataclass
class Drift:
    cost_per_case_before: float = 0.0
    cost_per_case_after: float = 0.0
    len_before: float = 0.0
    len_after: float = 0.0
    variants_before: int = 0
    variants_after: int = 0
    appeared: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    vanished: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    activity_delta: list[tuple[str, float, float]] = field(default_factory=list)
    # Новые/исчезнувшие ПЕРЕХОДЫ. При сотнях вариантов перечислять отдельные
    # траектории бессмысленно — они все уникальны. Новое ребро в графе означает
    # новый вид поведения, и это компактно и читаемо.
    new_edges: list[tuple[str, str, int]] = field(default_factory=list)
    gone_edges: list[tuple[str, str, int]] = field(default_factory=list)
    enumerable: bool = True
    reuse: float = 0.0

    @property
    def cost_change(self) -> float:
        b = self.cost_per_case_before
        return (self.cost_per_case_after - b) / b if b else 0.0

    @property
    def len_change(self) -> float:
        return (self.len_after - self.len_before) / self.len_before if self.len_before else 0.0


def drift(before: Model, after: Model) -> Drift:
    """Сравнение процессов до и после изменения — смены модели, релиза промпта.

    Триггер повторяется при каждом релизе модели, а релизы частые: это делает
    сравнение самой регулярно нужной операцией из всех.
    """
    def per_case(m: Model) -> float:
        return m.total_cost / m.n_cases if m.n_cases else 0.0

    def avg_len(m: Model) -> float:
        tot = sum(len(v.seq) * v.n for v in m.variants)
        return tot / m.n_cases if m.n_cases else 0.0

    a = {v.seq: v.n for v in before.variants}
    b = {v.seq: v.n for v in after.variants}

    def freq(m: Model) -> dict[str, float]:
        return {k: s["n"] / m.n_cases for k, s in m.nodes.items()} if m.n_cases else {}

    fa, fb = freq(before), freq(after)
    deltas = [(k, fa.get(k, 0.0), fb.get(k, 0.0)) for k in set(fa) | set(fb)]
    deltas.sort(key=lambda t: -abs(t[2] - t[1]))

    ea = {k for k in before.edges}
    eb = {k for k in after.edges}

    return Drift(
        new_edges=sorted(((x, y, after.edges[(x, y)].n) for (x, y) in eb - ea),
                         key=lambda t: -t[2])[:8],
        gone_edges=sorted(((x, y, before.edges[(x, y)].n) for (x, y) in ea - eb),
                          key=lambda t: -t[2])[:8],
        # Перечислять траектории имеет смысл, пока их немного. Смотрим и на число,
        # и на повторяемость: 52 варианта на 107 прогонов — это НЕ «каждый прогон
        # уникален», и говорить так было бы неправдой.
        enumerable=max(len(before.variants), len(after.variants)) <= 50,
        reuse=min(before.variant_reuse, after.variant_reuse),
        cost_per_case_before=per_case(before), cost_per_case_after=per_case(after),
        len_before=avg_len(before), len_after=avg_len(after),
        variants_before=len(before.variants), variants_after=len(after.variants),
        appeared=sorted(((s, n) for s, n in b.items() if s not in a),
                        key=lambda t: -t[1])[:5],
        vanished=sorted(((s, n) for s, n in a.items() if s not in b),
                        key=lambda t: -t[1])[:5],
        activity_delta=[d for d in deltas if abs(d[2] - d[1]) > 0.01][:8],
    )
