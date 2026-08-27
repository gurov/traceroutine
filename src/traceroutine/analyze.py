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

from .mine import END, START, Model, Variant, batched_parents
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
    # Четыре `tool:Edit`, выданные одним ходом, — это один шаг с четырьмя целями,
    # а не цикл длиной один, прокрутившийся четырежды. Без склейки самая частая
    # «петля» в логе кодового агента оказывается его сильной стороной:
    # параллельные вызовы делят одно чтение промпта на всех.
    batched = batched_parents(events)

    for case_id, evs in cases(events):
        seq: list[str] = []
        costs: list[float] = []
        prev: tuple[str, str] | None = None
        for e in evs:
            key = (e["activity"], e["parent_id"]) if e["parent_id"] in batched else None
            if key is not None and key == prev:
                costs[-1] += e["cost_usd"] or 0.0
                continue
            prev = key
            seq.append(e["activity"])
            costs.append(e["cost_usd"] or 0.0)
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


# Путь, встретившийся один раз, — это прогон, а не путь.
PATH_MIN_RUNS = 2


def _tail_variants(m: Model) -> tuple[list[Variant], float, float]:
    """«Длинный хвост»: редкие пути, каждый из которых дороже медианного прогона.

    Одиночки исключены. На логе первого внешнего пользователя находка цитировала
    три «пути» с `n = 1` длиной 1 225, 975 и 524 шага — то есть три отдельных
    прогона, названных классами. Это ровно та тавтология, ради которой заведён
    `variants_apply`, только проехавшая мимо него: порог там стоит на уровне ЛОГА
    (повторяемость ≥ 50%), а вырождается конкретная находка.
    """
    if not m.variants or not m.n_cases:
        return [], 0.0, 0.0
    med = sorted(v.cost_avg for v in m.variants)[len(m.variants) // 2]
    tail = [v for v in m.variants
            if PATH_MIN_RUNS <= v.n and v.n / m.n_cases < 0.05 and v.cost_avg > med * 2]
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
    return f"{' → '.join(seq[:full_upto])} …+{len(seq) - full_upto} more"


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
            title=(f"Working rhythm `{' → '.join(c.pattern)}` — {c.extra_cost / total:.0%} of the budget"
                   if rhythm else
                   f"Loop `{' → '.join(c.pattern)}` runs an extra time"),
            detail=(
                (f"Present in {c.cases} of {m.n_cases} runs — this is how the agent "
                 f"works, not an anomaly. Saving here means changing the approach, "
                 f"not fixing one path. Up to {c.max_repeats} iterations in a row.")
                if rhythm else
                (f"This block repeats in {c.cases} of {m.n_cases} runs, up to "
                 f"{c.max_repeats} times in a row. Iterations beyond the first "
                 f"account for {c.extra_events} events.")
            ),
            impact_usd=0.0 if rhythm else c.extra_cost,
            share=c.extra_cost / total,
            evidence=[f"iterations beyond the first: {c.occurrences}"],
        ))

    # 2. Дорогой длинный хвост.
    tail, tail_cost, tail_share = _tail_variants(m)
    tail_fired = bool(variants_apply and tail and tail_cost / total > 0.1)

    # 3. Концентрация стоимости по траекториям — та же мысль, что и хвост, только
    # без имён и без суммы. Если хвост сработал, это дубликат: на логе первого
    # внешнего пользователя список из пяти пунктов содержал «13% прогонов дают 50%
    # расходов» и «редкие пути — 13.5% прогонов — съедают 51% бюджета». Список дел,
    # где два пункта из пяти об одном, перестаёт быть списком дел.
    conc = m.cost_concentration(min_n=PATH_MIN_RUNS)
    acc_cases = acc_cost = 0.0
    for v, share, cum in conc:
        acc_cases += share
        acc_cost = cum
        if cum >= 0.5:
            break
    if variants_apply and not tail_fired and acc_cost >= 0.5 and 0 < acc_cases <= 0.25:
        out.append(Finding(
            kind="concentration",
            title=f"{acc_cases:.0%} of runs account for {acc_cost:.0%} of spend",
            detail=("Cost is concentrated in a narrow class of trajectories. "
                    "Optimising the average request is pointless — fix these paths."),
            share=acc_cost,
            evidence=[f"most expensive path: {' → '.join(conc[0][0].seq[:6])}"],
        ))

    if tail_fired:
        out.append(Finding(
            kind="tail",
            title=f"Rare paths ({tail_share:.1%} of runs) eat {tail_cost / total:.0%} of the budget",
            detail=("Each occurs in under 5% of runs but costs twice the median run. "
                    "Aggregate dashboards cannot see paths like these at all."),
            impact_usd=tail_cost,
            share=tail_cost / total,
            # Обрезка пути до N шагов делала разные варианты неотличимыми в отчёте.
            # Показываем длину и середину пути, где они и расходятся.
            evidence=[f"{v.n}× · ${v.cost_avg:.3f}/run · {len(v.seq)} steps · "
                      f"{_path_digest(v.seq)}" for v in tail[:3]],
        ))

    # 4. Цена сбоев.
    err_cost, err_cases, kinds = _error_cost(events)
    if err_cases and err_cost / total > 0.05:
        top = ", ".join(f"{k} ×{n}" for k, n in kinds.most_common(3))
        out.append(Finding(
            kind="errors",
            title=f"Recovering from failures costs {err_cost / total:.0%} of the budget",
            detail=(f"{err_cases} of {m.n_cases} runs contain at least one error. What "
                    f"you pay for is not the error but the recovery: counted here are "
                    f"the failing step and the {RECOVERY_WINDOW} turns after it."),
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
                    title=f"`{name}`: {cs:.1%} of calls, {ss:.0%} of spend",
                    detail=("This resource is used rarely but costs disproportionately "
                            "much. Check whether escalating to it is justified on every "
                            "one of those paths."),
                    impact_usd=r["cost"],
                    share=ss,
                    evidence=[f"{r['n']:,} calls, {r['tokens']:,} tokens"],
                ))
                break

    # 6. Кеш, который не работает.
    tin = sum(e["tokens_in"] or 0 for e in events)
    tcached = sum(e["tokens_cached"] or 0 for e in events)
    if tin > 100_000 and tcached / (tin + tcached) < 0.05:
        out.append(Finding(
            kind="cache",
            title="Prompt cache almost never hits",
            detail=(f"Of {tin + tcached:,} input tokens, {tcached:,} came from cache "
                    f"({tcached / (tin + tcached):.1%}). The usual cause is an unstable "
                    "prefix: a changing tool set, a timestamp, or unsorted JSON in the "
                    "system prompt."),
            evidence=["the cache matches by prefix: any change invalidates the tail"],
        ))

    # 7. Раздувание контекста. Работает там, где вариантный анализ уже бесполезен:
    # чем длиннее прогоны, тем метрика значимее.
    infl = context_inflation(events, top=3)
    if infl and infl[0].est_usd / total > 0.03:
        c = infl[0]
        out.append(Finding(
            kind="context",
            title=f"Results of `{c.step}` carry {c.est_usd / total:.0%} of the budget through context",
            detail=(
                f"The step itself burns no tokens and shows as $0.00 in every cost "
                f"breakdown. But each of its results adds ~{c.added_avg:,.0f} tokens to "
                f"the prompt, and those are re-read on EVERY later turn: "
                f"{c.carried_tokens / 1e6:,.1f}M tokens in total. Fix by truncating the "
                f"output, not by switching models."
            ),
            impact_usd=c.est_usd,
            share=c.est_usd / total,
            evidence=[f"{x.step}: {x.n:,}× · +{x.added_avg:,.0f} tok · ≈${x.est_usd:,.2f}"
                      for x in infl],
        ))

    # 8. Честный отказ. Инструмент, который на любых данных выдаёт пять находок,
    # рано или поздно выдаёт пять выдуманных. Сказать «эта линза не подходит вашему
    # логу и вот почему» — полезнее, чем тавтология с большим числом долларов.
    if not variants_apply:
        med = sorted(len(v.seq) for v in m.variants)[len(m.variants) // 2] if m.variants else 0
        out.append(Finding(
            kind="not_applicable",
            title="Variant analysis does not apply to this log",
            detail=(
                f"Only {m.variant_reuse:.0%} of paths repeat: {len(m.variants)} paths "
                f"across {m.n_cases} runs, median run length {med} steps. Trajectories "
                f"that long never repeat, so conclusions like \"rare paths eat the "
                f"budget\" would be tautologies here and are suppressed. What helps: "
                f"coarser activities (`traceroutine abstract`), a narrower case notion "
                f"(`--case task`) — or metrics that do not depend on length, namely "
                f"context inflation and cohort comparison via `traceroutine diff`."
            ),
            evidence=[f"applicability threshold: repeated paths ≥ {VARIANT_MIN_REUSE:.0%}"],
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


def _carry_until(evs: list[dict], turns: list[int], prompt_size) -> list[int]:
    """До какого хода доживает контекст, добавленный в промежутке k.

    Раньше здесь стояло `len(turns) - k - 1` — «до конца кейса». На коротких
    прогонах это верно, на длинных — нет: агент компактит контекст, и всё, что
    добавлено до сброса, следующие ходы уже не несут и не оплачивают.

    Дефект был не косметическим и виден только на чужом логе. У первого внешнего
    пользователя прогоны по 1 225 шагов, сбросов в них множество, и график
    раздувания суммировался в 123% от всех расходов: доли превышали целое. На
    моём логе (case notion `task`, медиана 24 шага) падение размера промпта
    случилось ОДИН раз на 6 628 ходов, поэтому три сессии подряд метрика
    выглядела здоровой.

    Отсечка по любому падению промпта — намеренно консервативная: какая именно
    часть контекста ушла, из счётчиков не видно, поэтому после сброса мы просто
    перестаём что-либо утверждать. Заодно это даёт инвариант, который стоит
    беречь: внутри отрезка без сбросов сумма carried равна расходам на промпт
    минус базовый промпт, то есть **сумма долей больше целого не бывает**.
    """
    n = len(turns)
    out = [n - 1] * max(n - 1, 0)
    last = n - 1
    for k in range(n - 2, -1, -1):
        out[k] = last
        if prompt_size(evs[turns[k + 1]]) < prompt_size(evs[turns[k]]):
            last = k
    return out


def context_inflation(events: list[dict], top: int = 6) -> list[ContextCost]:
    """Кто раздувает контекст — и во что это обходится на всей оставшейся траектории.

    Метрика, ради которой стоит смотреть на агента процессно. Вызов инструмента сам
    по себе не стоит ни одного токена: в разбивке по стоимости все инструменты ровно
    по $0.00, а 100% расходов висит на `chat`. Вывод «оптимизируйте chat» бесполезен.

    На деле результат инструмента попадает в промпт и **перечитывается на каждом
    последующем ходе**. Один `Read` большого файла на десятом шаге из сорока
    оплачивается тридцать раз. Это и есть стоимость ТРАЕКТОРИИ, а не запроса:
    цена шага определяется не им самим, а тем, сколько ходов после него осталось.

    Само явление известно: и вендоры, и работы 2026 года пишут, что перечитывание
    контекста — крупнейшая статья расходов агента, а входные токены дают свыше 99%
    объёма траектории. Чего не найдено ни у кого — это АТРИБУЦИЯ по конкретному
    шагу: не «контекст растёт», а «результаты вот этого инструмента унесли $83».
    Платформы рекомендуют счётчики токенов на каждом спане, то есть по вызову;
    здесь считается, сколько шаг стоил на всём ОСТАТКЕ траектории после себя.

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
        carry_until = _carry_until(evs, turns, prompt_size)
        for k, (i, j) in enumerate(zip(turns, turns[1:])):
            delta = prompt_size(evs[j]) - prompt_size(evs[i])
            if delta <= 0:
                continue                     # компакция или сброс контекста
            between = [evs[x]["activity"] for x in range(i + 1, j)] or [evs[i]["activity"]]
            remaining = carry_until[k] - k
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
