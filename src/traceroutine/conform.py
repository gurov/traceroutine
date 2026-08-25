"""Слой Conformance: не «что агент делал», а «делал ли он то, что задумано».

Зачем это отдельный слой, а не ещё одна метрика. Проверка на реальных данных
(Ц3.5) вскрыла границу всей остальной аналитики: **находка process mining
осмысленна только относительно ожидания.** На синтетике ожидание было зашито
в генератор неявно, поэтому отклонения от него бросались в глаза. У открытого
агента — кодового, исследовательского — заявленного happy path нет вообще, и
любая находка вырождается в описание нормы: «агент часто вызывает Bash» верно
и бесполезно. Объявленная процессная модель — это и есть недостающее ожидание.

Поэтому conformance здесь не «фича под AI Act», а условие работоспособности
остального: сначала объявляешь, как должно быть, потом всё прочее обретает смысл.

**Две нотации, и обе нужны.**

Императивная (`flow`) отвечает на вопрос «насколько мы вообще далеко от дизайна»
одним числом — fitness. Классика PM: выравнивание (alignment) трассы с языком
модели, цена выравнивания и есть расстояние.

Декларативная (`rules`) отвечает на вопрос «какое конкретно обещание нарушено и
где». Для агентов она важнее, и вот почему: агент недетерминирован по устройству.
Жёсткая императивная модель на реальном логе даёт fitness около нуля и никакой
информации — «всё плохо». А ожидание от агента формулируется не как «ровно такая
последовательность», а как «что бы ты ни делал, не действуй без проверки» и
«закончи ответом». Это ровно словарь DECLARE (Pesic & van der Aalst), и он ложится
на агентов лучше, чем сети Петри.

**Почему не pm4py, хотя PLAN обещал его.** Три причины. (1) Он тянет
pandas+scipy+networkx — это десятки мегабайт в дереве зависимостей, а Ц5 требует
`uvx traceroutine` без установки; цена слишком велика за одну функцию. (2) Наш язык
модели — регулярное выражение над активностями, а не произвольная сеть Петри:
для него выравнивание точное, а не приближённое, и укладывается в 0-1 BFS на
полсотни строк. Вся сложность алгоритмики pm4py живёт в неограниченных сетях,
которых у нас нет. (3) Формула fitness взята оттуда же и совпадает с эталонной:
`1 − cost / (|trace| + |кратчайший прогон модели|)`.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .abstract import canonical
from .analyze import Finding
from .store import cases

# Джокер в `flow`: шаг, на месте которого допустима любая активность.
# `plan -> any* -> respond` — самое частое реальное ожидание от агента:
# середина свободна, края обязательны.
ANY = "\x00any"


class ConfigError(Exception):
    """Ошибка в process.yaml. Отдельный тип, чтобы CLI вернул код 2, а не 1:
    сломанный конфиг и провалившаяся проверка — разные события для CI."""


# --- модель процесса: разбор flow ------------------------------------------

_TOK = re.compile(r"\s*(->|[|()*+?]|[A-Za-z_][\w.:/-]*)")


@dataclass
class NFA:
    """Автомат Томпсона. eps — переходы без чтения, sym — по активности."""
    eps: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    sym: dict[int, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    start: int = 0
    accept: int = 0
    n: int = 0
    _clos: dict[int, frozenset[int]] = field(default_factory=dict)
    _moves: dict[int, tuple] = field(default_factory=dict)
    _exp: dict[int, str] = field(default_factory=dict)

    def state(self) -> int:
        self.n += 1
        return self.n - 1

    def alphabet(self) -> set[str]:
        return {a for outs in self.sym.values() for a, _ in outs if a != ANY}

    def closure(self, q: int) -> frozenset[int]:
        """Состояния, достижимые из q без чтения. Считаем замыкание явно, а не
        ходим по eps отдельными шагами выравнивания: иначе в момент пропуска шага
        автомат уже стоит ВНУТРИ одной ветки альтернативы и точка выбора теряется."""
        if q in self._clos:
            return self._clos[q]
        seen, stack = {q}, [q]
        while stack:
            for r in self.eps[stack.pop()]:
                if r not in seen:
                    seen.add(r)
                    stack.append(r)
        self._clos[q] = frozenset(seen)
        return self._clos[q]

    def moves(self, q: int) -> tuple[tuple[str, int], ...]:
        if q not in self._moves:
            self._moves[q] = tuple({(a, r) for x in self.closure(q) for a, r in self.sym[x]})
        return self._moves[q]

    def expected(self, q: int) -> str:
        """Что модель допускала в этой точке — ВСЕ варианты, а не один из них.

        Найдено на реальном логе. Для `(Read | Grep | Glob)+` выравниванию всё
        равно, какую ветку объявить пропущенной: цена одинакова. Отчёт при этом
        говорил «пропущен Glob» в 410 прогонах — про инструмент, которого в логе
        нет ни разу. Читатель идёт добавлять вызовы Glob, то есть чинит не то.
        Честная формулировка — «ожидался один из», и она же склеивает отклонения,
        одинаковые по смыслу, в одну строку вместо нескольких случайных.
        """
        if q not in self._exp:
            syms = sorted({"any step" if a == ANY else a for a, _ in self.moves(q)})
            # Семь альтернатив в строке таблицы нечитаемы; ключ отклонения от
            # обрезки не плывёт — он всё равно определяется точкой в модели.
            head = " | ".join(syms[:4])
            self._exp[q] = (f"{head} … +{len(syms) - 4} more" if len(syms) > 4
                            else head) or "end of process"
        return self._exp[q]

    def shortest_run(self) -> int:
        """Длина кратчайшего принимаемого слова — знаменатель в формуле fitness."""
        dist = {self.start: 0}
        dq = deque([self.start])
        while dq:
            q = dq.popleft()
            for r in self.eps[q]:
                if r not in dist:
                    dist[r] = dist[q]
                    dq.appendleft(r)
            for _a, r in self.sym[q]:
                if r not in dist or dist[r] > dist[q] + 1:
                    dist[r] = dist[q] + 1
                    dq.append(r)
        return dist.get(self.accept, 0)


def parse_flow(expr: str) -> NFA:
    """`plan -> gather* -> (act -> verify)+ -> respond` → NFA.

    Грамматика намеренно крошечная: последовательность `->`, выбор `|`, скобки,
    и три квантора `* + ?`. Всё, что сложнее, читатель yaml уже не удержит в
    голове, а непонятая модель хуже отсутствующей — она даёт ложную уверенность.
    """
    toks: list[str] = []
    pos = 0
    while pos < len(expr):
        m = _TOK.match(expr, pos)
        if not m:
            if expr[pos:].strip():
                raise ConfigError(f"flow: cannot parse at position {pos}: {expr[pos:pos + 20]!r}")
            break
        toks.append(m.group(1))
        pos = m.end()

    nfa = NFA()
    i = 0

    def peek() -> str | None:
        return toks[i] if i < len(toks) else None

    def atom() -> tuple[int, int]:
        nonlocal i
        t = peek()
        if t is None:
            raise ConfigError("flow: expression ends unexpectedly")
        if t == "(":
            i += 1
            frag = alt()
            if peek() != ")":
                raise ConfigError("flow: unclosed parenthesis")
            i += 1
            return frag
        if t in ("|", ")", "->", "*", "+", "?"):
            raise ConfigError(f"flow: unexpected {t!r}")
        i += 1
        s, e = nfa.state(), nfa.state()
        nfa.sym[s].append((ANY if t == "any" else t, e))
        return s, e

    def rep() -> tuple[int, int]:
        nonlocal i
        s, e = atom()
        while peek() in ("*", "+", "?"):
            q = toks[i]
            i += 1
            ns, ne = nfa.state(), nfa.state()
            nfa.eps[ns].append(s)
            nfa.eps[e].append(ne)
            if q in ("*", "?"):
                nfa.eps[ns].append(ne)          # ноль повторов
            if q in ("*", "+"):
                nfa.eps[e].append(s)            # ещё повтор
            s, e = ns, ne
        return s, e

    def seq() -> tuple[int, int]:
        nonlocal i
        s, e = rep()
        while peek() == "->":
            i += 1
            s2, e2 = rep()
            nfa.eps[e].append(s2)
            e = e2
        return s, e

    def alt() -> tuple[int, int]:
        nonlocal i
        branches = [seq()]
        while peek() == "|":
            i += 1
            branches.append(seq())
        if len(branches) == 1:
            return branches[0]
        s, e = nfa.state(), nfa.state()
        for bs, be in branches:
            nfa.eps[s].append(bs)
            nfa.eps[be].append(e)
        return s, e

    nfa.start, nfa.accept = alt()
    if i != len(toks):
        raise ConfigError(f"flow: trailing input after expression: {' '.join(toks[i:])!r}")
    return nfa


# --- выравнивание -----------------------------------------------------------

@dataclass
class Move:
    kind: str            # sync | log | model
    activity: str
    index: int           # позиция в трассе; для model-хода — куда он вклинился


def align(nfa: NFA, trace: list[str]) -> tuple[int, list[Move]]:
    """Минимальная цена привести трассу к языку модели + сами ходы.

    Три вида ходов, как в каноническом alignment:
      sync  — модель и лог согласны, цена 0;
      log   — агент сделал шаг, которого модель не предусматривала (цена 1);
      model — модель требовала шаг, которого агент не сделал (цена 1).

    Веса 0/1 позволяют обойтись 0-1 BFS вместо A* — точность та же, кода втрое
    меньше. Узлов (|трасса|+1)×|состояний|, на реальных прогонах это тысячи.
    """
    n = len(trace)
    start = (0, nfa.start)
    dist: dict[tuple[int, int], int] = {start: 0}
    prev: dict[tuple[int, int], tuple[tuple[int, int], Move | None]] = {}
    dq: deque[tuple[int, int]] = deque([start])
    goal: tuple[int, int] | None = None

    while dq:
        cur = dq.popleft()
        i, q = cur
        if i == n and nfa.accept in nfa.closure(q):
            goal = cur
            break
        d = dist[cur]

        def relax(nxt: tuple[int, int], w: int, mv: Move | None) -> None:
            if nxt not in dist or dist[nxt] > d + w:
                dist[nxt] = d + w
                prev[nxt] = (cur, mv)
                (dq.appendleft if w == 0 else dq.append)(nxt)

        for a, r in nfa.moves(q):
            if i < n and (a == ANY or a == trace[i]):
                relax((i + 1, r), 0, Move("sync", trace[i], i))
            # Пропуск шага модели: агент до него не дошёл.
            relax((i, r), 1, Move("model", nfa.expected(q), i))
        if i < n:
            relax((i + 1, q), 1, Move("log", trace[i], i))

    if goal is None:                          # пустой язык модели — не бывает, но
        return n, [Move("log", a, k) for k, a in enumerate(trace)]

    moves: list[Move] = []
    node = goal
    while node != start:
        node, mv = prev[node]
        if mv is not None:
            moves.append(mv)
    moves.reverse()
    return dist[goal], moves


# --- декларативные правила --------------------------------------------------

_RULE_KINDS = ("always", "never", "first", "last", "after", "before", "forbid", "max")

# Правила, где нарушение — это лишний ШАГ: его стоимость известна поимённо.
# У остальных нарушение в том, чего НЕ произошло, и приписывать ему деньги значило
# бы повторить ошибку атрибуции сбоев (analyze.RECOVERY_WINDOW).
_COSTED = ("never", "forbid", "max")


@dataclass
class Rule:
    kind: str
    a: str
    b: str = ""
    n: int = 0
    allow: float = 0.0       # допустимая доля кейсов-нарушителей (агенты не бывают идеальны)
    warn: bool = False       # нарушение не валит CI, но попадает в отчёт

    @property
    def text(self) -> str:
        return {
            "always": f"`{self.a}` must occur",
            "never":  f"`{self.a}` must never occur",
            "first":  f"a run starts with `{self.a}`",
            "last":   f"a run ends with `{self.a}`",
            "after":  f"after `{self.a}`, `{self.b}` must eventually follow",
            "before": f"before `{self.a}`, `{self.b}` must have happened",
            "forbid": f"`{self.b}` must not come directly after `{self.a}`",
            "max":    f"`{self.a}` at most {self.n}× per run",
        }[self.kind]

    def activities(self) -> set[str]:
        return {x for x in (self.a, self.b) if x}


@dataclass
class RuleStat:
    rule: Rule
    cases: int = 0            # в скольких прогонах нарушено
    events: int = 0           # сколько раз
    cost: float = 0.0         # деньги, которые честно приписываются нарушению
    examples: list[str] = field(default_factory=list)

    def share(self, n_cases: int) -> float:
        return self.cases / n_cases if n_cases else 0.0

    def ok(self, n_cases: int) -> bool:
        return self.share(n_cases) <= self.rule.allow


def check_rule(rule: Rule, seq: list[str], costs: list[float]) -> tuple[int, float]:
    """Нарушения правила в одном прогоне: (сколько, во сколько обошлись).

    Деньги приписываются УЗКО и только там, где это защитимо. `never` и `max` —
    лишние шаги видны поимённо, их стоимость и есть цена нарушения. `after`,
    `always`, `first`, `last` — нарушение в том, чего НЕ произошло; приписывать
    сюда стоимость всего прогона было бы тем же переусердствованием, на котором
    уже обожглась атрибуция сбоев (см. analyze.RECOVERY_WINDOW). Возвращаем 0 и
    честно говорим об этом в отчёте.
    """
    k = rule.kind
    if k == "always":
        return (0, 0.0) if rule.a in seq else (1, 0.0)
    if k == "never":
        idx = [i for i, a in enumerate(seq) if a == rule.a]
        return len(idx), sum(costs[i] for i in idx)
    if k == "first":
        return (0, 0.0) if seq and seq[0] == rule.a else (1, 0.0)
    if k == "last":
        return (0, 0.0) if seq and seq[-1] == rule.a else (1, 0.0)
    if k == "after":
        bad = [i for i, a in enumerate(seq) if a == rule.a and rule.b not in seq[i + 1:]]
        return len(bad), 0.0
    if k == "before":
        bad = [i for i, a in enumerate(seq) if a == rule.a and rule.b not in seq[:i]]
        return len(bad), 0.0
    if k == "forbid":
        idx = [i for i in range(1, len(seq)) if seq[i - 1] == rule.a and seq[i] == rule.b]
        return len(idx), sum(costs[i] for i in idx)
    if k == "max":
        idx = [i for i, a in enumerate(seq) if a == rule.a]
        extra = idx[rule.n:]
        return len(extra), sum(costs[i] for i in extra)
    raise ConfigError(f"unknown rule kind {k!r}")


# --- конфиг -----------------------------------------------------------------

@dataclass
class Thresholds:
    fitness_min: float | None = None
    usd_per_case_max: float | None = None
    steps_p95_max: float | None = None
    off_model_share_max: float | None = None
    # Относительно базового лога — то, что нужно в CI: абсолютный порог придётся
    # подкручивать после каждого релиза, а «не хуже, чем было» живёт само.
    usd_per_case_increase_max: float | None = None
    fitness_drop_max: float | None = None
    steps_increase_max: float | None = None

    BASELINE = ("usd_per_case_increase_max", "fitness_drop_max", "steps_increase_max")


_THRESHOLD_KEYS = set(Thresholds.__annotations__) - {"BASELINE"}


@dataclass
class Process:
    name: str = "process"
    flow: str | None = None
    rules: list[Rule] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)
    case: str | None = None            # ожидаемый case notion — сверяем и предупреждаем
    nfa: NFA | None = None

    def activities(self) -> set[str]:
        """Всё, что упомянуто, — для проверки «а есть ли это вообще в логе»."""
        acts = set(self.nfa.alphabet()) if self.nfa else set()
        for r in self.rules:
            acts |= r.activities()
        return acts

    def forbidden(self) -> set[str]:
        return {r.a for r in self.rules if r.kind == "never"}

    def allowed(self) -> set[str]:
        """Что модель РАЗРЕШАЕТ. Не то же, что «упомянуто»: активность из `never:`
        упомянута, но её присутствие — самое сильное отклонение, какое бывает."""
        return self.activities() - self.forbidden()

    @staticmethod
    def load(path: Path) -> "Process":
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: cannot parse yaml: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        return Process.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Process":
        known = {"name", "flow", "rules", "thresholds", "case"}
        # Опечатка в ключе иначе молча отключает половину проверки, а CI при этом
        # остаётся зелёным. Зелёный CI, который ничего не проверяет, — худший
        # возможный исход для этого инструмента.
        if unknown := set(data) - known:
            raise ConfigError(f"unknown keys: {', '.join(sorted(unknown))}; "
                              f"allowed: {', '.join(sorted(known))}")
        p = Process(name=str(data.get("name") or "process"), case=data.get("case"))
        if data.get("flow"):
            p.flow = str(data["flow"]).strip()
            p.nfa = parse_flow(p.flow)
        for raw in data.get("rules") or []:
            p.rules.append(_rule(raw))
        t = data.get("thresholds") or {}
        if not isinstance(t, dict):
            raise ConfigError("thresholds: expected a mapping")
        if unknown := set(t) - _THRESHOLD_KEYS:
            raise ConfigError(f"thresholds: unknown keys {', '.join(sorted(unknown))}; "
                              f"allowed: {', '.join(sorted(_THRESHOLD_KEYS))}")
        p.thresholds = Thresholds(**{k: float(v) for k, v in t.items()})
        if not p.flow and not p.rules:
            raise ConfigError("empty process: needs at least a flow or one rule")
        return p


# Допустимые формы правила. Список — часть контракта: он же печатается в ошибке,
# поэтому формат остаётся выучиваемым без документации.
_FORMS = (
    "{always: X}", "{never: X}", "{first: X}", "{last: X}",
    "{after: A, expect: B}", "{before: A, expect: B}",
    "{after: A, forbid: B}", "{max: N, of: X}",
)


def _rule(raw: Any) -> Rule:
    """Правило из yaml. Формы читаются вслух как само требование — это не
    косметика: правило, которое нельзя прочесть, не будет написано."""
    if not isinstance(raw, dict):
        raise ConfigError(f"a rule must be a mapping, got: {raw!r}")
    opts = {"allow": float(raw.get("allow") or 0.0), "warn": bool(raw.get("warn"))}
    body = {k: v for k, v in raw.items() if k not in ("allow", "warn")}
    keys = set(body)

    # Пары разбираются ДО одиночных: `{after: A, forbid: B}` содержит два слова из
    # словаря видов, и определять вид по «единственному известному ключу» здесь нельзя.
    if keys == {"after", "expect"}:
        return Rule("after", str(body["after"]), str(body["expect"]), **opts)
    if keys == {"before", "expect"}:
        return Rule("before", str(body["before"]), str(body["expect"]), **opts)
    if keys == {"after", "forbid"}:
        return Rule("forbid", str(body["after"]), str(body["forbid"]), **opts)
    if keys == {"max", "of"}:
        try:
            n = int(body["max"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"max must be a number: {raw!r}") from exc
        return Rule("max", str(body["of"]), n=n, **opts)
    if len(keys) == 1:
        (k,) = keys
        if k in ("always", "never", "first", "last"):
            return Rule(k, str(body[k]), **opts)

    raise ConfigError(f"unrecognised rule {raw!r}; forms: {', '.join(_FORMS)} "
                      f"(plus optional allow: and warn:)")


# --- результат --------------------------------------------------------------

@dataclass
class Deviation:
    kind: str            # log — лишний шаг; model — пропущенный
    activity: str
    n: int = 0
    cases: int = 0
    cost: float = 0.0


@dataclass
class CheckReport:
    process: Process
    n_cases: int = 0
    fitting: int = 0                 # прогонов, легших на модель без единого отклонения
    fitness: float | None = None
    deviations: list[Deviation] = field(default_factory=list)
    rules: list[RuleStat] = field(default_factory=list)
    off_model_cost: float = 0.0      # деньги на шагах, которых нет в дизайне
    total_cost: float = 0.0
    usd_per_case: float = 0.0
    steps_p95: float = 0.0
    unseen: list[str] = field(default_factory=list)   # объявлено, но ни разу не встретилось
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)        # про данные и конфиг
    rule_warnings: list[str] = field(default_factory=list)   # нарушенные правила с warn: true

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def off_model_share(self) -> float:
        return self.off_model_cost / self.total_cost if self.total_cost else 0.0

    @property
    def conforming_share(self) -> float:
        return self.fitting / self.n_cases if self.n_cases else 0.0


def _p95(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))])


def check(process: Process, events: list[dict],
          baseline: list[dict] | None = None) -> CheckReport:
    """Лог + объявленный процесс → карта отклонений и вердикт для CI.

    baseline — лог «как было». Сравнивается тем же процессом и через тот же
    `check`, а не через сырую модель: иначе `fitness_drop_max` было бы нечем
    посчитать, и объявленный в yaml порог молча ничего бы не проверял. Порог,
    который ничего не проверяет, хуже отсутствующего — он даёт ложную уверенность.
    """
    rep = CheckReport(process=process)
    stats = {id(r): RuleStat(rule=r) for r in process.rules}
    dev: dict[tuple[str, str], Deviation] = {}
    dev_cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    cost_sum = worst_sum = 0
    lengths: list[int] = []
    seen: set[str] = set()

    min_run = process.nfa.shortest_run() if process.nfa else 0

    for case_id, evs in cases(events):
        seq = [e["activity"] for e in evs]
        costs = [e["cost_usd"] or 0.0 for e in evs]
        rep.n_cases += 1
        rep.total_cost += sum(costs)
        lengths.append(len(seq))
        seen.update(seq)

        if process.nfa is not None:
            cost, moves = align(process.nfa, seq)
            cost_sum += cost
            worst_sum += len(seq) + min_run
            if cost == 0:
                rep.fitting += 1
            for mv in moves:
                if mv.kind == "sync":
                    continue
                key = (mv.kind, mv.activity)
                d = dev.setdefault(key, Deviation(mv.kind, mv.activity))
                d.n += 1
                if mv.kind == "log":
                    d.cost += costs[mv.index] if mv.index < len(costs) else 0.0
                dev_cases[key].add(case_id)

        for r in process.rules:
            n, money = check_rule(r, seq, costs)
            if n:
                st = stats[id(r)]
                st.cases += 1
                st.events += n
                st.cost += money
                if len(st.examples) < 3:
                    st.examples.append(case_id)

    for key, d in dev.items():
        d.cases = len(dev_cases[key])
    rep.deviations = sorted(dev.values(), key=lambda d: (-d.cost, -d.n))
    rep.off_model_cost = sum(d.cost for d in rep.deviations if d.kind == "log")
    rep.rules = [stats[id(r)] for r in process.rules]
    rep.usd_per_case = rep.total_cost / rep.n_cases if rep.n_cases else 0.0
    rep.steps_p95 = _p95(lengths)
    if process.nfa is not None:
        rep.fitness = 1 - cost_sum / worst_sum if worst_sum else 1.0

    # Активность объявлена, но в логе её нет ни разу. Почти всегда это опечатка
    # либо несовпадение уровня абстракции (в модели `act`, в логе `tool:Bash`) —
    # и тогда проверка формально проходит, ничего не проверив.
    rep.unseen = sorted(process.activities() - seen)
    _reject_level_mismatch(rep.unseen, seen)
    if rep.unseen:
        rep.warnings.append(
            f"declared but never seen in the log: {', '.join(rep.unseen)}. Check the "
            f"activity names and the vocabulary (`traceroutine abstract`) — otherwise "
            f"the check is green simply because it has nothing to compare against."
        )
    if process.case:
        rep.warnings.append(
            f"this process is declared for case notion `{process.case}` — make sure the "
            f"log was built with `--case {process.case}`."
        )

    _verdict(rep, check(process, baseline) if baseline is not None else None)
    return rep


def _reject_level_mismatch(unseen: list[str], seen: set[str]) -> None:
    """Модель и лог написаны на разных уровнях абстракции — это ошибка конфига.

    Тот самый разрыв, в который упирается всякий, кто идёт по README подряд:
    quickstart не звал `abstract`, поэтому в логе `chat:claude-opus-5`, а в
    модели `chat`. Проверка при этом не падает — она выдаёт стену красного,
    где ни одна строка не верна: `last: chat` нарушен в 100% прогонов, вне
    модели 100% бюджета, fitness 0.391 вместо 0.898.

    Отличаем от честного «активности не случилось» строго: сообщаем, только
    если в логе есть метка, которая канонизируется РОВНО в объявленную. Тогда
    сравнивать было не с чем не потому, что агент так себя вёл, а потому что
    имена не совпали, — и мы можем сказать, что именно набрать.
    """
    hints = []
    for a in unseen:
        found = sorted(s for s in seen if s != a and canonical(s) == a)
        if found:
            shown = ", ".join(f"`{f}`" for f in found[:3])
            hints.append(f"`{a}` — the log has {shown}"
                         + (f" and {len(found) - 3} more" if len(found) > 3 else ""))
    if not hints:
        return
    raise ConfigError(
        "process.yaml and the log are at different abstraction levels:\n  "
        + "\n  ".join(hints)
        + "\nNothing is compared against those names, so fitness and the deviation "
          "map would both be fiction. Build the vocabulary and pass it in:\n"
          "  traceroutine abstract <log> -m activity_map.yaml\n"
          "  traceroutine check <log> -p <process.yaml> -m activity_map.yaml"
    )


def _verdict(rep: CheckReport, base: "CheckReport | None") -> None:
    t = rep.process.thresholds
    if t.fitness_min is not None and rep.fitness is not None and rep.fitness < t.fitness_min:
        rep.failures.append(f"fitness {rep.fitness:.3f} < {t.fitness_min:.3f}")
    if t.usd_per_case_max is not None and rep.usd_per_case > t.usd_per_case_max:
        rep.failures.append(
            f"${rep.usd_per_case:.4f} per run > ${t.usd_per_case_max:.4f}")
    if t.steps_p95_max is not None and rep.steps_p95 > t.steps_p95_max:
        rep.failures.append(f"run length p95 {rep.steps_p95:.0f} > {t.steps_p95_max:.0f}")
    if t.off_model_share_max is not None and rep.off_model_share > t.off_model_share_max:
        rep.failures.append(
            f"off-model spend {rep.off_model_share:.1%} > {t.off_model_share_max:.1%}")

    for st in rep.rules:
        if st.ok(rep.n_cases):
            continue
        msg = (f"{st.rule.text}: violated in {st.cases} runs "
               f"({st.share(rep.n_cases):.1%}"
               + (f", allowed {st.rule.allow:.1%}" if st.rule.allow else "") + ")")
        (rep.rule_warnings if st.rule.warn else rep.failures).append(msg)

    if base is None:
        return
    if t.usd_per_case_increase_max is not None and base.usd_per_case:
        grow = (rep.usd_per_case - base.usd_per_case) / base.usd_per_case
        if grow > t.usd_per_case_increase_max:
            rep.failures.append(
                f"cost per run grew {grow:+.1%} "
                f"(${base.usd_per_case:.4f} → ${rep.usd_per_case:.4f}), "
                f"allowed {t.usd_per_case_increase_max:+.1%}")
    if t.steps_increase_max is not None and base.steps_p95:
        grow = (rep.steps_p95 - base.steps_p95) / base.steps_p95
        if grow > t.steps_increase_max:
            rep.failures.append(
                f"run length p95 grew {grow:+.1%} "
                f"({base.steps_p95:.0f} → {rep.steps_p95:.0f}), "
                f"allowed {t.steps_increase_max:+.1%}")
    if (t.fitness_drop_max is not None and rep.fitness is not None
            and base.fitness is not None):
        drop = base.fitness - rep.fitness
        if drop > t.fitness_drop_max:
            rep.failures.append(
                f"fitness dropped by {drop:.3f} "
                f"({base.fitness:.3f} → {rep.fitness:.3f}), "
                f"allowed {t.fitness_drop_max:.3f}")


def findings(rep: CheckReport, limit: int = 3) -> list[Finding]:
    """Conformance → находки для общего отчёта.

    Именно ради этого слой и делался: остальные находки описывают лог сам по себе,
    а эти — расхождение с намерением. На открытых агентах только они и не
    вырождаются в описание нормы.
    """
    out: list[Finding] = []
    if rep.fitness is not None and rep.n_cases:
        broke = rep.n_cases - rep.fitting
        if broke:
            top = [d for d in rep.deviations if d.kind == "log"][:3]
            skipped = [d for d in rep.deviations if d.kind == "model"][:3]
            extra = sum(d.n for d in rep.deviations if d.kind == "log")
            trivial = bool(extra) and rep.off_model_share < 0.01
            if trivial:
                # Ноль здесь легко прочесть как «отклонения бесплатны». Это неверно:
                # вне модели почти всегда оказываются вызовы инструментов, а они не
                # тратят токенов сами — их цена в контексте, который они несут
                # дальше по траектории.
                money = (f"Their direct cost is near zero (${rep.off_model_cost:,.2f}) "
                         f"— these are tool calls and they burn no tokens themselves. "
                         f"Their real price is context: each result is re-read on every "
                         f"later turn.")
            else:
                money = (f"Steps that are not in the design cost "
                         f"${rep.off_model_cost:,.2f} ({rep.off_model_share:.0%} of the "
                         f"budget) — not an estimate, but the summed cost of those steps.")
            out.append(Finding(
                kind="conformance",
                title=(f"{broke / rep.n_cases:.0%} of runs deviate from the declared "
                       f"process \"{rep.process.name}\""),
                detail=(
                    f"fitness {rep.fitness:.2f}: {rep.fitting} of {rep.n_cases} runs fit "
                    f"the model with no deviation at all. {extra} steps off-model. {money}"
                ),
                # Ярлык «до $1.05» рядом с «96% прогонов отклоняются» обесценивает
                # находку сильнее, чем его отсутствие. Денег нет — не называем сумму.
                impact_usd=0.0 if trivial else rep.off_model_cost,
                share=rep.off_model_share,
                evidence=([f"extra step `{d.activity}`: {d.n}× in {d.cases} runs"
                           + (f", ${d.cost:,.2f}" if d.cost else "") for d in top]
                          + [f"missing `{d.activity}`: {d.cases} runs"
                             for d in skipped]),
            ))
    for st in sorted(rep.rules, key=lambda s: (-s.cost, -s.cases)):
        if not st.cases or st.ok(rep.n_cases):
            continue
        if st.rule.kind not in _COSTED:
            money = (" The violation is something that did NOT happen, so no money is "
                     "attributed to it.")
        elif st.cost:
            money = f" The offending steps themselves cost ${st.cost:,.2f}."
        else:
            # Ноль здесь не значит «бесплатно»: вызов инструмента не тратит токенов
            # сам, но его результат перечитывается на каждом следующем ходе.
            money = (" These steps burn no tokens themselves; their price is the "
                     "context they add for the rest of the run.")
        out.append(Finding(
            kind="rule",
            title=f"Rule violated: {st.rule.text}",
            detail=(f"{st.cases} of {rep.n_cases} runs ({st.share(rep.n_cases):.1%}), "
                    f"{st.events} occurrences." + money),
            impact_usd=st.cost,
            share=st.cost / rep.total_cost if rep.total_cost else 0.0,
            evidence=[f"for example: {', '.join(st.examples)}"] if st.examples else [],
        ))
    return out[:limit]
