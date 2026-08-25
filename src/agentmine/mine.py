"""DFG, варианты, циклы, rework.

Это горячий путь, поэтому он свой, а не через pm4py: pm4py подключается позже
(Ц6) для inductive miner и alignments, где важна корректность, а не скорость.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .store import cases

START, END = "▶ start", "■ end"


@dataclass
class Variant:
    seq: tuple[str, ...]
    n: int = 0
    cost: float = 0.0
    duration: float = 0.0
    errors: int = 0
    example_case: str = ""

    @property
    def cost_avg(self) -> float:
        return self.cost / self.n if self.n else 0.0


@dataclass
class Edge:
    n: int = 0
    cost: float = 0.0       # стоимость целевой активности, накопленная по этому переходу
    duration: float = 0.0


@dataclass
class Model:
    variants: list[Variant] = field(default_factory=list)
    edges: dict[tuple[str, str], Edge] = field(default_factory=dict)
    nodes: dict[str, dict] = field(default_factory=dict)
    n_cases: int = 0
    n_events: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    rework_events: int = 0
    loop_cost: dict[str, float] = field(default_factory=dict)
    # Модель — это РЕСУРС, а не активность (слой Abstract схлопывает chat:<model> → chat).
    # Разбивка по ресурсам возвращает потерянное: видно, что дорого не «обращение к
    # модели», а конкретно эскалация на большую.
    resource_cost: dict[str, dict] = field(default_factory=dict)

    @property
    def rework_rate(self) -> float:
        return self.rework_events / self.n_events if self.n_events else 0.0

    @property
    def variant_reuse(self) -> float:
        """Доля кейсов, чей путь встречается не в единственном экземпляре.

        Проверка применимости вариантного анализа, а не украшение. Варианты — это
        классы эквивалентности траекторий, и они осмысленны, только если траектории
        РЕАЛЬНО повторяются. На длинных прогонах они не повторяются никогда: на
        реальных транскриптах Claude Code уникальность путей растёт с 17% при 1–3
        шагах до 100% при 26+. Медиана там 15 шагов, то есть вариантный анализ
        применим примерно наполовину, а при 26+ шагах вырождается полностью —
        каждый кейс становится своим «вариантом», и вывод «редкие пути съедают
        бюджет» превращается в тавтологию «дорогие прогоны стоят дорого».
        """
        if not self.n_cases:
            return 0.0
        repeated = sum(v.n for v in self.variants if v.n > 1)
        return repeated / self.n_cases

    def top_variants(self, k: int = 20) -> list[Variant]:
        return sorted(self.variants, key=lambda v: -v.cost)[:k]

    def cost_concentration(self) -> list[tuple[Variant, float, float]]:
        """Парето: (вариант, доля кейсов, накопленная доля стоимости).

        Сортировка по стоимости ОДНОГО кейса, а не по суммарной: вопрос
        «какая доля прогонов съедает 80% счёта» — это про дорогие прогоны,
        а не про частые.
        """
        ordered = sorted(self.variants, key=lambda v: -v.cost_avg)
        out, acc = [], 0.0
        for v in ordered:
            acc += v.cost
            share = v.n / self.n_cases if self.n_cases else 0.0
            out.append((v, share, acc / self.total_cost if self.total_cost else 0.0))
        return out


def mine(events: list[dict]) -> Model:
    m = Model()
    vmap: dict[tuple[str, ...], Variant] = {}
    node_stat: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "cost": 0.0, "duration": 0.0, "errors": 0, "tokens": 0}
    )
    res_stat: dict[str, dict] = defaultdict(lambda: {"n": 0, "cost": 0.0, "tokens": 0})

    for case_id, evs in cases(events):
        m.n_cases += 1
        seq = tuple(e["activity"] for e in evs)
        seen: Counter[str] = Counter()
        c_cost = c_dur = 0.0
        c_err = 0

        for ev in evs:
            act = ev["activity"]
            dur = (ev["ts_end"] - ev["ts_start"]) if ev["ts_end"] else 0.0
            tok = (ev["tokens_in"] or 0) + (ev["tokens_cached"] or 0) + (ev["tokens_out"] or 0)
            cost = ev["cost_usd"] or 0.0

            s = node_stat[act]
            s["n"] += 1
            s["cost"] += cost
            s["duration"] += dur
            s["tokens"] += tok
            if ev["status"] != "ok":
                s["errors"] += 1
                c_err += 1

            if ev["resource"] and cost > 0:
                r = res_stat[ev["resource"]]
                r["n"] += 1
                r["cost"] += cost
                r["tokens"] += tok

            seen[act] += 1
            if seen[act] > 1:               # активность уже встречалась в этом кейсе
                m.rework_events += 1
                m.loop_cost[act] = m.loop_cost.get(act, 0.0) + cost

            c_cost += cost
            c_dur += dur
            m.n_events += 1
            m.total_tokens += tok

        m.total_cost += c_cost

        # seq.index(b) находил ПЕРВОЕ вхождение активности — при циклах стоимость
        # ребра приписывалась не тому событию. Идём по событиям напрямую.
        for i, (a, b) in enumerate(zip((START,) + seq, seq + (END,))):
            e = m.edges.setdefault((a, b), Edge())
            e.n += 1
            if b != END:
                e.cost += evs[i]["cost_usd"] or 0.0
                e.duration += (evs[i]["ts_end"] - evs[i]["ts_start"]) if evs[i]["ts_end"] else 0.0

        v = vmap.get(seq)
        if v is None:
            v = vmap[seq] = Variant(seq=seq, example_case=case_id)
        v.n += 1
        v.cost += c_cost
        v.duration += c_dur
        v.errors += c_err

    m.variants = list(vmap.values())
    m.nodes = dict(node_stat)
    m.resource_cost = dict(res_stat)
    return m
