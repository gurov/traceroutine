"""Рендер отчёта: markdown+mermaid и самодостаточный HTML.

Про mermaid. Он НЕ заменяет интерактивный граф (Ц4) — flowchart разваливается
уже на ~30-40 узлах, не показывает веса рёбер и не масштабируется. Но у него есть
слот, который canvas закрыть не может: markdown, который рендерится сам —
GitHub Actions job summary, README, PR-комментарий, Slack, Notion.
Поэтому mermaid-экспорт всегда идёт через фильтрацию до «хребта» процесса.

Побочная польза: фильтрация по порогу — это ровно тот «слайдер абстракции»,
который всё равно нужен основному отчёту.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from .analyze import Drift, Finding, _path_digest
from .conform import CheckReport
from .mine import END, START, Model


# Отчёт, который заканчивается числом, — разовое любопытство. Число само по себе
# ничего не требует сделать: посмотрел, удивился, закрыл. Повторно инструмент
# открывают только те, у кого есть ПОВТОРЯЮЩИЙСЯ триггер, и таких триггера два —
# правка промпта и релиз модели. Поэтому отчёт обязан заканчиваться не итогом,
# а следующим шагом.
NEXT_TITLE = "What next"
NEXT_LEAD = (
    "Everything above is a snapshot of one moment. It will move on the next prompt "
    "edit or model swap, and finding that out from the invoice is the expensive way."
)
NEXT_STEPS = [
    ("traceroutine diff before.parquet after.parquet",
     "what changed in behaviour: new paths, longer runs, cost per task"),
    ("traceroutine check events.parquet -p process.yaml",
     "keep regressions out of main: CI exit codes, a summary in the job summary"),
]
NEXT_TAIL = (
    "`process.yaml` is the declared process: how the agent is SUPPOSED to work. "
    "Without it the findings above describe the norm; with it they describe a "
    "deviation from intent. That is the difference between \"the agent calls Bash a "
    "lot\" and \"21% of tasks edit a file without reading it first\"."
)


def _cell(s: str) -> str:
    """Вертикальная черта режет ячейку markdown-таблицы даже внутри `code`.
    А в отчёте она законна: так записана альтернатива в самом process.yaml."""
    return s.replace("|", "\\|")


def _usd(v: float) -> str:
    return f"${v:,.2f}" if v >= 0.01 else f"${v:.4f}"


def _node_label(activity: str, stat: dict) -> tuple[str, str]:
    """Подпись узла: имя и вторая строка.

    Деньги не печатаются, когда их РОВНО ноль. `_usd` даёт `$0.0000` для всего
    мельче цента, и на графе агента это семь узлов из восьми: инструменты в
    момент вызова не тратят токенов. Формально верно, читается как «инструмент
    сломан» — и хоронит саму находку, ради которой граф и рисуют. Ноль на шаге
    не значит «бесплатно», он значит «плата придёт позже», и об этом говорит
    строка под графом, а не четыре нуля после точки.
    """
    n = f"{stat['n']:,}×"
    return activity, (f"{n} · {_usd(stat['cost'])}" if stat["cost"] else n)


# Подпись под графом. Появляется только когда бесплатные шаги там есть, иначе
# это шум. Без неё узел без суммы читается как «этот шаг ничего не стоит» —
# ровно то заблуждение, против которого инструмент и написан.
FREE_STEPS_NOTE = (
    "Steps shown without a figure spend no tokens at the moment of the call. "
    "That is not the same as free: their results stay in the prompt and are "
    "re-read on every later turn — see context inflation above."
)


# Транскрипты не содержат биллинга — только счётчики токенов. Деньги мы СЧИТАЕМ
# по прайс-листу, и на подписке это не счёт, а теневая цена. Без этой оговорки
# первый же пользователь на Pro за $20 видит «$788» и делает единственно
# разумный вывод: инструмент врёт. Число при этом полезное — оно даёт общую
# линейку, по которой сравнимы пути; врёт не оно, а слово «потрачено».
PRICING_NOTE_HEAD = "This is not a bill."
PRICING_NOTE = (
    "Transcripts record tokens, not charges — on a subscription you paid your flat fee "
    "instead. {amount} is what these tokens would cost at API list prices over the "
    "{days} above, and it is here as a common ruler: it makes trajectories comparable "
    "with each other."
)


def _period(m: Model) -> tuple[str, str]:
    """Диапазон лога и его длина. Сумма без периода не значит ничего.

    «$813» читается как «за сегодня», «за месяц» и «за всё время» одинаково —
    это три разных вывода. Дата в шапке была временем ГЕНЕРАЦИИ отчёта и только
    усиливала путаницу: читатель видел сегодняшнее число рядом с большой суммой.
    """
    if not m.t_min:
        return "", "log"
    a = datetime.fromtimestamp(m.t_min, timezone.utc)
    b = datetime.fromtimestamp(m.t_max, timezone.utc)
    days = max(1, round((m.t_max - m.t_min) / 86400))
    same_year = a.year == b.year
    left = f"{a.day} {a:%b}" + ("" if same_year else f" {a.year}")
    return f"{left} – {b.day} {b:%b} {b.year}", f"{days} day" + ("s" if days != 1 else "")


def _spine(m: Model, max_nodes: int, min_edge: float) -> tuple[list[str], dict]:
    """Хребет процесса: top-N узлов по стоимости + рёбра выше порога частоты."""
    ranked = sorted(m.nodes.items(), key=lambda kv: -kv[1]["cost"])
    keep = {a for a, _ in ranked[:max_nodes]}
    max_n = max((e.n for e in m.edges.values()), default=1)
    edges = {
        (a, b): e
        for (a, b), e in m.edges.items()
        if (a in keep or a == START) and (b in keep or b == END) and e.n / max_n >= min_edge
    }
    # Узел прошёл по стоимости, но все его рёбра отсеялись по частоте — на графе
    # это коробка, висящая в воздухе. На синтетике не встречается, на реальном
    # логе Claude Code таких оказалось двенадцать из восемнадцати: у агента
    # длинный хвост инструментов, вызываемых считаные разы. Их место — в таблице
    # расходов по ресурсам, а не в схеме процесса.
    linked = {a for e in edges for a in e}
    order = [a for a, _ in ranked if a in keep and a in linked]
    return order, edges


def render_markdown(m: Model, *, max_nodes: int = 25, min_edge: float = 0.02,
                    found: list[Finding] | None = None) -> str:
    order, edges = _spine(m, max_nodes, min_edge)
    ids = {START: "S", END: "E"}
    for i, a in enumerate(order):
        ids[a] = f"n{i}"

    L = [
        "# traceroutine — agent process report",
        "",
        f"_{_period(m)[0] or 'log'} · {_period(m)[1]} · "
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_",
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Cases | {m.n_cases:,} |",
        f"| Events | {m.n_events:,} |",
        f"| Distinct paths | {len(m.variants):,} |",
        f"| Tokens | {m.total_tokens:,} |",
        f"| Cost (API list prices) | {_usd(m.total_cost)} |",
        f"| Rework rate | {m.rework_rate:.1%} |",
        "",
        f"**{PRICING_NOTE_HEAD}** "
        + PRICING_NOTE.format(amount=_usd(m.total_cost), days=_period(m)[1]),
        "",
    ]

    # Вывод про концентрацию печатается только если концентрация ЕСТЬ.
    # Отчёт, объявляющий «90% кейсов дают 80% расходов», разрушает доверие
    # к остальным цифрам: это ровное распределение, а не находка.
    # Находки идут ПЕРЕД графиками: дашборд — витамин, находка — обезболивающее.
    if found:
        L += ["## What to fix", "",
              "_Savings estimates are deliberately conservative and **may overlap**: one "
              "expensive run lands in both \"rare paths\" and \"escalation\". Do not add "
              "them up._", ""]
        for i, f in enumerate(found, 1):
            head = f"**{i}. {f.title}**"
            if f.impact_usd:
                head += f" — up to {_usd(f.impact_usd)} ({f.share:.0%} of budget)"
            L += [head, "", f.detail, ""]
            for e in f.evidence:
                L.append(f"- {e}")
            L.append("")

    L += [f"## {NEXT_TITLE}", "", NEXT_LEAD, "", "```bash"]
    for cmd, why in NEXT_STEPS:
        L += [f"{cmd}", f"#   {why}"]
    L += ["```", "", NEXT_TAIL, ""]

    conc = m.cost_concentration()
    if conc and m.total_cost > 0:
        acc_cases = 0.0
        for v, share, acc_cost in conc:
            acc_cases += share
            if acc_cost >= 0.8:
                if acc_cases <= 0.35:
                    L += [
                        f"> **{acc_cases:.0%} of runs account for 80% of spend.** "
                        f"Cost is a property of the trajectory, not of the request.",
                        "",
                    ]
                break

    L += ["## Process graph", "", "```mermaid", "flowchart TD"]
    L.append(f'    {ids[START]}(["▶ start"])')
    for a in order:
        name, second = _node_label(a, m.nodes[a])
        L.append(f'    {ids[a]}["{_mm(name)}<br/>{_mm(second)}"]')
    L.append(f'    {ids[END]}(["■ end"])')
    for (a, b), e in sorted(edges.items(), key=lambda kv: -kv[1].n):
        if a in ids and b in ids:
            L.append(f"    {ids[a]} -->|{e.n}| {ids[b]}")
    # Раскраска по доле стоимости — читается и в светлой, и в тёмной теме GitHub.
    hot = [a for a in order if m.total_cost and m.nodes[a]["cost"] / m.total_cost > 0.15]
    err = [a for a in order if m.nodes[a]["errors"]]
    L.append("    classDef hot fill:#b4322e,stroke:#7d1f1c,color:#fff")
    L.append("    classDef err stroke:#d97706,stroke-width:3px")
    if hot:
        L.append(f"    class {','.join(ids[a] for a in hot)} hot")
    if err:
        L.append(f"    class {','.join(ids[a] for a in err)} err")
    L += ["```", ""]
    if any(not m.nodes[a]["cost"] for a in order):
        L += [f"_{FREE_STEPS_NOTE}_", ""]

    if len(m.nodes) > max_nodes:
        L += [
            f"_Showing {max_nodes} of {len(m.nodes)} activities (mermaid is unreadable "
            f"beyond that). The full graph is in the HTML report._",
            "",
        ]

    L += ["## Most expensive paths", "",
          "| Share of cases | Total | Average | Steps | Errors | Path |",
          "|---:|---:|---:|---:|---:|---|"]
    for v in m.top_variants(10):
        share = v.n / m.n_cases if m.n_cases else 0
        path = " → ".join(v.seq[:6]) + (" → …" if len(v.seq) > 6 else "")
        L.append(
            f"| {share:.1%} ({v.n}) | {_usd(v.cost)} | {_usd(v.cost_avg)} | "
            f"{len(v.seq)} | {v.errors} | `{path}` |"
        )
    L.append("")

    if m.resource_cost:
        L += ["## Spend by resource", "",
              "A model is a resource, not a process step: in the graph every call is one "
              "activity, but they are not priced alike.", "",
              "| Resource | Calls | Tokens | Cost | Share |", "|---|---:|---:|---:|---:|"]
        for name, r in sorted(m.resource_cost.items(), key=lambda kv: -kv[1]["cost"]):
            share = r["cost"] / m.total_cost if m.total_cost else 0
            L.append(f"| `{name}` | {r['n']:,} | {r['tokens']:,} | {_usd(r['cost'])} | {share:.1%} |")
        L.append("")

    if m.loop_cost:
        L += ["## Repeats — the price of loops", "",
              "Activities executed more than once within a single case.", "",
              "| Activity | Cost of repeats |", "|---|---:|"]
        for a, c in sorted(m.loop_cost.items(), key=lambda kv: -kv[1])[:10]:
            L.append(f"| `{a}` | {_usd(c)} |")
        L.append("")
    return "\n".join(L)


def _mm(s: str) -> str:
    """Экранирование для mermaid: кавычки ломают label, скобки ломают форму узла."""
    return s.replace('"', "'").replace("(", "&#40;").replace(")", "&#41;")


# --- HTML --------------------------------------------------------------------

def _back_edges(order: list[str], edges: dict) -> set[tuple[str, str]]:
    """Рёбра, ведущие назад по циклу, — обход в глубину, итеративный.

    Рекурсия здесь была бы уместнее, но длинная цепочка активностей упирается
    в лимит стека Python раньше, чем в память.
    """
    adj: dict[str, list[str]] = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
    state: dict[str, int] = {}               # 0 — в текущем стеке, 1 — закрыт
    back: set[tuple[str, str]] = set()
    for root in [START] + order:
        if root in state:
            continue
        state[root] = 0
        stack = [(root, iter(adj.get(root, ())))]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[node] = 1
                stack.pop()
            elif state.get(nxt) == 0:        # цель ещё в стеке — это ребро назад
                back.add((node, nxt))
            elif nxt not in state:
                state[nxt] = 0
                stack.append((nxt, iter(adj.get(nxt, ()))))
    return back


def _layout(m: Model, order: list[str], edges: dict) -> dict:
    """Мини-Sugiyama: ранг = самый длинный путь от start. Без минимизации пересечений —
    это задача Ц4, здесь нужен только читаемый набросок.

    Обратные рёбра выбрасываются ДО расчёта ранга. Прежняя релаксация называлась
    устойчивой к циклам, но не была ею: каждый проход поднимал ранг всех узлов
    цикла на единицу, а проходов делалось len(order)+2. Граф агента состоит из
    циклов — демо-отчёт на СЕМИ активностях получал холст высотой 11 340 px,
    почти весь пустой. На ациклическом графе релаксация сходится за один проход
    по топологическому порядку и за len(order)+2 в худшем случае.
    """
    back = _back_edges(order, edges)
    forward = [e for e in edges if e not in back]
    rank: dict[str, int] = {START: 0}
    for _ in range(len(order) + 2):
        for (a, b) in forward:
            if b not in (START, END):
                rank[b] = max(rank.get(b, 1), rank.get(a, 0) + 1)
    rank[END] = max(rank.values(), default=0) + 1
    by_rank: dict[int, list[str]] = {}
    for node in [START] + order + [END]:
        by_rank.setdefault(rank.get(node, 1), []).append(node)
    # Широкий ранг сворачивается в несколько строк. Семь инструментов в одну
    # линию давали холст в 1940 px — читатель видел кусок и должен был скроллить
    # вбок, а у агента ранг «всё, что вызывает chat» широк по определению.
    ROW = 4
    pos, y = {}, 60
    for r in sorted(by_rank):
        nodes = by_rank[r]
        for i in range(0, len(nodes), ROW):
            row = nodes[i:i + ROW]
            for j, node in enumerate(row):
                pos[node] = (60 + j * 260 - (len(row) - 1) * 130, y)
            y += 120
    del r
    # Ранг с несколькими узлами центрируется относительно 60 и уезжает в
    # отрицательный x, а viewBox начинается с нуля: узел и подпись ребра просто
    # обрезались слева. Сдвигаем весь холст, а не отдельный ранг, иначе поедут
    # колонки.
    if pos:
        dx = 60 - min(x for x, _ in pos.values())
        if dx:
            pos = {k: (x + dx, y) for k, (x, y) in pos.items()}
    return pos


def render_html(m: Model, *, max_nodes: int = 60, min_edge: float = 0.01,
                found: list[Finding] | None = None) -> str:
    order, edges = _spine(m, max_nodes, min_edge)
    pos = _layout(m, order, edges)
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    w, h = max(xs) + 320, max(ys) + 120

    svg = []
    top = max((x.n for x in edges.values()), default=1)
    drawn: set[tuple[str, str]] = set()
    for (a, b), e in sorted(edges.items(), key=lambda kv: -kv[1].n):
        if a not in pos or b not in pos or (a, b) in drawn:
            continue
        drawn.add((a, b))
        width = 1 + 4 * (e.n / top)

        # Петля: `tool:Bash → tool:Bash`, 142 раза. Прямая кривая из узла в себя
        # вырождается в точку, поэтому рисуем дужку сбоку — иначе самый частый
        # цикл в логе просто не виден.
        if a == b:
            x, y = pos[a]
            svg.append(
                # Сбоку, а не сверху: сверху к узлу приходят входящие рёбра, и
                # петля садилась ровно на их стрелку. Дуга с подписью умещаются
                # в стометровый зазор между соседями по рангу.
                f'<path d="M{x+160},{y+14} C{x+198},{y+6} {x+198},{y+46} {x+160},{y+38}" '
                f'fill="none" stroke="var(--edge)" stroke-width="{width:.1f}" '
                f'marker-end="url(#a)"/>'
                f'<text x="{x+218}" y="{y+30}" class="el">↺ {e.n:,}</text>')
            continue

        x1, y1 = pos[a]; x2, y2 = pos[b]
        # Обратное ребро между той же парой ложится на ту же кривую: 4 088 туда и
        # 4 068 обратно сливались в одну линию с двумя налезающими подписями.
        # Пара — это один round trip, так его и рисуем.
        back = edges.get((b, a))
        if back is not None:
            drawn.add((b, a))
            width = 1 + 4 * (max(e.n, back.n) / top)
            label = f"{e.n:,} ⇄ {back.n:,}"
            marker = ' marker-start="url(#r)" marker-end="url(#a)"'
        else:
            label, marker = f"{e.n:,}", ' marker-end="url(#a)"'

        svg.append(
            f'<path d="M{x1+80},{y1+26} C{x1+80},{y1+70} {x2+80},{y2-40} {x2+80},{y2}" '
            f'fill="none" stroke="var(--edge)" stroke-width="{width:.1f}"{marker}/>'
            f'<text x="{(x1+x2)/2+84}" y="{(y1+y2)/2+20}" class="el">{label}</text>'
        )
    for node, (x, y) in pos.items():
        if node in (START, END):
            svg.append(f'<rect x="{x+20}" y="{y}" width="120" height="26" rx="13" class="term"/>'
                       f'<text x="{x+80}" y="{y+18}" class="nl term-t">{html.escape(node)}</text>')
            continue
        s = m.nodes[node]
        share = s["cost"] / m.total_cost if m.total_cost else 0
        cls = "hot" if share > 0.15 else ("warm" if share > 0.05 else "cool")
        name, second = _node_label(node, s)
        svg.append(
            f'<rect x="{x}" y="{y}" width="160" height="52" rx="6" class="node {cls}"/>'
            f'<text x="{x+80}" y="{y+21}" class="nl">{html.escape(name[:24])}</text>'
            f'<text x="{x+80}" y="{y+39}" class="ns">{html.escape(second)}</text>'
        )

    rows = "".join(
        f"<tr><td class=r>{v.n / m.n_cases:.1%}</td><td class=r>{_usd(v.cost)}</td>"
        f"<td class=r>{_usd(v.cost_avg)}</td><td class=r>{len(v.seq)}</td>"
        f"<td><code>{html.escape(' → '.join(v.seq[:8]))}</code></td></tr>"
        for v in m.top_variants(15)
    )
    res_rows = "".join(
        f"<tr><td><code>{html.escape(n)}</code></td><td class=r>{r['n']:,}</td>"
        f"<td class=r>{r['tokens']:,}</td><td class=r>{_usd(r['cost'])}</td>"
        f"<td class=r>{r['cost'] / m.total_cost if m.total_cost else 0:.1%}</td></tr>"
        for n, r in sorted(m.resource_cost.items(), key=lambda kv: -kv[1]["cost"])
    )
    free_note = (f'<div class=sub>{html.escape(FREE_STEPS_NOTE)}</div>'
                 if any(not m.nodes[a]["cost"] for a in order) else "")
    res_block = (
        "<h2>Spend by resource</h2><div class=scroll><table>"
        "<tr><th>Resource</th><th>Calls</th><th>Tokens</th><th>Cost</th><th>Share</th></tr>"
        f"{res_rows}</table></div>" if res_rows else ""
    )
    find_block = ""
    if found:
        cards = "".join(
            f'<div class=f><div class=fh><span class=fn>{i}</span>{html.escape(f.title)}'
            + (f'<b>{_usd(f.impact_usd)}</b>' if f.impact_usd else "")
            + f'</div><p>{html.escape(f.detail)}</p>'
            + "".join(f'<div class=fe><code>{html.escape(e)}</code></div>' for e in f.evidence)
            + "</div>"
            for i, f in enumerate(found, 1)
        )
        find_block = f"<h2>What to fix</h2><div class=finds>{cards}</div>"
    steps = "".join(
        f'<div class=st><code>{html.escape(cmd)}</code><span>{html.escape(why)}</span></div>'
        for cmd, why in NEXT_STEPS
    )
    next_block = (
        f"<h2>{NEXT_TITLE}</h2><div class=next><p>{html.escape(NEXT_LEAD)}</p>"
        f"{steps}<p class=nt>{html.escape(NEXT_TAIL.replace(chr(96), ''))}</p></div>"
    )
    stats = {
        "Cases": f"{m.n_cases:,}", "Events": f"{m.n_events:,}",
        "Paths": f"{len(m.variants):,}", "Tokens": f"{m.total_tokens:,}",
        "Cost, list prices": _usd(m.total_cost), "Rework": f"{m.rework_rate:.1%}",
    }
    tiles = "".join(f'<div class=t><b>{v}</b><span>{k}</span></div>' for k, v in stats.items())
    period, days = _period(m)
    # Не class=sub: серым мелким шрифтом это читалось как служебная подпись рядом
    # с датой, и вопрос «откуда $800, у меня же подписка» задавали, глядя прямо
    # на неё. Отдельный блок, жирное начало, ответ первым предложением.
    pricing_note = (f'<div class=note><b>{html.escape(PRICING_NOTE_HEAD)}</b> '
                    + html.escape(PRICING_NOTE.format(amount=_usd(m.total_cost), days=days))
                    + '</div>')

    return f"""<!doctype html><html lang=en><meta charset=utf-8>
<title>traceroutine — agent process report</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#18181b;--mut:#71717a;--line:#e4e4e7;--edge:#a1a1aa;
--cool:#f4f4f5;--warm:#fde68a;--hot:#fca5a5;--card:#fafafa;--accent:#b4322e}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0c0c0e;--fg:#fafafa;--mut:#a1a1aa;
--line:#27272a;--edge:#52525b;--cool:#1c1c20;--warm:#78350f;--hot:#7f1d1d;--card:#141417;
--accent:#f87171}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:32px;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}}h2{{font-size:15px;margin:32px 0 12px;color:var(--mut);
text-transform:uppercase;letter-spacing:.06em}}
.sub{{color:var(--mut);font-size:13px;margin-bottom:24px}}
.tiles{{display:flex;flex-wrap:wrap;gap:10px}}
.note{{margin:14px 0 26px;padding:10px 14px;border-left:3px solid var(--mut);color:var(--fg);font-size:14px;line-height:1.5}}
.t{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:110px}}
.t b{{display:block;font-size:20px}}.t span{{color:var(--mut);font-size:12px}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--mut);font-weight:500}}td.r{{text-align:right;white-space:nowrap}}
code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}
.node{{stroke:var(--line)}}.cool{{fill:var(--cool)}}.warm{{fill:var(--warm)}}.hot{{fill:var(--hot)}}
.term{{fill:var(--card);stroke:var(--edge)}}
.nl{{text-anchor:middle;font-size:11px;fill:var(--fg)}}
.ns{{text-anchor:middle;font-size:10px;fill:var(--mut)}}
.el{{font-size:9px;fill:var(--mut)}}
.finds{{display:flex;flex-direction:column;gap:10px}}
.f{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:8px;padding:14px 16px}}
.fh{{display:flex;align-items:baseline;gap:10px;font-weight:600;font-size:14px}}
.fh b{{margin-left:auto;white-space:nowrap;color:var(--accent)}}
.fn{{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;
border-radius:50%;background:var(--accent);color:#fff;font-size:11px;flex:none}}
.f p{{margin:8px 0 0;color:var(--mut);font-size:13px}}
.fe{{margin-top:6px;font-size:12px;color:var(--mut)}}
.next{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}}
.next p{{margin:0 0 12px;font-size:13px}}.next .nt{{margin:12px 0 0;color:var(--mut)}}
.st{{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;padding:6px 0;
border-top:1px solid var(--line)}}
.st code{{background:var(--cool);padding:3px 7px;border-radius:5px;white-space:nowrap}}
.st span{{color:var(--mut);font-size:12px}}
</style>
<h1>traceroutine</h1><div class=sub>{period} &middot; {days} &middot; generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</div>
<div class=tiles>{tiles}</div>
{pricing_note}
{find_block}
{next_block}
<h2>Process graph</h2>
<div class=scroll><svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs><marker id=a markerWidth=8 markerHeight=8 refX=7 refY=3 orient=auto>
<path d="M0,0 L0,6 L7,3 z" fill="var(--edge)"/></marker>
<marker id=r markerWidth=8 markerHeight=8 refX=1 refY=3 orient=auto>
<path d="M8,0 L8,6 L1,3 z" fill="var(--edge)"/></marker></defs>{''.join(svg)}</svg></div>
{free_note}
<h2>Most expensive paths</h2>
<div class=scroll><table><tr><th>Share</th><th>Total</th><th>Avg</th><th>Steps</th><th>Path</th></tr>
{rows}</table></div>
{res_block}
</html>"""


def render_drift(d: Drift, *, name_a: str = "before", name_b: str = "after") -> str:
    def pct(v: float) -> str:
        return f"{v:+.1%}" if v else "0"

    L = [
        "# traceroutine — process comparison", "",
        f"_{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_", "",
        "| | " + name_a + " | " + name_b + " | Δ |", "|---|---:|---:|---:|",
        f"| Cost per run | {_usd(d.cost_per_case_before)} | "
        f"{_usd(d.cost_per_case_after)} | **{pct(d.cost_change)}** |",
        f"| Steps per run | {d.len_before:.1f} | {d.len_after:.1f} | {pct(d.len_change)} |",
        f"| Distinct paths | {d.variants_before} | {d.variants_after} | "
        f"{d.variants_after - d.variants_before:+d} |",
        "",
    ]
    if d.cost_change > 0.1:
        L += [f"> A run got {d.cost_change:.0%} more expensive. "
              f"Look at which steps became more frequent.", ""]
    elif d.cost_change < -0.1:
        L += [f"> A run got {abs(d.cost_change):.0%} cheaper.", ""]

    if d.activity_delta:
        L += ["## Activity frequency", "",
              "How many times an activity occurs in an average run.", "",
              f"| Activity | {name_a} | {name_b} | Δ |", "|---|---:|---:|---:|"]
        for k, a, b in d.activity_delta:
            L.append(f"| `{k}` | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
        L.append("")
    L += ["## Graph changes", ""]
    if d.new_edges or d.gone_edges:
        L.append("A new edge means a new kind of agent behaviour.")
        L.append("")
        for x, y, n in d.new_edges:
            L.append(f"- **appeared** `{x} → {y}` ×{n}")
        for x, y, n in d.gone_edges:
            L.append(f"- gone `{x} → {y}` ×{n}")
    else:
        # Отсутствие структурных изменений — это не пустота, а вывод: агент не
        # научился новому поведению, он делает то же самое, но больше.
        L.append("**The process structure did not change** — not one edge appeared or "
                 "disappeared. The agent is doing the same things, just more of them: "
                 "see the activity frequencies above.")
    L.append("")

    if d.enumerable:
        if d.appeared:
            L += ["## Paths that appeared", ""] + [
                f"- `{_path_digest(s)}` ×{n}" for s, n in d.appeared] + [""]
        if d.vanished:
            L += ["## Paths that disappeared", ""] + [
                f"- `{_path_digest(s)}` ×{n}" for s, n in d.vanished] + [""]
    else:
        L += [f"_Individual trajectories are not listed: "
              f"{max(d.variants_before, d.variants_after)}. "
              + ("Paths do repeat here "
                 f"({d.reuse:.0%} of runs), so the edges above are a reliable signal._"
                 if d.reuse >= 0.5 else
                 "Almost every run is unique, so look at the edges instead._"),
              ""]
    return "\n".join(L)


def render_check(rep: "CheckReport", m: Model, *, max_nodes: int = 18,
                 min_edge: float = 0.05) -> str:
    """Markdown-саммари conformance для GitHub Actions job summary.

    Формат выбран под потребителя: это то, что человек увидит в PR, не открывая
    артефактов. Поэтому вердикт первой строкой, дальше — что именно нарушено, и
    только потом граф. Mermaid здесь незаменим: он рендерится сам, а canvas-отчёт
    в job summary не показать.
    """
    p = rep.process
    L = [f"{'✅' if rep.ok else '❌'} **{p.name}** — "
         + ("the process matches what was declared" if rep.ok
            else f"{len(rep.failures)} violation(s)"),
         ""]

    if rep.failures:
        L += ["| | Violated |", "|---|---|"]
        L += [f"| ❌ | {f} |" for f in rep.failures]
        L.append("")

    t = p.thresholds
    rows = [("Runs", f"{rep.n_cases:,}", ""),
            ("Cost (API list prices)", _usd(rep.total_cost), "")]
    if rep.fitness is not None:
        rows.append(("Fitness", f"{rep.fitness:.3f}",
                     f"≥ {t.fitness_min:.3f}" if t.fitness_min is not None else ""))
        rows.append(("Runs with no deviation", f"{rep.conforming_share:.1%} "
                     f"({rep.fitting}/{rep.n_cases})", ""))
    rows.append(("$ per run", f"${rep.usd_per_case:.4f}",
                 f"≤ ${t.usd_per_case_max:.4f}" if t.usd_per_case_max is not None else ""))
    rows.append(("Steps, p95", f"{rep.steps_p95:.0f}",
                 f"≤ {t.steps_p95_max:.0f}" if t.steps_p95_max is not None else ""))
    if rep.fitness is not None:
        rows.append(("Off-model budget",
                     f"{_usd(rep.off_model_cost)} ({rep.off_model_share:.1%})",
                     f"≤ {t.off_model_share_max:.1%}"
                     if t.off_model_share_max is not None else ""))
    L += ["| Metric | Value | Threshold |", "|---|---:|---:|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    L.append("")

    if rep.rules:
        L += ["## Rules", "", "| | Rule | Runs | Occurrences | Cost |",
              "|---|---|---:|---:|---:|"]
        for st in rep.rules:
            ok = st.ok(rep.n_cases)
            mark = "✅" if ok else ("⚠️" if st.rule.warn else "❌")
            share = f"{st.cases} ({st.share(rep.n_cases):.0%})" if st.cases else "—"
            L.append(f"| {mark} | {_cell(st.rule.text)} | {share} | {st.events or '—'} | "
                     f"{_usd(st.cost) if st.cost else '—'} |")
        L.append("")

    if rep.deviations:
        L += ["## Deviation map", "",
              "_\"Extra\" — the agent took a step the model does not contain; its cost "
              "is the price of the deviation. \"Missing\" — the model required a step "
              "that never happened: no money is attributed to it. A dash does not mean "
              "free: a tool call costs $0 by itself, but its result is re-read on every "
              "later turn — see the context inflation finding._", "",
              "| | Step | Times | Runs | Cost |", "|---|---|---:|---:|---:|"]
        for d in rep.deviations[:12]:
            kind = "extra" if d.kind == "log" else "missing"
            L.append(f"| {kind} | `{_cell(d.activity)}` | {d.n} | {d.cases} | "
                     f"{_usd(d.cost) if d.cost else '—'} |")
        L.append("")

    allowed, forbidden = p.allowed(), p.forbidden()
    order, edges = _spine(m, max_nodes, min_edge)
    if order:
        ids = {START: "S", END: "E"}
        for i, a in enumerate(order):
            ids[a] = f"n{i}"
        L += ["## Observed process", "",
              "_Red marks activities that are not in the declared model._", "",
              "```mermaid", "flowchart TD", f'    {ids[START]}(["▶ start"])']
        for a in order:
            st = m.nodes[a]
            label = _mm(f"{a}<br/>{st['n']}× · {_usd(st['cost'])}")
            L.append(f'    {ids[a]}["{label}"]')
        L.append(f'    {ids[END]}(["■ end"])')
        for (a, b), e in sorted(edges.items(), key=lambda kv: -kv[1].n):
            if a in ids and b in ids:
                L.append(f"    {ids[a]} -->|{e.n}| {ids[b]}")
        off = [a for a in order if a not in allowed or a in forbidden]
        L.append("    classDef off fill:#b4322e,stroke:#7d1f1c,color:#fff")
        if off:
            L.append(f"    class {','.join(ids[a] for a in off)} off")
        L += ["```", ""]

    for w in rep.warnings:
        L += [f"> ⚠️ {w}", ""]
    return "\n".join(L)
