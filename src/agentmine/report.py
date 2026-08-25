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
NEXT_TITLE = "Что дальше"
NEXT_LEAD = (
    "Всё выше — срез одного момента. Он поедет при первой же правке промпта или "
    "смене модели, и узнать об этом постфактум по счёту — самый дорогой способ."
)
NEXT_STEPS = [
    ("agentmine diff before.parquet after.parquet",
     "что изменилось в поведении: новые пути, рост длины прогона, цена за задачу"),
    ("agentmine check events.parquet -p process.yaml",
     "не пустить регрессию в мерж: коды возврата для CI, саммари в job summary"),
]
NEXT_TAIL = (
    "`process.yaml` — объявленный процесс: как агент ДОЛЖЕН работать. Без него "
    "находки выше описывают норму, с ним — отклонение от намерения. Это разница "
    "между «агент часто вызывает Bash» и «в 21% задач файл правится без чтения»."
)


def _cell(s: str) -> str:
    """Вертикальная черта режет ячейку markdown-таблицы даже внутри `code`.
    А в отчёте она законна: так записана альтернатива в самом process.yaml."""
    return s.replace("|", "\\|")


def _usd(v: float) -> str:
    return f"${v:,.2f}" if v >= 0.01 else f"${v:.4f}"


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
    order = [a for a, _ in ranked if a in keep]
    return order, edges


def render_markdown(m: Model, *, max_nodes: int = 25, min_edge: float = 0.02,
                    found: list[Finding] | None = None) -> str:
    order, edges = _spine(m, max_nodes, min_edge)
    ids = {START: "S", END: "E"}
    for i, a in enumerate(order):
        ids[a] = f"n{i}"

    L = [
        "# agentmine — отчёт по процессу агента",
        "",
        f"_{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_",
        "",
        "## Итого",
        "",
        "| | |",
        "|---|---|",
        f"| Кейсов | {m.n_cases:,} |",
        f"| Событий | {m.n_events:,} |",
        f"| Уникальных путей | {len(m.variants):,} |",
        f"| Токенов | {m.total_tokens:,} |",
        f"| Стоимость | {_usd(m.total_cost)} |",
        f"| Rework rate | {m.rework_rate:.1%} |",
        "",
    ]

    # Вывод про концентрацию печатается только если концентрация ЕСТЬ.
    # Отчёт, объявляющий «90% кейсов дают 80% расходов», разрушает доверие
    # к остальным цифрам: это ровное распределение, а не находка.
    # Находки идут ПЕРЕД графиками: дашборд — витамин, находка — обезболивающее.
    if found:
        L += ["## Что чинить", "",
              "_Оценки экономии консервативны и **могут пересекаться**: один дорогой "
              "прогон попадает и в «редкие пути», и в «эскалацию». Складывать их нельзя._",
              ""]
        for i, f in enumerate(found, 1):
            head = f"**{i}. {f.title}**"
            if f.impact_usd:
                head += f" — до {_usd(f.impact_usd)} ({f.share:.0%} бюджета)"
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
                        f"> **{acc_cases:.0%} прогонов дают 80% расходов.** "
                        f"Стоимость — свойство траектории, а не запроса.",
                        "",
                    ]
                break

    L += ["## Граф процесса", "", "```mermaid", "flowchart TD"]
    L.append(f'    {ids[START]}(["▶ start"])')
    for a in order:
        s = m.nodes[a]
        label = f"{a}<br/>{s['n']}× · {_usd(s['cost'])}"
        L.append(f'    {ids[a]}["{_mm(label)}"]')
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

    if len(m.nodes) > max_nodes:
        L += [
            f"_Показаны {max_nodes} из {len(m.nodes)} активностей "
            f"(mermaid нечитаем на большем). Полный граф — в HTML-отчёте._",
            "",
        ]

    L += ["## Самые дорогие пути", "", "| Доля кейсов | Всего | В среднем | Шагов | Ошибки | Путь |",
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
        L += ["## Расход по ресурсам", "",
              "Модель — ресурс, а не шаг процесса: в графе все вызовы это одна активность, "
              "а цена у них разная.", "",
              "| Ресурс | Вызовов | Токенов | Стоимость | Доля |", "|---|---:|---:|---:|---:|"]
        for name, r in sorted(m.resource_cost.items(), key=lambda kv: -kv[1]["cost"]):
            share = r["cost"] / m.total_cost if m.total_cost else 0
            L.append(f"| `{name}` | {r['n']:,} | {r['tokens']:,} | {_usd(r['cost'])} | {share:.1%} |")
        L.append("")

    if m.loop_cost:
        L += ["## Повторы — цена циклов", "",
              "Активности, выполнявшиеся в одном кейсе более одного раза.", "",
              "| Активность | Стоимость повторов |", "|---|---:|"]
        for a, c in sorted(m.loop_cost.items(), key=lambda kv: -kv[1])[:10]:
            L.append(f"| `{a}` | {_usd(c)} |")
        L.append("")
    return "\n".join(L)


def _mm(s: str) -> str:
    """Экранирование для mermaid: кавычки ломают label, скобки ломают форму узла."""
    return s.replace('"', "'").replace("(", "&#40;").replace(")", "&#41;")


# --- HTML --------------------------------------------------------------------

def _layout(m: Model, order: list[str], edges: dict) -> dict:
    """Мини-Sugiyama: ранг = самый длинный путь от start. Без минимизации пересечений —
    это задача Ц4, здесь нужен только читаемый набросок."""
    rank: dict[str, int] = {START: 0}
    adj: dict[str, list[str]] = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
    for _ in range(len(order) + 2):          # релаксация, устойчива к циклам
        for (a, b) in edges:
            if b not in (START, END):
                rank[b] = max(rank.get(b, 1), rank.get(a, 0) + 1)
    rank[END] = max(rank.values(), default=0) + 1
    by_rank: dict[int, list[str]] = {}
    for node in [START] + order + [END]:
        by_rank.setdefault(rank.get(node, 1), []).append(node)
    pos = {}
    for r, nodes in by_rank.items():
        for i, node in enumerate(nodes):
            pos[node] = (60 + i * 260 - (len(nodes) - 1) * 130, 60 + r * 120)
    return pos


def render_html(m: Model, *, max_nodes: int = 60, min_edge: float = 0.01,
                found: list[Finding] | None = None) -> str:
    order, edges = _spine(m, max_nodes, min_edge)
    pos = _layout(m, order, edges)
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    w, h = max(xs) + 320, max(ys) + 120

    svg = []
    for (a, b), e in edges.items():
        if a in pos and b in pos:
            x1, y1 = pos[a]; x2, y2 = pos[b]
            width = 1 + 4 * (e.n / max(x.n for x in edges.values()))
            svg.append(
                f'<path d="M{x1+80},{y1+26} C{x1+80},{y1+70} {x2+80},{y2-40} {x2+80},{y2}" '
                f'fill="none" stroke="var(--edge)" stroke-width="{width:.1f}" marker-end="url(#a)"/>'
                f'<text x="{(x1+x2)/2+84}" y="{(y1+y2)/2+20}" class="el">{e.n}</text>'
            )
    for node, (x, y) in pos.items():
        if node in (START, END):
            svg.append(f'<rect x="{x+20}" y="{y}" width="120" height="26" rx="13" class="term"/>'
                       f'<text x="{x+80}" y="{y+18}" class="nl term-t">{html.escape(node)}</text>')
            continue
        s = m.nodes[node]
        share = s["cost"] / m.total_cost if m.total_cost else 0
        cls = "hot" if share > 0.15 else ("warm" if share > 0.05 else "cool")
        svg.append(
            f'<rect x="{x}" y="{y}" width="160" height="52" rx="6" class="node {cls}"/>'
            f'<text x="{x+80}" y="{y+21}" class="nl">{html.escape(node[:24])}</text>'
            f'<text x="{x+80}" y="{y+39}" class="ns">{s["n"]}× · {_usd(s["cost"])}</text>'
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
    res_block = (
        "<h2>Расход по ресурсам</h2><div class=scroll><table>"
        "<tr><th>Ресурс</th><th>Вызовов</th><th>Токенов</th><th>Стоимость</th><th>Доля</th></tr>"
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
        find_block = f"<h2>Что чинить</h2><div class=finds>{cards}</div>"
    steps = "".join(
        f'<div class=st><code>{html.escape(cmd)}</code><span>{html.escape(why)}</span></div>'
        for cmd, why in NEXT_STEPS
    )
    next_block = (
        f"<h2>{NEXT_TITLE}</h2><div class=next><p>{html.escape(NEXT_LEAD)}</p>"
        f"{steps}<p class=nt>{html.escape(NEXT_TAIL.replace(chr(96), ''))}</p></div>"
    )
    stats = {
        "Кейсов": f"{m.n_cases:,}", "Событий": f"{m.n_events:,}",
        "Путей": f"{len(m.variants):,}", "Токенов": f"{m.total_tokens:,}",
        "Стоимость": _usd(m.total_cost), "Rework": f"{m.rework_rate:.1%}",
    }
    tiles = "".join(f'<div class=t><b>{v}</b><span>{k}</span></div>' for k, v in stats.items())

    return f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>agentmine — отчёт</title><meta name=viewport content="width=device-width,initial-scale=1">
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
<h1>agentmine</h1><div class=sub>{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · вертикальный срез (Ц1)</div>
<div class=tiles>{tiles}</div>
{find_block}
{next_block}
<h2>Граф процесса</h2>
<div class=scroll><svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs><marker id=a markerWidth=8 markerHeight=8 refX=7 refY=3 orient=auto>
<path d="M0,0 L0,6 L7,3 z" fill="var(--edge)"/></marker></defs>{''.join(svg)}</svg></div>
<h2>Самые дорогие пути</h2>
<div class=scroll><table><tr><th>Доля</th><th>Всего</th><th>Средн.</th><th>Шагов</th><th>Путь</th></tr>
{rows}</table></div>
{res_block}
</html>"""


def render_drift(d: Drift, *, name_a: str = "до", name_b: str = "после") -> str:
    def pct(v: float) -> str:
        return f"{v:+.1%}" if v else "0"

    L = [
        "# agentmine — сравнение процессов", "",
        f"_{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_", "",
        "| | " + name_a + " | " + name_b + " | Δ |", "|---|---:|---:|---:|",
        f"| Стоимость прогона | {_usd(d.cost_per_case_before)} | "
        f"{_usd(d.cost_per_case_after)} | **{pct(d.cost_change)}** |",
        f"| Шагов в прогоне | {d.len_before:.1f} | {d.len_after:.1f} | {pct(d.len_change)} |",
        f"| Уникальных путей | {d.variants_before} | {d.variants_after} | "
        f"{d.variants_after - d.variants_before:+d} |",
        "",
    ]
    if d.cost_change > 0.1:
        L += [f"> Прогон подорожал на {d.cost_change:.0%}. "
              f"Смотрите, какие шаги стали встречаться чаще.", ""]
    elif d.cost_change < -0.1:
        L += [f"> Прогон подешевел на {abs(d.cost_change):.0%}.", ""]

    if d.activity_delta:
        L += ["## Частота активностей", "",
              "Сколько раз активность встречается в среднем прогоне.", "",
              f"| Активность | {name_a} | {name_b} | Δ |", "|---|---:|---:|---:|"]
        for k, a, b in d.activity_delta:
            L.append(f"| `{k}` | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
        L.append("")
    L += ["## Изменения в графе", ""]
    if d.new_edges or d.gone_edges:
        L.append("Новый переход — это новый вид поведения агента.")
        L.append("")
        for x, y, n in d.new_edges:
            L.append(f"- **появился** `{x} → {y}` ×{n}")
        for x, y, n in d.gone_edges:
            L.append(f"- исчез `{x} → {y}` ×{n}")
    else:
        # Отсутствие структурных изменений — это не пустота, а вывод: агент не
        # научился новому поведению, он делает то же самое, но больше.
        L.append("**Структура процесса не изменилась** — ни один переход не появился и "
                 "не исчез. Агент делает то же самое, просто больше: смотрите частоты "
                 "активностей выше.")
    L.append("")

    if d.enumerable:
        if d.appeared:
            L += ["## Появились пути", ""] + [
                f"- `{_path_digest(s)}` ×{n}" for s, n in d.appeared] + [""]
        if d.vanished:
            L += ["## Исчезли пути", ""] + [
                f"- `{_path_digest(s)}` ×{n}" for s, n in d.vanished] + [""]
    else:
        L += [f"_Отдельные траектории не перечисляются: вариантов "
              f"{max(d.variants_before, d.variants_after)}. "
              + ("Пути при этом повторяются "
                 f"({d.reuse:.0%} прогонов), поэтому переходы выше — надёжный сигнал._"
                 if d.reuse >= 0.5 else
                 "Почти каждый прогон уникален, поэтому смотреть надо на переходы._"),
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
         + ("процесс соответствует объявленному" if rep.ok
            else f"нарушений: {len(rep.failures)}"),
         ""]

    if rep.failures:
        L += ["| | Нарушено |", "|---|---|"]
        L += [f"| ❌ | {f} |" for f in rep.failures]
        L.append("")

    t = p.thresholds
    rows = [("Прогонов", f"{rep.n_cases:,}", ""),
            ("Стоимость", _usd(rep.total_cost), "")]
    if rep.fitness is not None:
        rows.append(("Fitness", f"{rep.fitness:.3f}",
                     f"≥ {t.fitness_min:.3f}" if t.fitness_min is not None else ""))
        rows.append(("Прогонов без отклонений", f"{rep.conforming_share:.1%} "
                     f"({rep.fitting}/{rep.n_cases})", ""))
    rows.append(("$ на прогон", f"${rep.usd_per_case:.4f}",
                 f"≤ ${t.usd_per_case_max:.4f}" if t.usd_per_case_max is not None else ""))
    rows.append(("Шагов, p95", f"{rep.steps_p95:.0f}",
                 f"≤ {t.steps_p95_max:.0f}" if t.steps_p95_max is not None else ""))
    if rep.fitness is not None:
        rows.append(("Бюджет вне модели",
                     f"{_usd(rep.off_model_cost)} ({rep.off_model_share:.1%})",
                     f"≤ {t.off_model_share_max:.1%}"
                     if t.off_model_share_max is not None else ""))
    L += ["| Метрика | Значение | Порог |", "|---|---:|---:|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    L.append("")

    if rep.rules:
        L += ["## Правила", "", "| | Правило | Прогонов | Случаев | Стоимость |",
              "|---|---|---:|---:|---:|"]
        for st in rep.rules:
            ok = st.ok(rep.n_cases)
            mark = "✅" if ok else ("⚠️" if st.rule.warn else "❌")
            share = f"{st.cases} ({st.share(rep.n_cases):.0%})" if st.cases else "—"
            L.append(f"| {mark} | {_cell(st.rule.text)} | {share} | {st.events or '—'} | "
                     f"{_usd(st.cost) if st.cost else '—'} |")
        L.append("")

    if rep.deviations:
        L += ["## Карта отклонений", "",
              "_«Лишний» — агент сделал шаг, которого нет в модели; его стоимость "
              "и есть цена отклонения. «Пропущен» — модель требовала шаг, которого "
              "не случилось: денег ему не приписываем. Шаг с прочерком не бесплатен: "
              "вызов инструмента стоит $0 сам по себе, но его результат перечитывается "
              "на каждом следующем ходе — это видно в находке про раздувание контекста._",
              "",
              "| | Шаг | Раз | Прогонов | Стоимость |", "|---|---|---:|---:|---:|"]
        for d in rep.deviations[:12]:
            kind = "лишний" if d.kind == "log" else "пропущен"
            L.append(f"| {kind} | `{_cell(d.activity)}` | {d.n} | {d.cases} | "
                     f"{_usd(d.cost) if d.cost else '—'} |")
        L.append("")

    allowed, forbidden = p.allowed(), p.forbidden()
    order, edges = _spine(m, max_nodes, min_edge)
    if order:
        ids = {START: "S", END: "E"}
        for i, a in enumerate(order):
            ids[a] = f"n{i}"
        L += ["## Наблюдаемый процесс", "",
              "_Красным — активности, которых нет в объявленной модели._", "",
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
