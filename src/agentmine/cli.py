from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .abstract import (GEMINI_DEFAULT, MODEL_DEFAULT, OLLAMA_DEFAULT, ActivityMap,
                       AnthropicBackend, GeminiBackend, NoBackend, OllamaBackend, audit,
                       build, stability)
from .adapters import pick
from .analyze import drift, findings
from .conform import ConfigError, Process
from .conform import check as run_check
from .conform import findings as conformance_findings
from .mine import mine
from .normalize import normalize
from .pricing import Pricing
from .report import render_check, render_drift, render_html, render_markdown
from .store import read, write

app = typer.Typer(add_completion=False, help="Process mining для трейсов LLM-агентов.")


@app.command()
def ingest(
    src: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("events.parquet"), "-o", "--out"),
    adapter: Optional[str] = typer.Option(None, "--adapter"),
    case_notion: str = typer.Option("trace", "--case", help="trace | session | attr:<имя>"),
    flatten: str = typer.Option("genai", "--flatten", help="genai | leaves | all"),
    pricing: Optional[Path] = typer.Option(None, "--pricing"),
):
    """Трейсы → канонический event log."""
    ad = pick(src, adapter)
    pr = Pricing(pricing)
    unknown_cache: set[str] = set()
    events = normalize(ad.read(src), case_notion=case_notion, flatten=flatten, pricing=pr,
                       unknown_cache_convention=unknown_cache)
    n = write(events, out)
    typer.echo(f"{ad.name}: {n:,} событий → {out}  (case={case_notion}, flatten={flatten})")
    if unknown_cache:
        typer.secho(
            f"неизвестно, включён ли кеш в input_tokens у: {', '.join(sorted(unknown_cache)[:5])}"
            f" — стоимость может быть неточной.",
            fg=typer.colors.YELLOW,
        )
    if pr.unknown:
        typer.secho(
            f"нет цен для моделей: {', '.join(sorted(pr.unknown))} — стоимость занижена. "
            f"Добавьте их через --pricing.",
            fg=typer.colors.YELLOW,
        )
    if n == 0:
        typer.secho("0 событий: попробуйте --flatten all, чтобы увидеть сырые спаны.",
                    fg=typer.colors.YELLOW)


def _backend(kind: str, model: str | None, batch: bool):
    if kind == "anthropic":
        return AnthropicBackend(model or MODEL_DEFAULT, batch=batch)
    if kind == "gemini":
        return GeminiBackend(model or GEMINI_DEFAULT)
    if kind == "ollama":
        return OllamaBackend(model or OLLAMA_DEFAULT)
    if kind == "none":
        return NoBackend()
    raise SystemExit("--backend: anthropic | gemini | ollama | none")


@app.command()
def abstract(
    events: Path = typer.Argument(Path("events.parquet"), exists=True),
    map_path: Path = typer.Option(Path("activity_map.yaml"), "-m", "--map"),
    backend: str = typer.Option("anthropic", "--backend", help="anthropic | gemini | ollama | none"),
    model: Optional[str] = typer.Option(None, "--model"),
    target: int = typer.Option(12, "--target", help="потолок числа активностей"),
    batch: bool = typer.Option(False, "--batch", help="Batch API: -50% цены, но до часа ожидания"),
    runs: int = typer.Option(0, "--stability", help="N прогонов для замера Rand index"),
):
    """Сырые метки → словарь активностей. Ручные overrides сохраняются."""
    evs = read(events)
    raw = [e["activity_raw"] for e in evs]
    be = _backend(backend, model, batch)
    existing = ActivityMap.load(map_path)
    if existing.overrides:
        typer.echo(f"сохраняю {len(existing.overrides)} ручных override")

    amap = build(raw, be, target=target, existing=existing)
    amap.save(map_path)

    c = amap.coverage
    typer.echo(
        f"{c['raw_labels']} сырых меток → {c['canonical_labels']} канонических "
        f"(правила убрали {c['recovered_by_rules']}) → {c['activities']} активностей "
        f"→ {map_path}"
    )
    if c["activities"] > target:
        typer.secho(
            f"активностей {c['activities']} при цели {target}: граф будет плохо читаем. "
            f"Слейте лишнее руками в overrides.",
            fg=typer.colors.YELLOW,
        )
    if c["missing_from_llm"] and backend != "none":
        lost = c["missing_from_llm"] / max(c["canonical_labels"], 1)
        if lost > 0.5:
            typer.secho(
                f"модель вернула негодный результат: потеряно {c['missing_from_llm']} из "
                f"{c['canonical_labels']} меток, словарь фактически = уровень правил. "
                f"Проверьте бэкенд и модель.",
                fg=typer.colors.RED,
            )
        else:
            typer.secho(f"{c['missing_from_llm']} меток модель потеряла — откачены на правила.",
                        fg=typer.colors.YELLOW)
    if c["invented_by_llm"]:
        typer.secho(f"{c['invented_by_llm']} меток модель выдумала — отброшены.",
                    fg=typer.colors.YELLOW)
    for w in audit(amap, evs):
        typer.secho(f"качество: {w}", fg=typer.colors.YELLOW)

    if getattr(be, "usage", None):
        u = be.usage[-1]
        typer.echo(f"токены: {u['input']} in / {u['output']} out / "
                   f"{u['cache_read']} из кеша")

    if runs:
        typer.echo(f"замер стабильности: {runs} прогонов...")
        score = stability(raw, be, runs=runs, target=target)
        typer.echo(f"Rand index между прогонами: {score:.3f}")
        if score < 0.9:
            typer.secho("нестабильно: словарь «плавает» между прогонами, отчётам верить нельзя.",
                        fg=typer.colors.RED)


@app.command()
def report(
    events: Path = typer.Argument(Path("events.parquet"), exists=True),
    out: Optional[Path] = typer.Option(None, "-o", "--out"),
    fmt: str = typer.Option("html", "-f", "--format", help="html | md"),
    max_nodes: int = typer.Option(0, "--max-nodes", help="0 = по умолчанию для формата"),
    map_path: Optional[Path] = typer.Option(None, "-m", "--map", help="словарь активностей"),
    process: Optional[Path] = typer.Option(None, "-p", "--process",
                                           help="process.yaml: как агент ДОЛЖЕН работать"),
):
    """Event log → отчёт."""
    evs = read(events)
    if map_path:
        amap = ActivityMap.load(map_path)
        for e in evs:
            e["activity"] = amap.activity(e["activity_raw"])
        typer.echo(f"словарь применён: {len(set(amap.mapping.values()))} активностей "
                   f"({amap.backend})")
    m = mine(evs)
    found = findings(m, evs)
    if process:
        # Находки conformance идут ПЕРВЫМИ и намеренно: остальные описывают лог
        # сам по себе, а эти — расхождение с намерением. Ц3.5 показал, что на
        # открытых агентах только они и не вырождаются в описание нормы.
        try:
            rep = run_check(Process.load(process), evs)
        except ConfigError as exc:
            raise typer.Exit(_fail(exc)) from exc
        # Общий потолок держим: восемь пунктов — уже не список дел, а свалка.
        found = (conformance_findings(rep) + found)[:6]
        for w in rep.warnings:
            typer.secho(f"conformance: {w}", fg=typer.colors.YELLOW)
    kw = {"found": found}
    if max_nodes:
        kw["max_nodes"] = max_nodes
    if fmt == "md":
        text = render_markdown(m, **kw)
        out = out or Path("report.md")
    elif fmt == "html":
        text = render_html(m, **kw)
        out = out or Path("report.html")
    else:
        raise SystemExit("--format: html | md")
    out.write_text(text, encoding="utf-8")
    typer.echo(
        f"{m.n_cases:,} кейсов · {len(m.variants):,} путей · ${m.total_cost:,.2f} · "
        f"rework {m.rework_rate:.1%} → {out}"
    )
    for i, f in enumerate(found[:3], 1):
        tag = f" (до ${f.impact_usd:,.2f})" if f.impact_usd else ""
        typer.secho(f"  {i}. {f.title}{tag}", fg=typer.colors.CYAN)


def _emit(var: str, text: str) -> None:
    """Дописать в файл, на который указывает переменная окружения, если она есть."""
    path = os.environ.get(var)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)


def _fail(exc: ConfigError) -> int:
    typer.secho(f"process.yaml: {exc}", fg=typer.colors.RED, err=True)
    return 2


@app.command()
def check(
    events: Path = typer.Argument(Path("events.parquet"), exists=True),
    process: Path = typer.Option(..., "-p", "--process", exists=True,
                                 help="process.yaml: как агент ДОЛЖЕН работать"),
    map_path: Optional[Path] = typer.Option(None, "-m", "--map"),
    baseline: Optional[Path] = typer.Option(None, "--baseline", exists=True,
                                            help="лог «как было» для порогов на рост"),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="markdown-саммари"),
    warn_only: bool = typer.Option(False, "--warn-only",
                                   help="всегда выходить с 0 — режим приручения порогов"),
):
    """Проверить лог на соответствие объявленному процессу. Для CI.

    Коды возврата: 0 — норма, 1 — пороги нарушены, 2 — ошибка в process.yaml.
    Разделение неслучайно: сломанный конфиг и провалившаяся проверка требуют
    разной реакции, а CI видит только код.
    """
    try:
        proc = Process.load(process)
    except ConfigError as exc:
        raise typer.Exit(_fail(exc)) from exc

    def load(p: Path) -> list[dict]:
        evs = read(p)
        if map_path:
            amap = ActivityMap.load(map_path)
            for e in evs:
                e["activity"] = amap.activity(e["activity_raw"])
        return evs

    evs = load(events)
    rep = run_check(proc, evs, baseline=load(baseline) if baseline else None)

    head = "OK" if rep.ok else "НАРУШЕНО"
    typer.secho(
        f"{head}: {proc.name} · {rep.n_cases:,} прогонов · "
        + (f"fitness {rep.fitness:.3f} · " if rep.fitness is not None else "")
        + f"${rep.usd_per_case:.4f}/прогон",
        fg=typer.colors.GREEN if rep.ok else typer.colors.RED,
    )
    for f in rep.failures:
        typer.secho(f"  ✗ {f}", fg=typer.colors.RED)
    for w in rep.rule_warnings + rep.warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)
    if rep.fitness is not None and rep.off_model_cost:
        typer.echo(f"  вне модели: ${rep.off_model_cost:,.2f} "
                   f"({rep.off_model_share:.1%} бюджета)")

    text = render_check(rep, mine(evs))
    if out:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"  → {out}")
    # GitHub Actions сам подставляет эти переменные. Ради нескольких строк здесь
    # экшену не нужно ни знания о формате отчёта, ни парсинга вывода — а значения
    # становятся доступны следующим шагам: закомментировать PR, придержать деплой.
    _emit("GITHUB_STEP_SUMMARY", text + "\n")
    _emit("GITHUB_OUTPUT", "".join(
        f"{k}={v}\n" for k, v in {
            "status": "ok" if rep.ok else "failed",
            "fitness": "" if rep.fitness is None else f"{rep.fitness:.4f}",
            "conforming": f"{rep.conforming_share:.4f}",
            "usd-per-case": f"{rep.usd_per_case:.6f}",
            "off-model-share": f"{rep.off_model_share:.4f}",
            "failures": str(len(rep.failures)),
        }.items()))

    raise typer.Exit(0 if (rep.ok or warn_only) else 1)


@app.command()
def diff(
    before: Path = typer.Argument(..., exists=True),
    after: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("drift.md"), "-o", "--out"),
    map_path: Optional[Path] = typer.Option(None, "-m", "--map"),
):
    """Сравнить два лога: что изменилось после смены модели или релиза промпта."""
    def load(p: Path):
        evs = read(p)
        if map_path:
            amap = ActivityMap.load(map_path)
            for e in evs:
                e["activity"] = amap.activity(e["activity_raw"])
        return mine(evs)

    d = drift(load(before), load(after))
    out.write_text(render_drift(d, name_a=before.stem, name_b=after.stem), encoding="utf-8")
    typer.echo(
        f"стоимость прогона {d.cost_change:+.1%} · шагов {d.len_change:+.1%} · "
        f"путей {d.variants_after - d.variants_before:+d} → {out}"
    )


@app.command()
def version():
    typer.echo(__version__)


if __name__ == "__main__":
    app()
