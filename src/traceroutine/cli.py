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

app = typer.Typer(add_completion=False, help="Process mining for LLM agent traces.")


@app.command()
def ingest(
    src: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("events.parquet"), "-o", "--out"),
    adapter: Optional[str] = typer.Option(None, "--adapter"),
    case_notion: str = typer.Option("trace", "--case", help="trace | session | task | attr:<name>"),
    flatten: str = typer.Option("genai", "--flatten", help="genai | leaves | all"),
    pricing: Optional[Path] = typer.Option(None, "--pricing"),
):
    """Traces to a canonical event log."""
    ad = pick(src, adapter)
    pr = Pricing(pricing)
    unknown_cache: set[str] = set()
    events = normalize(ad.read(src), case_notion=case_notion, flatten=flatten, pricing=pr,
                       unknown_cache_convention=unknown_cache)
    n = write(events, out)
    typer.echo(f"{ad.name}: {n:,} events -> {out}  (case={case_notion}, flatten={flatten})")
    if unknown_cache:
        typer.secho(
            f"unknown whether cache is folded into input_tokens for: "
            f"{', '.join(sorted(unknown_cache)[:5])} — cost may be inaccurate.",
            fg=typer.colors.YELLOW,
        )
    if pr.unknown:
        typer.secho(
            f"no prices for models: {', '.join(sorted(pr.unknown))} — cost is "
            f"understated. Add them via --pricing.",
            fg=typer.colors.YELLOW,
        )
    if n == 0:
        typer.secho("0 events: try --flatten all to see the raw spans.",
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
    target: int = typer.Option(12, "--target", help="ceiling on the number of activities"),
    batch: bool = typer.Option(False, "--batch", help="Batch API: half price, but up to an hour of waiting"),
    runs: int = typer.Option(0, "--stability", help="N runs to measure the Rand index"),
):
    """Raw labels to an activity vocabulary. Manual overrides are preserved."""
    evs = read(events)
    raw = [e["activity_raw"] for e in evs]
    be = _backend(backend, model, batch)
    existing = ActivityMap.load(map_path)
    if existing.overrides:
        typer.echo(f"keeping {len(existing.overrides)} manual override(s)")

    amap = build(raw, be, target=target, existing=existing)
    amap.save(map_path)

    c = amap.coverage
    typer.echo(
        f"{c['raw_labels']} raw labels -> {c['canonical_labels']} canonical "
        f"(rules removed {c['recovered_by_rules']}) -> {c['activities']} activities "
        f"-> {map_path}"
    )
    if c["activities"] > target:
        typer.secho(
            f"{c['activities']} activities against a target of {target}: the graph will "
            f"be hard to read. Merge the extras by hand in overrides.",
            fg=typer.colors.YELLOW,
        )
    if c["missing_from_llm"] and backend != "none":
        lost = c["missing_from_llm"] / max(c["canonical_labels"], 1)
        if lost > 0.5:
            typer.secho(
                f"the model returned garbage: {c['missing_from_llm']} of "
                f"{c['canonical_labels']} labels lost, so the vocabulary is effectively "
                f"the rules layer. Check the backend and the model.",
                fg=typer.colors.RED,
            )
        else:
            typer.secho(f"the model dropped {c['missing_from_llm']} labels — rolled back "
                        f"to rules.",
                        fg=typer.colors.YELLOW)
    if c["invented_by_llm"]:
        typer.secho(f"the model invented {c['invented_by_llm']} labels — discarded.",
                    fg=typer.colors.YELLOW)
    for w in audit(amap, evs):
        typer.secho(f"quality: {w}", fg=typer.colors.YELLOW)

    if getattr(be, "usage", None):
        u = be.usage[-1]
        typer.echo(f"tokens: {u['input']} in / {u['output']} out / "
                   f"{u['cache_read']} from cache")

    if runs:
        typer.echo(f"measuring stability: {runs} runs...")
        score = stability(raw, be, runs=runs, target=target)
        typer.echo(f"Rand index across runs: {score:.3f}")
        if score < 0.9:
            typer.secho("unstable: the vocabulary drifts between runs, so the reports "
                        "cannot be trusted.",
                        fg=typer.colors.RED)


@app.command()
def report(
    events: Path = typer.Argument(Path("events.parquet"), exists=True),
    out: Optional[Path] = typer.Option(None, "-o", "--out"),
    fmt: str = typer.Option("html", "-f", "--format", help="html | md"),
    max_nodes: int = typer.Option(0, "--max-nodes", help="0 = format default"),
    map_path: Optional[Path] = typer.Option(None, "-m", "--map", help="activity vocabulary"),
    process: Optional[Path] = typer.Option(None, "-p", "--process",
                                           help="process.yaml: how the agent is supposed to work"),
):
    """Event log to a report."""
    evs = read(events)
    if map_path:
        amap = ActivityMap.load(map_path)
        for e in evs:
            e["activity"] = amap.activity(e["activity_raw"])
        typer.echo(f"vocabulary applied: {len(set(amap.mapping.values()))} activities "
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
        f"{m.n_cases:,} cases · {len(m.variants):,} paths · ${m.total_cost:,.2f} · "
        f"rework {m.rework_rate:.1%} -> {out}"
    )
    for i, f in enumerate(found[:3], 1):
        tag = f" (up to ${f.impact_usd:,.2f})" if f.impact_usd else ""
        typer.secho(f"  {i}. {f.title}{tag}", fg=typer.colors.CYAN)
    # Разовый отчёт — разовое любопытство. Повторно открывают только по
    # повторяющемуся триггеру, а их два: правка промпта и релиз модели.
    if not process:
        typer.secho("  next: `traceroutine diff` — what changed after a release; "
                    "`traceroutine check -p process.yaml` — keep regressions out of CI",
                    fg=typer.colors.BRIGHT_BLACK)


def _emit(var: str, text: str) -> None:
    """Append to the file an environment variable points at, if it is set."""
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
                                 help="process.yaml: how the agent is supposed to work"),
    map_path: Optional[Path] = typer.Option(None, "-m", "--map"),
    baseline: Optional[Path] = typer.Option(None, "--baseline", exists=True,
                                            help="the \"before\" log, enabling growth thresholds"),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="markdown summary"),
    warn_only: bool = typer.Option(False, "--warn-only",
                                   help="always exit 0 — for taming thresholds"),
):
    """Check a log against a declared process. Built for CI.

    Exit codes: 0 fine, 1 thresholds violated, 2 broken process.yaml. The split is
    deliberate: a broken config and a failed check need different reactions, and CI
    only sees the code.
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

    head = "OK" if rep.ok else "VIOLATED"
    typer.secho(
        f"{head}: {proc.name} · {rep.n_cases:,} runs · "
        + (f"fitness {rep.fitness:.3f} · " if rep.fitness is not None else "")
        + f"${rep.usd_per_case:.4f}/run",
        fg=typer.colors.GREEN if rep.ok else typer.colors.RED,
    )
    for f in rep.failures:
        typer.secho(f"  ✗ {f}", fg=typer.colors.RED)
    for w in rep.rule_warnings + rep.warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)
    if rep.fitness is not None and rep.off_model_cost:
        typer.echo(f"  off-model: ${rep.off_model_cost:,.2f} "
                   f"({rep.off_model_share:.1%} of budget)")

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
    """Compare two logs: what changed after a model swap or a prompt release."""
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
        f"cost per run {d.cost_change:+.1%} · steps {d.len_change:+.1%} · "
        f"paths {d.variants_after - d.variants_before:+d} -> {out}"
    )


@app.command()
def version():
    typer.echo(__version__)


if __name__ == "__main__":
    app()
