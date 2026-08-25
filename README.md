# traceroutine

Process mining for LLM agent traces — find out where the tokens and the time actually go.

Observability platforms attribute spend by **who**: per user, per team, per API key, per
request. `traceroutine` attributes it by **how** — per execution path.

> An agent's cost is a property of its trajectory, not of its request.

## 30 seconds, zero instrumentation

If you use Claude Code, the data is already on your disk. No exporter, no collector,
no account anywhere:

```bash
uvx traceroutine ingest ~/.claude/projects -o events.parquet --case task
uvx traceroutine report events.parquet -f md
```

Here is what that reported on 442 of my own tasks, $748 of real spend:

> **Results of `Bash` carry 11% of the budget through context.**
> The step itself burns no tokens and shows as **$0.00** in every cost breakdown.
> But each of its results adds ~1,371 tokens to the prompt, and those are re-read on
> **every** subsequent turn: 100M tokens carried in total. Fix by truncating output,
> not by switching models.

That is the whole thesis in one finding: a tool call costs nothing when it happens and
keeps costing for the rest of the run, so the unit of cost is the path.

The phenomenon itself is known — context re-reading is widely documented as the largest
cost driver in agent loops. What is missing everywhere else is the **attribution**: not
"context grows", but *which* step's results carried *how many dollars* across the rest of
the trajectory. Platforms recommend token counts per span, which is per call; this counts
what a step cost on everything that came after it.

## Install

```bash
uvx traceroutine --help          # no install
pip install traceroutine         # or into your environment
```

## Or on the demo dataset

Synthetic traces with deliberately planted pathologies — a retrieval loop, an
escalation to an expensive model, a long tail of rare paths:

```bash
git clone https://github.com/gurov/traceroutine && cd traceroutine
uv pip install -e .
python examples/gen_otlp.py examples/traces.json

traceroutine ingest examples/traces.json -o events.parquet
traceroutine report events.parquet -f md -m examples/activity_map.yaml
traceroutine check events.parquet -p examples/process.yaml -m examples/activity_map.yaml
```

The last command exits 1 and tells you that in 60% of runs the agent answers without
verifying — which is in the declared process and not in the log.

## What it does

| | |
|---|---|
| `ingest <src>` | traces → a canonical event log (parquet) |
| `abstract <log>` | raw span labels → `activity_map.yaml`, a semantic activity vocabulary |
| `report <log>` | findings + process graph: `-f html` or `-f md` |
| `check <log>` | conformance against a declared `process.yaml`; exit codes for CI |
| `diff <a> <b>` | compare two logs: a prompt release, a model swap, two cohorts |

### Sources

Three adapters, three genuinely different data models — which is most of the work:

- **OpenTelemetry / OTLP** — GenAI semantic conventions, plus OpenInference aliases.
  A tree of spans.
- **Claude Code transcripts** — `~/.claude/projects/**/*.jsonl`. Zero instrumentation.
  One model response is written as *several* records sharing a `requestId`, with `usage`
  repeated in each; miss that and cost is overstated 2.1×.
- **Chat transcripts** — the OpenAI-style `messages` format public agent corpora ship in.
  Often no timestamps and no token counts at all.

Nothing but names, timings and counters leaves the Claude Code adapter — no message text,
no tool arguments, no file paths, no commands. That is enforced at the adapter, not at the
report, because reports get shared. There is a test for it.

### Case notion — the decision that determines everything

`--case trace | session | task | attr:<name>` produces three different processes from the
same data, and the difference is not cosmetic. Measured on real Claude Code transcripts:
taking a *session* as the case gives 61 cases and 61 unique paths — variant analysis
degenerates completely, because no two working days are alike. A *task* (one user request
and all the work under it) is the unit that actually repeats.

### The abstraction layer

Two levels. **Rules** strip call arguments, IDs and retry counters — free and reproducible.
An **LLM** does what a regex cannot: decide that `validate_answer`, `self_critique` and
`check_citations` are one step called *verify*.

The LLM runs over the set of **distinct** canonical labels, not over events: two million
events usually have 300–500 distinct shapes, so this is one request, not two million.

A model is a **resource**, not an activity: `chat:claude-opus-5` and `chat:claude-sonnet-5`
collapse into `chat`, and the price difference shows up in the per-resource breakdown.
Backends: `anthropic` (default), `gemini`, `ollama` (for traces that must not leave the
building), `none` (rules only).

`activity_map.yaml` is a versioned project artifact. Its `overrides` section is
hand-edited, wins over `mapping`, and survives regeneration.

Two quality checks, because one is not enough. `--stability N` reports the Rand index
across runs: if the vocabulary drifts, the reports cannot be trusted. And an automatic
merge audit — the activity counter misses the failure that matters, where the model hits
the target count by merging a **paid** step (a model call, where all the money is) with a
**free** one (response formatting). Target met, cost attribution destroyed.

## Conformance — the declared process

The finding that shaped the tool: **a process mining result is only meaningful relative to
an expectation.** An open-ended agent has no declared happy path, so every result decays
into a description of the norm — "the agent calls Bash a lot" is true and useless.
`process.yaml` is that missing expectation.

```yaml
name: coding-agent
flow: |
  chat -> (tool:Read | tool:Grep)+ -> (chat | tool:Edit | tool:Bash)* -> chat
rules:
  - last: chat
  - {before: tool:Edit, expect: tool:Read}     # never edit a file you have not read
  - {after: tool:Edit, expect: tool:Bash}      # after editing, run something
  - {max: 40, of: tool:Bash}
thresholds:
  fitness_min: 0.95
  usd_per_case_max: 2.00
```

Two notations, and both are needed. The **imperative** one (`flow`) yields fitness — a
single "how far from the design" number, computed by aligning each trace against the
model's language. The **declarative** one (`rules`) answers which specific promise was
broken and where. For agents the declarative half matters more, and not as a matter of
taste: an agent is non-deterministic by construction, so a rigid sequence scores near-zero
fitness and tells you nothing. What you actually expect of an agent reads as *"whatever
else you do, don't answer without checking"* — that is the DECLARE vocabulary, and it fits
agents better than Petri nets.

Rule forms: `always`, `never`, `first`, `last`, `{after: A, expect: B}`,
`{before: A, expect: B}`, `{after: A, forbid: B}`, `{max: N, of: X}`, plus `allow:`
(tolerated share of offending runs — no agent is perfect) and `warn:` (reported, does not
fail the build).

Money is attributed **narrowly**. `never`, `forbid` and `max` are broken by an extra step,
and that step's cost is known by name. `after`, `always`, `first`, `last` are broken by
something that did *not* happen — those get no dollar figure at all, rather than a
plausible invented one.

On 442 real Claude Code tasks this reported fitness 0.898, with only 16 runs conforming
outright: **21% of tasks edit a file without reading it first, 12% run no check after an
edit, and 9.3% end on a tool call rather than an answer.** No unit test sees any of that.

### In CI

Exit codes: `0` fine, `1` thresholds violated, `2` broken `process.yaml`. A broken config
and a failed check need different reactions, and CI only sees the code.

`action.yml` is a ready composite action. `check` writes its own summary — with a mermaid
deviation graph — to `$GITHUB_STEP_SUMMARY`, and its metrics to `$GITHUB_OUTPUT`.

```yaml
- uses: gurov/traceroutine@v1
  with: {traces: traces/, process: process.yaml, case: session}
```

With `--baseline` you also get "no worse than before" thresholds: cost per run, run length,
fitness drop. Absolute thresholds need re-tuning after every release; relative ones do not.

## What gets computed

- variants (equivalence classes of paths) and the cost of each
- Pareto concentration over per-run cost
- a directly-follows graph weighted by frequency and by money
- rework rate, and the price of each cycle
- separate accounting for cached and cache-written tokens — cache *writes* cost 1.25–2×
  input, so ignoring them undercounts, and folding cache reads into input overcounts by 10×
- **context inflation** — what a step that shows $0.00 actually costs on the rest of the run
- fitness and a deviation map against a declared process

## When this will not help you

Stated up front, because a tool that always returns five findings eventually returns five
invented ones.

**Variant analysis has a measured limit.** Path uniqueness climbs from 17% at 1–3 steps to
100% at 26+. Above roughly 13 steps trajectories stop repeating, and "rare paths eat the
budget" becomes the tautology "expensive runs are expensive". So the variant lens fits
short structured agents — RAG, support, routing — and not long ones. When repeated paths
drop below 50%, `traceroutine` says so and suppresses those findings instead of dressing up a
tautology. Abstraction helps but does not rescue it: on synthetic data a vocabulary lifts
reuse from 11% to 78%, on real coding traces only from 32% to 37%. Length is the binding
constraint, not granularity.

What still works on long runs: context inflation, cohort `diff`, and conformance — all
three get *stronger* with trace length rather than degenerating.

## Architecture

```
Ingest → Normalize → Abstract → Mine → Analyze → Conform → Render
```

Adapters are the extension point: a two-method contract (`detect`, `read`), see
`adapters/base.py`.

Alignments are implemented here rather than via pm4py, deliberately. pm4py pulls in
pandas + scipy + networkx, and `uvx traceroutine` has to install in seconds; the model
language here is a regular expression over activities rather than an arbitrary Petri net,
so alignment is exact and fits in a 0-1 BFS; all of pm4py's hard algorithmics exist for
unbounded nets, which we do not have. The fitness formula is theirs:
`1 − cost / (|trace| + |shortest model run|)`.

Built on public literature: van der Aalst, *Process Mining: Data Science in Action*;
Leemans, inductive miner; Adriansyah, alignments; Pesic & van der Aalst, DECLARE;
OpenTelemetry GenAI semantic conventions; XES / OCEL 2.0.

## Status

Alpha, and honest about it: it runs end to end on three sources, it has been validated
against real data twice, and it has 145 tests. The interactive graph renderer is not
built yet.

## License

Apache-2.0
