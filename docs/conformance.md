# Conformance — the declared process

The finding that shaped the tool: **a process mining result is only meaningful relative
to an expectation.** An open-ended agent has no declared happy path, so every result
decays into a description of the norm — "the agent calls Bash a lot" is true and useless.
`process.yaml` is that missing expectation.

```yaml
name: coding-agent
case: task
flow: |
  chat -> (tool:Read | tool:Grep)+ -> (chat | tool:Edit | tool:Bash)* -> chat
rules:
  - last: chat                                 # don't end on a tool call
  - {before: tool:Edit, expect: tool:Read}     # never edit a file you have not read
  - {after: tool:Edit, expect: tool:Bash}      # after editing, run something
  - {max: 40, of: tool:Bash}
thresholds:
  fitness_min: 0.95
  usd_per_case_max: 2.00
```

Run it against a log:

```console
$ traceroutine check events.parquet -p process.yaml
vocabulary activity_map.yaml
VIOLATED: coding-agent · 463 runs · fitness 0.900 · $1.7035/run
  ✗ fitness 0.900 < 0.950
  ✗ run length p95 111 > 60
  ✗ a run ends with `chat`: violated in 42 runs (9.1%)
  ✗ before `tool:Edit`, `tool:Read` must have happened: violated in 93 runs (20.1%)
  ✗ after `tool:Edit`, `tool:Bash` must eventually follow: violated in 54 runs (11.7%)
  ✗ `tool:Bash` at most 40× per run: violated in 16 runs (3.5%)
  off-model: $1.05 (0.1% of budget)
```

**20% of tasks edit a file without reading it first, 12% run no check after an edit, and
9% end on a tool call rather than an answer.** No unit test sees any of that.

## Two notations, and both are needed

The **imperative** one (`flow`) yields fitness — a single "how far from the design"
number, from aligning each trace against the model's language. Operators: `-> | ( ) * + ?`
and the wildcard `any`.

The **declarative** one (`rules`) answers which specific promise was broken and where.
For agents the declarative half matters more, and not as a matter of taste: an agent is
non-deterministic by construction, so a rigid sequence scores near-zero fitness and tells
you nothing. What you actually expect reads as *"whatever else you do, don't answer
without checking"* — that is the DECLARE vocabulary, and it fits agents better than
Petri nets.

Rule forms: `always`, `never`, `first`, `last`, `{after: A, expect: B}`,
`{before: A, expect: B}`, `{after: A, forbid: B}`, `{max: N, of: X}`, plus `allow:`
(tolerated share of offending runs — no agent is perfect) and `warn:` (reported, does not
fail the build).

## Money is attributed narrowly

`never`, `forbid` and `max` are broken by an extra step, whose cost is known by name.
`after`, `always`, `first`, `last` are broken by something that did *not* happen — those
get no dollar figure at all, rather than a plausible invented one.

## In CI

Exit codes: `0` fine, `1` thresholds violated, `2` broken `process.yaml`. A broken config
and a failed check need different reactions, and CI only sees the code. A model written
against activities the log cannot contain is a broken config, not a violation:

```console
$ traceroutine check events.parquet -p process.yaml     # log never abstracted
process.yaml and the log are at different abstraction levels:
  `chat` — the log has `chat:claude-opus-4-8`, `chat:claude-opus-5`
Nothing is compared against those names, so fitness and the deviation map would both be
fiction. Build the vocabulary and pass it in: …
```

`action.yml` is a ready composite action. `check` writes its own summary — with a mermaid
deviation graph — to `$GITHUB_STEP_SUMMARY`, and its metrics to `$GITHUB_OUTPUT`.

```yaml
- uses: gurov/traceroutine@v1
  with: {traces: traces/, process: process.yaml, case: session}
```

A complete workflow to copy is in [`examples/workflow.yml`](../examples/workflow.yml).

With `--baseline` you also get "no worse than before" thresholds: cost per run, run
length, fitness drop. Absolute thresholds need re-tuning after every release; relative
ones do not.

## Why not infer the model from the log

An inductive miner would produce a process model from the same log you are checking. That
model contains the norm, not the intent — and the norm is exactly what you were trying to
differ from. The value here is in the *declared* expectation, written by a person who
knows what the agent was supposed to do.
