"""Слой Abstract: сырые метки спанов → семантические активности.

Два уровня, и это принципиально:

1. **Детерминированный** — регулярки. Срезает аргументы, ID, номера попыток.
   Бесплатно, воспроизводимо, покрывает большую часть разнообразия.
2. **LLM** — делает то, чего регулярка не может: решает, что `validate_answer`,
   `self_critique` и `check_citations` — один шаг процесса «verify».

Ключевой приём: LLM работает по множеству РАЗЛИЧНЫХ канонических меток, а не по
событиям. На 2 млн событий различных форм обычно 300-500 — это один запрос, а не
2 млн. Поэтому шаг стоит центы и кешируется.

Ручные правки в activity_map.yaml имеют приоритет над обеими уровнями и переживают
перегенерацию: словарь — версионируемый артефакт пользователя, а не наш вывод.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Protocol

import yaml

MODEL_DEFAULT = "claude-opus-5"
OLLAMA_DEFAULT = "qwen3:14b"
GEMINI_DEFAULT = "gemini-3.1-pro-preview"
CHUNK = 400          # выше этого LLM начинает терять метки — режем и мержим вторым проходом

# --- уровень 1: детерминированный -------------------------------------------

_ARGS = re.compile(r"\s*\([^()]*\)")            # search(query='x') → search
_KV = re.compile(r"\s*\[[^\[\]]*\]")            # search[k=8] → search
_ID_TAIL = re.compile(r"[-_/]\d[\w.]*$")        # read_document_4471 → read_document
_ATTEMPT = re.compile(r"[-_](attempt|try|retry|n)[-_]?\d+$", re.I)
# Путь/файл как аргумент: `python /testbed/reproduce_error.py` → `python`.
# На открытом корпусе SWE-agent без этого одна операция «запустить скрипт»
# рассыпалась на два десятка меток — по одной на имя временного файла.
_PATH_ARG = re.compile(r"[\s:=]+[~./]?[\w.-]*/[\w./-]*")
_WS = re.compile(r"\s+")


def canonical(label: str) -> str:
    """Убирает всё, что зависит от аргументов вызова, но не от самой операции."""
    s = label
    for _ in range(3):                          # вложенные скобки за несколько проходов
        s2 = _KV.sub("", _ARGS.sub("", s))
        if s2 == s:
            break
        s = s2
    s = _ATTEMPT.sub("", s)
    s = _PATH_ARG.sub("", s)
    s = _ID_TAIL.sub("", s)
    s = _WS.sub(" ", s).strip(" :/-_")

    # Модель — это РЕСУРС, а не активность. В классическом process mining это разные
    # измерения: активность отвечает «что произошло», ресурс — «кем/чем сделано».
    # Информация не теряется: Event.resource хранит модель, отчёт даёт разбивку по ней.
    if ":" in s:
        prefix, _, rest = s.partition(":")
        if prefix in ("chat", "completion", "embedding", "text_completion"):
            return prefix
        return f"{prefix}:{rest}" if rest else prefix
    return s or "unknown"


# --- бэкенды ----------------------------------------------------------------

@dataclass
class Cluster:
    name: str
    labels: list[str]


SYSTEM = """You group raw span labels from an LLM-agent trace into semantic process \
activities for process mining.

An ACTIVITY answers "what step of the process happened". It must NOT encode:
- arguments, IDs, queries, indexes, or file names
- which model or vendor served the call
- retry attempt numbers

Merge aggressively across DIFFERENT operation names when they are the same process step.
This is the part that matters most:
  retrieve  <- vector_search, bm25_lookup, keyword_search, rerank, fetch_url, read_document
  verify    <- validate_answer, self_critique, check_citations, grade_response
  plan      <- plan_steps, decompose_task, pick_tool, route
  respond   <- write_answer, format_response, emit_final
  notify    <- send_email, notify_slack, page_oncall
  remember  <- memory_read, memory_write, load_context, save_summary

Names: short snake_case verb phrases. Every input label must appear in exactly one \
activity. Prefer too few activities over too many: a process graph with more than a \
dozen activities is unreadable, which defeats the purpose."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "activities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "labels"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["activities"],
    "additionalProperties": False,
}


def _user_msg(labels: list[str], target: int) -> str:
    return (
        f"Group these {len(labels)} labels into AT MOST {target} activities.\n\n"
        + "\n".join(labels)
    )


class Backend(Protocol):
    name: str

    def cluster(self, labels: list[str], target: int) -> list[Cluster]: ...


class NoBackend:
    """Только детерминированный уровень. Дефолт, когда нет ни ключа, ни Ollama."""
    name = "none"

    def cluster(self, labels: list[str], target: int) -> list[Cluster]:
        return [Cluster(name=l, labels=[l]) for l in labels]


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = MODEL_DEFAULT, batch: bool = False):
        try:
            import anthropic
        except ImportError:
            raise SystemExit("the anthropic package is required: uv pip install anthropic")
        self.model = model
        self.batch = batch
        self._an = anthropic
        self.client = anthropic.Anthropic()
        self.usage: list[dict] = []

    def _params(self, labels: list[str], target: int) -> dict:
        return dict(
            model=self.model,
            max_tokens=16000,
            # Стабильный префикс идёт первым, переменная часть (метки) — после него.
            # Кеш окупается на замерах стабильности: там N одинаковых префиксов подряд.
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": _user_msg(labels, target)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            thinking={"type": "adaptive"},
        )

    def cluster(self, labels: list[str], target: int) -> list[Cluster]:
        if self.batch:
            return self._cluster_batch(labels, target)
        resp = self.client.messages.create(**self._params(labels, target))
        self._record(resp)
        return _parse(next(b.text for b in resp.content if b.type == "text"))

    def _cluster_batch(self, labels: list[str], target: int) -> list[Cluster]:
        import time

        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        batch = self.client.messages.batches.create(requests=[
            Request(custom_id="cluster",
                    params=MessageCreateParamsNonStreaming(**self._params(labels, target)))
        ])
        while True:
            got = self.client.messages.batches.retrieve(batch.id)
            if got.processing_status == "ended":
                break
            time.sleep(30)
        for res in self.client.messages.batches.results(batch.id):
            if res.result.type != "succeeded":
                raise SystemExit(f"batch failed: {res.result.type}")
            self._record(res.result.message)
            return _parse(next(b.text for b in res.result.message.content if b.type == "text"))
        raise SystemExit("batch returned no results")

    def _record(self, resp) -> None:
        u = resp.usage
        self.usage.append({
            "input": u.input_tokens, "output": u.output_tokens,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        })


class OllamaBackend:
    """Локальный путь. Нужен не ради экономии, а ради приватности: трейсы содержат
    промпты, пользовательские данные и аргументы инструментов."""
    name = "ollama"

    def __init__(self, model: str = OLLAMA_DEFAULT, host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def cluster(self, labels: list[str], target: int) -> list[Cluster]:
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "system": SYSTEM,
            "prompt": _user_msg(labels, target),
            "stream": False, "think": False, "format": _SCHEMA,
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 8192},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as fh:
            return _parse(json.loads(fh.read())["response"])


def _gemini_schema(schema: dict) -> dict:
    """responseSchema у Gemini — подмножество OpenAPI, а не JSON Schema.

    Отличия, на которых спотыкаются: не поддерживается `additionalProperties`
    (запрос падает с 400), а порядок ключей задаётся через `propertyOrdering`.
    """
    out = {}
    for k, v in schema.items():
        if k == "additionalProperties":
            continue
        if k == "properties":
            out[k] = {pk: _gemini_schema(pv) for pk, pv in v.items()}
            out["propertyOrdering"] = list(v)
        elif k == "items":
            out[k] = _gemini_schema(v)
        else:
            out[k] = v
    return out


class GeminiBackend:
    """Google Gemini через REST — без дополнительной зависимости, как и Ollama.

    Ключ берётся из GEMINI_API_KEY (или GOOGLE_API_KEY) и нигде не сохраняется.
    """
    name = "gemini"

    def __init__(self, model: str = GEMINI_DEFAULT, api_key: str | None = None):
        self.model = model
        self.key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) must be set in the environment")
        self.usage: list[dict] = []

    def cluster(self, labels: list[str], target: int) -> list[Cluster]:
        import urllib.error
        import urllib.request

        body = json.dumps({
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": _user_msg(labels, target)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _gemini_schema(_SCHEMA),
                "temperature": 0,
                "maxOutputTokens": 32768,
            },
        }).encode()
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as fh:
                data = json.loads(fh.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise SystemExit(f"Gemini {e.code}: {detail}")

        u = data.get("usageMetadata") or {}
        self.usage.append({
            "input": u.get("promptTokenCount", 0),
            "output": u.get("candidatesTokenCount", 0),
            "cache_read": u.get("cachedContentTokenCount", 0), "cache_write": 0,
        })
        cands = data.get("candidates") or []
        if not cands:
            raise SystemExit(f"Gemini returned no candidates: {str(data)[:300]}")
        c0 = cands[0]
        if c0.get("finishReason") not in (None, "STOP"):
            # MAX_TOKENS здесь означает обрезанный JSON, то есть потерянные метки.
            raise SystemExit(f"Gemini truncated the response: finishReason={c0['finishReason']}")
        parts = (c0.get("content") or {}).get("parts") or []
        return _parse("".join(p.get("text", "") for p in parts))


def _parse(text: str) -> list[Cluster]:
    """Терпим к форме ответа: у Anthropic её гарантирует output_config, у локальных
    бэкендов — как повезёт. Молча отдавать пустоту нельзя: слой деградирует до правил,
    и это заметно только по метрике покрытия."""
    data = json.loads(text)
    out: list[Cluster] = []

    for a in data.get("activities") or []:
        name = str(a.get("name") or "").strip()
        labels = [str(x) for x in (a.get("labels") or [])]
        if name and labels:
            out.append(Cluster(name=name, labels=labels))
    if out:
        return out

    # Альтернатива: плоский словарь {метка: активность}.
    flat = data.get("mapping") if isinstance(data.get("mapping"), dict) else (
        data if all(isinstance(v, str) for v in data.values()) else None)
    if flat:
        inv: dict[str, list[str]] = {}
        for label, act in flat.items():
            inv.setdefault(str(act), []).append(str(label))
        return [Cluster(n, ls) for n, ls in inv.items()]
    return out


# --- словарь ----------------------------------------------------------------

@dataclass
class ActivityMap:
    mapping: dict[str, str] = field(default_factory=dict)     # каноническая метка → активность
    overrides: dict[str, str] = field(default_factory=dict)   # правки человека, приоритетны
    backend: str = "none"
    generated: str = ""
    coverage: dict = field(default_factory=dict)

    def activity(self, raw_label: str) -> str:
        c = canonical(raw_label)
        return self.overrides.get(raw_label) or self.overrides.get(c) \
            or self.mapping.get(c) or c

    @classmethod
    def load(cls, path: Path) -> "ActivityMap":
        if not path.exists():
            return cls()
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            mapping=d.get("mapping") or {},
            overrides=d.get("overrides") or {},
            backend=d.get("backend", "none"),
            generated=d.get("generated", ""),
            coverage=d.get("coverage") or {},
        )

    def save(self, path: Path) -> None:
        head = (
            "# traceroutine activity vocabulary — a versioned project artifact.\n"
            "#\n"
            "# `overrides` is hand-edited, wins over `mapping`, and SURVIVES\n"
            "# regeneration. `mapping` is overwritten on every `traceroutine abstract`.\n"
            "# Edit overrides, not mapping.\n"
        )
        body = yaml.safe_dump(
            {"version": 1, "generated": self.generated, "backend": self.backend,
             "coverage": self.coverage, "overrides": self.overrides,
             "mapping": dict(sorted(self.mapping.items()))},
            allow_unicode=True, sort_keys=False, width=100,
        )
        path.write_text(head + body, encoding="utf-8")


def build(raw_labels: list[str], backend: Backend, target: int = 12,
          existing: ActivityMap | None = None) -> ActivityMap:
    """Сырые метки → словарь. Уровень 1 всегда, уровень 2 — если бэкенд не `none`."""
    canon = sorted({canonical(l) for l in raw_labels})
    amap = ActivityMap(
        overrides=dict(existing.overrides) if existing else {},
        backend=f"{backend.name}:{getattr(backend, 'model', '-')}",
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    clusters: list[Cluster] = []
    for i in range(0, len(canon), CHUNK):
        clusters += backend.cluster(canon[i:i + CHUNK], target)
    if len(canon) > CHUNK and backend.name != "none":
        # Второй проход: имена активностей из разных чанков тоже надо слить.
        names = sorted({c.name for c in clusters})
        merged = {l: c.name for c in backend.cluster(names, target) for l in c.labels}
        clusters = [Cluster(merged.get(c.name, c.name), c.labels) for c in clusters]

    for c in clusters:
        for l in c.labels:
            amap.mapping.setdefault(l, c.name)

    # Ремонт: метки, которые модель потеряла или выдумала.
    missing = [l for l in canon if l not in amap.mapping]
    for l in missing:
        amap.mapping[l] = l                     # честный фолбэк на уровень 1
    invented = [l for l in amap.mapping if l not in set(canon)]
    for l in invented:
        del amap.mapping[l]

    amap.coverage = {
        "raw_labels": len(set(raw_labels)),
        "canonical_labels": len(canon),
        "activities": len(set(amap.mapping.values())),
        "recovered_by_rules": len(set(raw_labels)) - len(canon),
        "missing_from_llm": len(missing),
        "invented_by_llm": len(invented),
    }
    return amap


def audit(amap: "ActivityMap", events: list[dict]) -> list[str]:
    """Проверки качества словаря, которые счётчик активностей поймать не может.

    Главная: **слияние платного шага с бесплатным**. Модель может уложиться в целевое
    число активностей и при этом слить вызов модели (где лежат все деньги) с
    форматированием ответа (где их нет). Формально цель достигнута, фактически
    атрибуция стоимости уничтожена: граф скажет «respond стоит $9», и это бесполезно.
    """
    cost: dict[str, float] = {}
    calls: dict[str, int] = {}
    for e in events:
        c = canonical(e["activity_raw"])
        cost[c] = cost.get(c, 0.0) + (e["cost_usd"] or 0.0)
        calls[c] = calls.get(c, 0) + 1

    # ЭФФЕКТИВНАЯ группировка: через amap.activity(), а не по сырому mapping.
    # Иначе проверка игнорирует overrides и продолжает ругаться на уже починенное —
    # предупреждение, которое нельзя погасить, обесценивает все остальные.
    groups: dict[str, list[str]] = {}
    for label in amap.mapping:
        groups.setdefault(amap.activity(label), []).append(label)

    total = sum(cost.values())
    out: list[str] = []
    for act, labels in sorted(groups.items()):
        priced = [l for l in labels if cost.get(l, 0.0) > 0]
        free = [l for l in labels if l in cost and cost.get(l, 0.0) == 0]
        if priced and free:
            share = sum(cost.get(l, 0.0) for l in priced) / total if total else 0
            out.append(
                f"\"{act}\" merges paid steps ({', '.join(sorted(priced))}) with free "
                f"ones ({', '.join(sorted(free))}). The paid ones carry {share:.0%} of "
                f"all spend — once merged they cannot be told apart. Split them via "
                f"overrides."
            )
    return out


def stability(raw_labels: list[str], backend: Backend, runs: int = 3,
              target: int = 12) -> float:
    """Rand index между прогонами: доля пар меток, одинаково разведённых или сведённых.

    Метрика инвариантна к именам кластеров — важно, потому что модель вольна назвать
    один и тот же кластер по-разному. Нестабильность здесь = потеря доверия к отчёту.
    """
    canon = sorted({canonical(l) for l in raw_labels})
    parts = [build(canon, backend, target).mapping for _ in range(runs)]
    pairs = list(combinations(canon, 2))
    if not pairs:
        return 1.0
    scores = []
    for a, b in combinations(parts, 2):
        agree = sum(
            (a.get(x) == a.get(y)) == (b.get(x) == b.get(y)) for x, y in pairs
        )
        scores.append(agree / len(pairs))
    return sum(scores) / len(scores) if scores else 1.0
