"""RawSpan → Event.

Здесь живут три решения, которые определяют вообще всё в отчёте.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from .model import Event, RawSpan
from .pricing import Pricing
from .semconv import extract, input_includes_cache, is_operation

# Флаттенинг: трейс — дерево, event log — плоская последовательность.
#   genai  — только спаны с семантикой GenAI (вызовы модели и инструментов). Дефолт:
#            даёт процесс агента, а не процесс HTTP-обвязки вокруг него.
#   leaves — только листья дерева.
#   all    — все спаны (обычно шум, полезно при отладке адаптера).
FLATTEN = ("genai", "leaves", "all")

# Case notion — ГЛАВНОЕ модельное решение. Из одних и тех же данных
# trace / session / task дают три РАЗНЫХ процесса, и разница не косметическая:
# на реальных транскриптах Claude Code сессия как кейс даёт 61 кейс и 61 уникальный
# путь — то есть вариантный анализ вырождается полностью, потому что двух одинаковых
# рабочих дней не бывает. Задача (один запрос пользователя) — единица, которая
# действительно повторяется.
CASE_NOTIONS = ("trace", "session", "task")


def normalize(
    spans: Iterable[RawSpan],
    *,
    case_notion: str = "trace",
    flatten: str = "genai",
    pricing: Pricing | None = None,
    unknown_cache_convention: set[str] | None = None,
) -> Iterator[Event]:
    if case_notion not in CASE_NOTIONS and not case_notion.startswith("attr:"):
        raise SystemExit(
            f"case notion {case_notion!r} is not supported; "
            f"available: {', '.join(CASE_NOTIONS)} or attr:<attribute name>"
        )
    if flatten not in FLATTEN:
        raise SystemExit(f"flatten {flatten!r}; available: {', '.join(FLATTEN)}")

    pricing = pricing or Pricing()
    spans = list(spans)
    # Источники, для которых соглашение о кеше неизвестно: складываем в переданное
    # множество, чтобы CLI мог честно сказать «цена этих вызовов может быть неточной»,
    # а не молча соврать. normalize — генератор, вернуть их значением нельзя.
    unknown_convention = unknown_cache_convention if unknown_cache_convention is not None else set()
    has_child = {s.parent_id for s in spans if s.parent_id}

    # Сессия обычно проставлена не на каждом спане, а на корневом. Поднимаем её
    # по дереву, иначе половина событий уедет в case_id=None.
    by_id = {s.span_id: s for s in spans}
    caches: dict[str, dict[str, str | None]] = {"session_id": {}, "task_id": {}}

    def inherited(span: RawSpan, field: str, depth: int = 0) -> str | None:
        cache = caches[field]
        if span.span_id in cache:
            return cache[span.span_id]
        own = extract(span)[field]
        if own is None and span.parent_id and depth < 64:
            parent = by_id.get(span.parent_id)
            own = inherited(parent, field, depth + 1) if parent else None
        cache[span.span_id] = own
        return own

    for span in spans:
        if flatten == "genai" and not is_operation(span):
            continue
        if flatten == "leaves" and span.span_id in has_child:
            continue

        f = extract(span)

        if case_notion == "trace":
            case_id = span.trace_id
        elif case_notion == "session":
            case_id = inherited(span, "session_id") or span.trace_id  # не терять события
        elif case_notion == "task":
            # Фолбэк на сессию, а не на trace: события без задачи логичнее собрать
            # в кейс своей сессии, чем размазать по одиночным кейсам.
            case_id = (inherited(span, "task_id") or inherited(span, "session_id")
                       or span.trace_id)
        else:
            case_id = str(span.attrs.get(case_notion[5:]) or span.trace_id)

        tin, tout, tcached = f["tokens_in"], f["tokens_out"], f["tokens_cached"]
        tcw_5m, tcw_1h = f["tokens_cache_write_5m"], f["tokens_cache_write_1h"]
        # Частая ловушка: часть провайдеров кладёт кеш ВНУТРЬ input_tokens (OpenAI),
        # часть — рядом (Anthropic). См. semconv.input_includes_cache.
        folded = input_includes_cache(span.attrs, f["resource"])
        if folded is None and (tcached or tin):
            unknown_convention.add(f["resource"] or span.name)
        if folded and tcached and tin:
            tin = max(0, tin - tcached)

        yield Event(
            case_id=case_id or "unknown",
            event_id=span.span_id,
            activity_raw=f["activity_raw"],
            activity=f["activity_raw"],       # слой Abstract (Ц2) перезапишет
            ts_start=span.ts_start,
            ts_end=span.ts_end,
            parent_id=span.parent_id,
            agent=f["agent"],
            resource=f["resource"],
            tokens_in=tin,
            tokens_cached=tcached,
            tokens_cache_write=tcw_5m + tcw_1h,
            tokens_out=tout,
            cost_usd=pricing.cost(f["resource"], tin, tout, tcached, tcw_5m, tcw_1h),
            status=span.status,
            error_type=span.error_type,
            attrs=span.attrs,
        )
