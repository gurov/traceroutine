"""Словарь OpenTelemetry GenAI semantic conventions — общий язык всех адаптеров.

Почему отдельный модуль, а не метод адаптера. Задача адаптера — разобрать ФОРМАТ
(OTLP/JSON, JSONL Claude Code, дамп Langfuse). Задача этого файла — понять СМЫСЛ:
где здесь входные токены, где модель, где имя инструмента. Смысл у всех источников
один и тот же, формат — у каждого свой.

Отсюда правило для нового адаптера: разобрать свой формат и разложить значения по
ключам semconv в `RawSpan.attrs`. Больше ничего писать не нужно — `extract()` ниже
подхватит их сам. Именно это держит порог входа нового адаптера на двух методах,
как обещано в `adapters/base.py`.

Порядок псевдонимов важен: первый найденный выигрывает, свежие имена идут первыми.
"""
from __future__ import annotations

from typing import Any

from .model import RawSpan

TOKENS_IN = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",       # устаревшее, но живо в проде
    "llm.token_count.prompt",           # OpenInference / Arize
)
TOKENS_OUT = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
)
TOKENS_CACHED = (
    "gen_ai.usage.cache_read_input_tokens",
    "gen_ai.usage.cached_input_tokens",
    "gen_ai.usage.cache_read_tokens",
    "llm.token_count.prompt_cache_hit",
)
# Запись в кеш. В GenAI semconv стандарта пока нет, а деньги вполне реальные:
# у Anthropic это 1.25× от input для 5-минутного TTL и 2× для часового.
# Разделение по TTL встречается только у Anthropic — под него отдельные ключи.
TOKENS_CACHE_WRITE = (
    "gen_ai.usage.cache_creation_input_tokens",
    "gen_ai.usage.cache_write_tokens",
    "llm.token_count.prompt_cache_write",
)
TOKENS_CACHE_WRITE_5M = ("anthropic.usage.cache_creation.ephemeral_5m_input_tokens",)
TOKENS_CACHE_WRITE_1H = ("anthropic.usage.cache_creation.ephemeral_1h_input_tokens",)

MODEL = ("gen_ai.response.model", "gen_ai.request.model", "llm.model_name")
TOOL = ("gen_ai.tool.name", "tool.name")
AGENT = ("gen_ai.agent.name", "agent.name")
SESSION = ("gen_ai.conversation.id", "session.id", "gen_ai.session.id")
# Задача — один запрос пользователя и вся работа агента по нему. Для агентов это
# куда более осмысленный process instance, чем сессия: сессия — это рабочий день,
# в ней десяток независимых задач подряд, и как единый «прогон процесса» она
# бессмысленна. Стандарта в semconv нет, поэтому свой ключ первым.
TASK = ("traceroutine.task.id", "gen_ai.task.id", "task.id")
OPERATION = ("gen_ai.operation.name",)

# Маркер ОПЕРАЦИИ, а не любой gen_ai.*-атрибут. Спаны вроде agent.run несут только
# agent.name/conversation.id — это контейнеры, не шаги процесса.
OP_MARKERS = ("gen_ai.operation.name", "gen_ai.tool.name", "tool.name",
              "llm.request.type", "llm.model_name")


def first(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in attrs and attrs[k] is not None:
            return attrs[k]
    return None


def first_int(attrs: dict[str, Any], keys: tuple[str, ...]) -> int:
    v = first(attrs, keys)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def is_operation(span: RawSpan) -> bool:
    return any(k in span.attrs for k in OP_MARKERS)


def activity_name(span_name: str, attrs: dict[str, Any]) -> str:
    """Имя активности: сначала семантика, потом имя спана.

    Имена спанов у разных инструментов сильно разные ("chat gpt-4o", "ChatAnthropic",
    "execute_tool search"), а semconv-атрибуты — единообразны. Поэтому предпочитаем их.
    """
    tool = first(attrs, TOOL)
    if tool:
        return f"tool:{tool}"
    op = first(attrs, OPERATION)
    if op:
        model = first(attrs, MODEL)
        return f"{op}:{model}" if model else str(op)
    return span_name or "unknown"


def extract(span: RawSpan) -> dict[str, Any]:
    """RawSpan → плоские поля, которые нужны normalize. Единая точка для всех адаптеров."""
    a = span.attrs
    tcw_5m = first_int(a, TOKENS_CACHE_WRITE_5M)
    tcw_1h = first_int(a, TOKENS_CACHE_WRITE_1H)
    total = first_int(a, TOKENS_CACHE_WRITE)
    # Источник может дать либо общее число, либо разбивку, либо и то и другое.
    # Если разбивки нет — считаем весь объём пятиминутным: это дефолтный TTL, и
    # ошибка тут в СТОРОНУ ЗАНИЖЕНИЯ цены, что честнее, чем завысить счёт.
    if not (tcw_5m or tcw_1h):
        tcw_5m = total
    return {
        "activity_raw": activity_name(span.name, a),
        "agent": first(a, AGENT) or span.resource_attrs.get("service.name"),
        "resource": first(a, MODEL) or first(a, TOOL),
        "session_id": first(a, SESSION),
        "task_id": first(a, TASK),
        "tokens_in": first_int(a, TOKENS_IN),
        "tokens_out": first_int(a, TOKENS_OUT),
        "tokens_cached": first_int(a, TOKENS_CACHED),
        "tokens_cache_write_5m": tcw_5m,
        "tokens_cache_write_1h": tcw_1h,
    }


# --- соглашение о том, включён ли кеш в input_tokens ---------------------------
# Это НЕ мелочь: ошибка здесь либо теряет входные токены, либо считает кеш по полной
# ставке вместо 0.1× — то есть завышает счёт примерно в десять раз на кешированной
# части, а на агентных трейсах кеш и есть почти весь вход.
#
# Соглашение задаёт ПРОВАЙДЕР, а не инструментация:
#   OpenAI:    prompt_tokens ВКЛЮЧАЕТ prompt_tokens_details.cached_tokens → вычитать
#   Anthropic: input_tokens НЕ включает cache_read_input_tokens          → не трогать
# Поэтому угадывать по величинам (`кеш больше входа — значит включён`) нельзя: у
# Anthropic кеш штатно больше входа в тысячи раз, и такая эвристика молча обнуляет вход.
INPUT_INCLUDES_CACHE = ("traceroutine.usage.input_includes_cache",)

_FOLDED_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt", "text-embedding")
_EXCLUSIVE_PREFIXES = ("claude-", "anthropic.claude", "gemini-", "models/gemini")


def input_includes_cache(attrs: dict[str, Any], model: str | None) -> bool | None:
    """True — вычитать кеш из input, False — не трогать, None — источник не опознан."""
    declared = first(attrs, INPUT_INCLUDES_CACHE)
    if declared is not None:
        return bool(declared)
    name = (model or "").lower().lstrip("/")
    if any(name.startswith(p) for p in _FOLDED_PREFIXES):
        return True
    if any(name.startswith(p) for p in _EXCLUSIVE_PREFIXES):
        return False
    return None
