"""Хранилище: parquet через DuckDB.

DuckDB выбран потому, что миллионы спанов должны обрабатываться на ноутбуке без
сервера, а анализ удобнее писать на SQL. attrs сериализуются в JSON-строку —
структура у них произвольная, колоночного смысла в ней нет.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

import duckdb

from .model import COLUMNS, Event

# Типы колонок объявлены явно, а не выведены: read_json угадывает тип по выборке,
# а колонка, где первые сотни значений null (parent_id, error_type), угадывается
# как VARCHAR лишь по счастливой случайности.
_TYPES = {
    "case_id": "VARCHAR", "event_id": "VARCHAR", "activity_raw": "VARCHAR",
    "activity": "VARCHAR", "ts_start": "DOUBLE", "ts_end": "DOUBLE",
    "parent_id": "VARCHAR", "agent": "VARCHAR", "resource": "VARCHAR",
    "tokens_in": "BIGINT", "tokens_cached": "BIGINT", "tokens_cache_write": "BIGINT",
    "tokens_out": "BIGINT", "cost_usd": "DOUBLE",
    "status": "VARCHAR", "error_type": "VARCHAR", "attrs": "VARCHAR",
}
assert list(_TYPES) == COLUMNS

_DDL = "CREATE OR REPLACE TABLE events ({})".format(
    ", ".join(f"{c} {t}" for c, t in _TYPES.items())
)


def write(events: Iterable[Event], path: Path) -> int:
    """События → parquet.

    Через NDJSON-файл, а не через `executemany`. Причина измеренная, а не
    вкусовая: параметризованный INSERT конвертирует каждое значение по одному
    на стороне Python — 13 тыс. событий вставлялись 48 секунд, те же события
    через read_json — 0.16 с. Ingest целиком упирался в эту одну строчку.
    Заодно исчезает буфер на 10 тыс. строк: события уходят на диск потоком.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = COLUMNS.index("attrs")
    fd, tmp_name = tempfile.mkstemp(suffix=".ndjson", dir=path.parent)
    tmp = Path(tmp_name)
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for ev in events:
                row = list(ev.as_row())
                row[idx] = json.dumps(row[idx], ensure_ascii=False)
                f.write(json.dumps(dict(zip(COLUMNS, row)), ensure_ascii=False))
                f.write("\n")
                n += 1
        con = duckdb.connect()
        try:
            con.execute(_DDL)
            if n:
                # format задан явно: на файле из одной строки автоопределение
                # может решить, что это один JSON-объект, а не поток.
                con.execute(
                    "INSERT INTO events SELECT * FROM "
                    "read_json(?, columns=?, format='newline_delimited')",
                    [str(tmp), _TYPES],
                )
            con.execute("COPY events TO ? (FORMAT PARQUET)", [str(path)])
        finally:
            con.close()
    finally:
        tmp.unlink(missing_ok=True)
    return n


def read(path: Path) -> list[dict]:
    con = duckdb.connect()
    cur = con.execute(
        "SELECT * FROM read_parquet(?) ORDER BY case_id, ts_start, event_id", [str(path)]
    )
    cols = [d[0] for d in cur.description]
    out = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    return out


def cases(events: list[dict]) -> Iterator[tuple[str, list[dict]]]:
    """События уже отсортированы по (case_id, ts_start, event_id).

    Тайбрейк по event_id обязателен: при совпадающих таймстемпах недетерминированный
    порядок расщепляет один и тот же процесс на несколько «вариантов».
    """
    cur_id, buf = None, []
    for ev in events:
        if ev["case_id"] != cur_id:
            if cur_id is not None:
                yield cur_id, buf
            cur_id, buf = ev["case_id"], []
        buf.append(ev)
    if cur_id is not None:
        yield cur_id, buf
