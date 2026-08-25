"""Хранилище: parquet через DuckDB.

DuckDB выбран потому, что миллионы спанов должны обрабатываться на ноутбуке без
сервера, а анализ удобнее писать на SQL. attrs сериализуются в JSON-строку —
структура у них произвольная, колоночного смысла в ней нет.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import duckdb

from .model import COLUMNS, Event

_DDL = """
CREATE OR REPLACE TABLE events (
    case_id VARCHAR, event_id VARCHAR, activity_raw VARCHAR, activity VARCHAR,
    ts_start DOUBLE, ts_end DOUBLE, parent_id VARCHAR, agent VARCHAR, resource VARCHAR,
    tokens_in BIGINT, tokens_cached BIGINT, tokens_cache_write BIGINT,
    tokens_out BIGINT, cost_usd DOUBLE,
    status VARCHAR, error_type VARCHAR, attrs VARCHAR
)
"""


def write(events: Iterable[Event], path: Path) -> int:
    con = duckdb.connect()
    con.execute(_DDL)
    rows, n = [], 0
    for ev in events:
        row = list(ev.as_row())
        row[COLUMNS.index("attrs")] = json.dumps(row[COLUMNS.index("attrs")], ensure_ascii=False)
        rows.append(row)
        n += 1
        if len(rows) >= 10_000:
            con.executemany(f"INSERT INTO events VALUES ({','.join('?' * len(COLUMNS))})", rows)
            rows.clear()
    if rows:
        con.executemany(f"INSERT INTO events VALUES ({','.join('?' * len(COLUMNS))})", rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute("COPY events TO ? (FORMAT PARQUET)", [str(path)])
    con.close()
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
