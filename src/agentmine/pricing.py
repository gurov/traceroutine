from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT = Path(__file__).with_name("pricing.yaml")


class Pricing:
    def __init__(self, path: Path | None = None):
        data = yaml.safe_load((path or _DEFAULT).read_text(encoding="utf-8")) or {}
        self.models: dict = data.get("models") or {}
        d = data.get("defaults") or {}
        self.cache_read: float = d.get("cache_read", 0.1)
        self.cache_write_5m: float = d.get("cache_write_5m", 1.25)
        self.cache_write_1h: float = d.get("cache_write_1h", 2.0)
        self.unknown: set[str] = set()

    def _row(self, model: str | None) -> dict | None:
        if not model:
            return None
        if model in self.models:
            return self.models[model]
        # снапшоты вида claude-opus-5-20260101 → префиксное совпадение
        for name, row in self.models.items():
            if model.startswith(name):
                return row
        self.unknown.add(model)
        return None

    def cost(
        self,
        model: str | None,
        tin: int,
        tout: int,
        tcached: int,
        tcw_5m: int = 0,
        tcw_1h: int = 0,
    ) -> float:
        # Спаны инструментов не тратят токенов — не искать их в таблице цен и не
        # ругаться на «неизвестную модель» с именем инструмента.
        if not (tin or tout or tcached or tcw_5m or tcw_1h):
            return 0.0
        row = self._row(model)
        if row is None:
            return 0.0
        # Четыре разные ставки на один и тот же по смыслу «вход». tokens_in НЕ
        # включает ни чтение кеша, ни запись — иначе двойной счёт.
        inp = row["input"]
        return (
            tin * inp / 1e6
            + tcached * inp * self.cache_read / 1e6
            + tcw_5m * inp * self.cache_write_5m / 1e6
            + tcw_1h * inp * self.cache_write_1h / 1e6
            + tout * row["output"] / 1e6
        )
