from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

from fastapi import HTTPException


@dataclass
class _RateLimitEntry:
    day: str
    count: int


class InMemoryDailyRateLimiter:
    def __init__(self, limit_per_day: int) -> None:
        if limit_per_day < 1:
            raise ValueError("limit_per_day must be >= 1")
        self._limit_per_day = limit_per_day
        self._entries: dict[str, _RateLimitEntry] = {}
        self._lock = Lock()

    def check_and_consume(self, ip_address: str) -> None:
        today = datetime.now(timezone.utc).date().isoformat()

        with self._lock:
            entry = self._entries.get(ip_address)

            if entry is None or entry.day != today:
                entry = _RateLimitEntry(day=today, count=0)
                self._entries[ip_address] = entry

            if entry.count >= self._limit_per_day:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Daily request limit reached for this IP. "
                        f"Allowed: {self._limit_per_day} evaluations per day."
                    ),
                )

            entry.count += 1

            # Lightweight cleanup for stale entries from previous days.
            stale_ips = [ip for ip, item in self._entries.items() if item.day != today]
            for stale_ip in stale_ips:
                self._entries.pop(stale_ip, None)
