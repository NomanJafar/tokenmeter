"""One SQLite row per request. No prompt or completion text is ever stored."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, tenant TEXT NOT NULL, model TEXT, upstream TEXT,
  prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0,
  usage_estimated INTEGER NOT NULL DEFAULT 0, ttft_ms REAL, e2e_ms REAL, status INTEGER
);
CREATE INDEX IF NOT EXISTS ix_requests_tenant_ts ON requests(tenant, ts);
CREATE INDEX IF NOT EXISTS ix_requests_ts ON requests(ts);
"""


def _p50(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * 0.5
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _p95(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)

    def record(self, ts: float, tenant: str, model: str | None, upstream: str | None, prompt_tokens: int,
               completion_tokens: int, usage_estimated: bool, ttft_ms: float | None, e2e_ms: float | None, status: int) -> None:
        self.conn.execute(
            "INSERT INTO requests(ts, tenant, model, upstream, prompt_tokens, completion_tokens, usage_estimated, ttft_ms, e2e_ms, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts, tenant, model, upstream, int(prompt_tokens or 0), int(completion_tokens or 0), int(bool(usage_estimated)), ttft_ms, e2e_ms, status))
        self.conn.commit()

    def per_tenant(self) -> dict[str, dict]:
        rows = self.conn.execute("SELECT tenant, prompt_tokens, completion_tokens, usage_estimated, ttft_ms, status FROM requests").fetchall()
        out: dict[str, dict] = {}
        for tenant, p, c, est, ttft, status in rows:
            a = out.setdefault(tenant, {"requests": 0, "errors": 0, "estimated": 0, "prompt_tokens": 0, "completion_tokens": 0, "_ttft": []})
            a["requests"] += 1
            a["errors"] += int(status is None or status >= 400)
            a["estimated"] += int(est)
            a["prompt_tokens"] += p
            a["completion_tokens"] += c
            if ttft is not None:
                a["_ttft"].append(ttft)
        for a in out.values():
            a["ttft_p50_ms"], a["ttft_p95_ms"] = _p50(a["_ttft"]), _p95(a["_ttft"])
            del a["_ttft"]
        return out

    def totals(self) -> dict:
        r = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), MIN(ts), MAX(ts), "
                              "SUM(CASE WHEN status>=400 OR status IS NULL THEN 1 ELSE 0 END) FROM requests").fetchone()
        return {"requests": r[0], "prompt_tokens": r[1], "completion_tokens": r[2], "first_ts": r[3], "last_ts": r[4], "errors": r[5] or 0}

    def buckets(self, seconds: float = 60) -> list[dict]:
        rows = self.conn.execute("SELECT CAST(ts / ? AS INTEGER) * ? AS b, SUM(prompt_tokens), SUM(completion_tokens), COUNT(*), AVG(e2e_ms) "
                                 "FROM requests GROUP BY b ORDER BY b", (seconds, seconds)).fetchall()
        return [{"t": float(b), "prompt_tokens": p, "completion_tokens": c, "requests": n, "e2e_ms_mean": e} for b, p, c, n, e in rows]
