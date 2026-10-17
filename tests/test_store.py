from tokenmeter.store import Store


def test_store_records_and_aggregates_per_tenant(tmp_path):
    s = Store(tmp_path / "m.db")
    s.record(ts=100.0, tenant="chat", model="m", upstream="u", prompt_tokens=10, completion_tokens=40, usage_estimated=False, ttft_ms=50, e2e_ms=600, status=200)
    s.record(ts=101.0, tenant="chat", model="m", upstream="u", prompt_tokens=20, completion_tokens=60, usage_estimated=False, ttft_ms=70, e2e_ms=900, status=200)
    s.record(ts=102.0, tenant="rag", model="m", upstream="u", prompt_tokens=4000, completion_tokens=50, usage_estimated=True, ttft_ms=400, e2e_ms=1200, status=200)
    s.record(ts=103.0, tenant="rag", model="m", upstream="u", prompt_tokens=0, completion_tokens=0, usage_estimated=True, ttft_ms=None, e2e_ms=30, status=502)
    agg = s.per_tenant()
    assert agg["chat"]["requests"] == 2 and agg["chat"]["prompt_tokens"] == 30 and agg["chat"]["completion_tokens"] == 100
    assert agg["chat"]["ttft_p50_ms"] == 60 and agg["rag"]["errors"] == 1 and agg["rag"]["estimated"] == 2
    tot = s.totals()
    assert tot["prompt_tokens"] == 4030 and tot["completion_tokens"] == 150 and tot["first_ts"] == 100.0 and tot["last_ts"] == 103.0


def test_store_buckets_tokens_over_time(tmp_path):
    s = Store(tmp_path / "m.db")
    for i in range(10):
        s.record(ts=1020 + i * 30, tenant="t", model="m", upstream="u", prompt_tokens=5, completion_tokens=10, usage_estimated=False, ttft_ms=10, e2e_ms=100, status=200)
    b = s.buckets(seconds=60)
    assert len(b) == 5 and all(x["completion_tokens"] == 20 for x in b) and b[0]["t"] == 1020.0
