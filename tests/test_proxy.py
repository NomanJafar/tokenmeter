import json

import httpx

from tokenmeter.proxy import create_app
from tokenmeter.store import Store


def _body(stream, max_tokens=6, usage=True):
    b = {"model": "m", "messages": [{"role": "user", "content": "one two three"}], "max_tokens": max_tokens, "stream": stream, "ignore_eos": True}
    if stream and usage:
        b["stream_options"] = {"include_usage": True}
    return b


def test_non_streaming_request_is_metered_with_upstream_usage(upstream, live_app, tmp_path):
    db = tmp_path / "m.db"
    with live_app(create_app(upstream, db)) as url:
        r = httpx.post(f"{url}/v1/chat/completions", json=_body(False), headers={"x-tenant": "chat"}, timeout=30)
        assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 6
    agg = Store(db).per_tenant()
    assert agg["chat"]["requests"] == 1 and agg["chat"]["prompt_tokens"] == 3 and agg["chat"]["completion_tokens"] == 6
    assert agg["chat"]["estimated"] == 0 and agg["chat"]["ttft_p50_ms"] is not None


def test_streaming_request_passes_through_and_reads_usage_from_final_chunk(upstream, live_app, tmp_path):
    db = tmp_path / "m.db"
    with live_app(create_app(upstream, db)) as url:
        with httpx.stream("POST", f"{url}/v1/chat/completions", json=_body(True), headers={"x-tenant": "rag"}, timeout=30) as r:
            lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
        content = [json.loads(ln[5:]) for ln in lines if "[DONE]" not in ln]
        assert sum(1 for c in content if c["choices"] and c["choices"][0]["delta"].get("content")) == 6
    agg = Store(db).per_tenant()
    assert agg["rag"]["completion_tokens"] == 6 and agg["rag"]["prompt_tokens"] == 3 and agg["rag"]["estimated"] == 0


def test_streaming_without_usage_falls_back_to_counting_and_marks_estimated(upstream, live_app, tmp_path):
    db = tmp_path / "m.db"
    with live_app(create_app(upstream, db)) as url, \
         httpx.stream("POST", f"{url}/v1/chat/completions", json=_body(True, usage=False), timeout=30) as r:
        list(r.iter_lines())
    agg = Store(db).per_tenant()
    assert agg["anonymous"]["completion_tokens"] == 6 and agg["anonymous"]["estimated"] == 1


def test_upstream_error_is_recorded_with_status(live_app, tmp_path):
    db = tmp_path / "m.db"
    with live_app(create_app("http://127.0.0.1:9", db)) as url:   # nothing listens on port 9
        r = httpx.post(f"{url}/v1/chat/completions", json=_body(False), headers={"x-tenant": "t"}, timeout=30)
        assert r.status_code == 502
    agg = Store(db).per_tenant()
    assert agg["t"]["requests"] == 1 and agg["t"]["errors"] == 1


def test_metrics_endpoint_summarises_tenants(upstream, live_app, tmp_path):
    with live_app(create_app(upstream, tmp_path / "m.db")) as url:
        httpx.post(f"{url}/v1/chat/completions", json=_body(False), headers={"x-tenant": "chat"}, timeout=30)
        m = httpx.get(f"{url}/metrics").text
        assert 'tokenmeter_completion_tokens_total{tenant="chat"} 6' in m
