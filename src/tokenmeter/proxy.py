"""Metering proxy: forwards /v1/chat/completions to one upstream, streams the response
through unchanged, and records tenant, tokens and latency per request (CONCEPTS.md section 4)."""

from __future__ import annotations

import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from tokenmeter.store import Store


def _usage_from_sse(buf: bytes) -> tuple[dict | None, int]:
    """Scan buffered SSE bytes: return (usage dict if a chunk carried one, number of content chunks)."""
    usage, content = None, 0
    for line in buf.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            if (ch.get("delta") or {}).get("content"):
                content += 1
    return usage, content


def create_app(upstream: str, db_path, tenant_header: str = "x-tenant") -> FastAPI:
    app = FastAPI(title="tokenmeter")
    store = Store(db_path)
    client = httpx.AsyncClient(timeout=None)
    upstream = upstream.rstrip("/")

    @app.get("/health")
    async def health():
        return {"status": "ok", "upstream": upstream}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        lines = []
        for tenant, a in store.per_tenant().items():
            lines += [f'tokenmeter_requests_total{{tenant="{tenant}"}} {a["requests"]}',
                      f'tokenmeter_errors_total{{tenant="{tenant}"}} {a["errors"]}',
                      f'tokenmeter_prompt_tokens_total{{tenant="{tenant}"}} {a["prompt_tokens"]}',
                      f'tokenmeter_completion_tokens_total{{tenant="{tenant}"}} {a["completion_tokens"]}']
        return "\n".join(lines) + "\n"

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        tenant = request.headers.get(tenant_header) or "anonymous"
        body = await request.json()
        model = body.get("model")
        t0 = time.monotonic()
        ts = time.time()
        try:
            req = client.build_request("POST", f"{upstream}/v1/chat/completions", json=body)
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            store.record(ts, tenant, model, upstream, 0, 0, True, None, (time.monotonic() - t0) * 1000, 502)
            return JSONResponse({"error": f"upstream unreachable: {type(e).__name__}"}, status_code=502)
        if resp.status_code != 200:
            data = await resp.aread()
            await resp.aclose()
            store.record(ts, tenant, model, upstream, 0, 0, True, None, (time.monotonic() - t0) * 1000, resp.status_code)
            return JSONResponse({"error": data.decode(errors="replace")[:500]}, status_code=resp.status_code)

        if body.get("stream"):
            async def gen():
                buf = bytearray()
                ttft = None
                try:
                    async for chunk in resp.aiter_raw():
                        if ttft is None and b'"content":"' in chunk.replace(b" ", b"") and b'"content":""' not in chunk.replace(b" ", b""):
                            ttft = (time.monotonic() - t0) * 1000
                        buf.extend(chunk)
                        yield chunk
                finally:
                    await resp.aclose()
                    usage, n = _usage_from_sse(bytes(buf))
                    store.record(ts, tenant, model, upstream,
                                 usage.get("prompt_tokens", 0) if usage else 0,
                                 usage.get("completion_tokens", n) if usage else n,
                                 usage is None, ttft, (time.monotonic() - t0) * 1000, 200)
            return StreamingResponse(gen(), media_type=resp.headers.get("content-type", "text/event-stream"))

        data = await resp.aread()
        await resp.aclose()
        e2e = (time.monotonic() - t0) * 1000
        obj = json.loads(data)
        usage = obj.get("usage")
        store.record(ts, tenant, model, upstream, usage.get("prompt_tokens", 0) if usage else 0,
                     usage.get("completion_tokens", 0) if usage else 0, usage is None, e2e, e2e, 200)
        return JSONResponse(obj)

    app.state.store = store
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="token metering proxy for an OpenAI-compatible upstream")
    p.add_argument("--upstream", required=True)
    p.add_argument("--db", default="results/tokenmeter.db")
    p.add_argument("--tenant-header", default="x-tenant")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8070)
    a = p.parse_args(argv)
    uvicorn.run(create_app(a.upstream, a.db, a.tenant_header), host=a.host, port=a.port, log_level="warning")
