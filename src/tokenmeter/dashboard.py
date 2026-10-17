"""Build a static dashboard (HTML + PNG charts + report.json) from the metering database."""

from __future__ import annotations

import json
from pathlib import Path

from tokenmeter.costs import attribute, bandwidth_utilisation, hosted_equivalent, mfu, unit_prices
from tokenmeter.store import Store


def build_report(db_path, out_dir, gpu_usd_per_hour: float, params: float, peak_tflops: float, weight_bytes: float,
                 peak_gb_s: float, prompt_weight: float = 0.25, hosted_in: float = 0.15, hosted_out: float = 0.60,
                 gpu_hours: float | None = None, step_s: float | None = None, bucket_s: float = 60) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(db_path)
    tot = store.totals()
    per = store.per_tenant()
    buckets = store.buckets(bucket_s)
    span_s = max(1.0, (tot["last_ts"] or 0) - (tot["first_ts"] or 0)) if tot["requests"] else 0.0
    hours = gpu_hours if gpu_hours is not None else span_s / 3600
    prices = unit_prices(gpu_usd_per_hour, hours, tot["prompt_tokens"], tot["completion_tokens"], prompt_weight, busy_hours=span_s / 3600)
    tenants = {t: (a["prompt_tokens"], a["completion_tokens"]) for t, a in per.items()}
    attribution = {m: attribute(tenants, prices, m) for m in ("loaded", "marginal")}
    tps = tot["completion_tokens"] / span_s if span_s else 0.0
    step = step_s if step_s is not None else (1 / (tps / max(1, _mean_concurrency(buckets, bucket_s))) if tps else None)
    util = {"completion_tokens_per_s": tps, "mfu": mfu(tps, params, peak_tflops),
            "bandwidth": bandwidth_utilisation(weight_bytes, step, peak_gb_s) if step else None, "step_s_assumed": step}
    hosted = hosted_equivalent(tot["prompt_tokens"], tot["completion_tokens"], hosted_in, hosted_out)
    report = {"totals": tot, "span_s": span_s, "prices": prices, "per_tenant": per, "attribution": attribution,
              "utilisation": util, "hosted_equivalent_usd": hosted, "hosted_rates": {"in": hosted_in, "out": hosted_out},
              "idle_scenario": _idle_scenario(gpu_usd_per_hour, tot, prompt_weight)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    _charts(buckets, bucket_s, out_dir)
    (out_dir / "dashboard.html").write_text(_html(report))
    return report


def _mean_concurrency(buckets, bucket_s):
    # requests x mean e2e / bucket length approximates in-flight requests (Little's law)
    vals = [b["requests"] * (b["e2e_ms_mean"] or 0) / 1000 / bucket_s for b in buckets if b["requests"]]
    return sum(vals) / len(vals) if vals else 1.0


def _idle_scenario(rate, tot, w):
    """What the unit price becomes if this traffic were the only use of a GPU billed at 30% utilisation."""
    weighted = tot["prompt_tokens"] * w + tot["completion_tokens"]
    if not weighted:
        return None
    full = unit_prices(rate, 1.0, tot["prompt_tokens"], tot["completion_tokens"], w)
    return {"at_30pct_utilisation_usd_per_m_completion": full["usd_per_m_completion"] / 0.30 if full["usd_per_m_completion"] else None}


def _charts(buckets, bucket_s, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6))
    if buckets:
        t0 = buckets[0]["t"]
        xs = [(b["t"] - t0) / 60 for b in buckets]
        ax.bar(xs, [b["completion_tokens"] / bucket_s for b in buckets], width=bucket_s / 60 * 0.9, label="completion tokens/s")
        ax.bar(xs, [b["prompt_tokens"] / bucket_s for b in buckets], width=bucket_s / 60 * 0.9,
               bottom=[b["completion_tokens"] / bucket_s for b in buckets], alpha=0.5, label="prompt tokens/s")
    ax.set_xlabel("minutes"); ax.set_ylabel("tokens / s"); ax.set_title("Tokens over time"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "tokens_over_time.png", dpi=120); plt.close(fig)


def _html(r: dict) -> str:
    p, u = r["prices"], r["utilisation"]
    f = lambda x, d=2: "–" if x is None else f"{x:,.{d}f}"
    rows = []
    for t, a in r["per_tenant"].items():
        lo, ma = r["attribution"]["loaded"][t], r["attribution"]["marginal"][t]
        rows.append(f"<tr><td>{t}</td><td>{a['requests']}</td><td>{a['errors']}</td><td>{a['prompt_tokens']:,}</td><td>{a['completion_tokens']:,}</td>"
                    f"<td>{f(lo['share']*100,1)}%</td><td>${f(lo['usd'],4)}</td><td>${f(ma['usd'],4)}</td><td>{f(a['ttft_p50_ms'],0)}</td><td>{f(a['ttft_p95_ms'],0)}</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>tokenmeter</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:.3rem .6rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}.k{{display:inline-block;margin:.4rem 1.2rem .4rem 0}}.k b{{font-size:1.4rem}}</style></head><body>
<h1>tokenmeter</h1>
<p>{r['totals']['requests']:,} requests over {r['span_s']/60:.1f} min · {r['totals']['prompt_tokens']:,} prompt + {r['totals']['completion_tokens']:,} completion tokens · {r['totals']['errors']} errors</p>
<h2>Unit price</h2>
<div class="k">GPU bill<br><b>${f(p['total_usd'],4)}</b><br>{f(p['gpu_hours'],3)} h × ${p['gpu_usd_per_hour']}/h</div>
<div class="k">$/M completion (fully loaded)<br><b>${f(p['usd_per_m_completion'])}</b></div>
<div class="k">$/M prompt (weight {p['prompt_weight']})<br><b>${f(p['usd_per_m_prompt'])}</b></div>
<div class="k">$/M completion (marginal, busy time only)<br><b>${f(p['marginal_usd_per_m_completion'])}</b></div>
<div class="k">Hosted-API equivalent (${r['hosted_rates']['in']}/${r['hosted_rates']['out']} per M in/out)<br><b>${f(r['hosted_equivalent_usd'],4)}</b></div>
<div class="k">If the GPU ran at 30% utilisation<br><b>${f((r['idle_scenario'] or {{}}).get('at_30pct_utilisation_usd_per_m_completion'))}/M</b></div>
<h2>Utilisation</h2>
<div class="k">completion tokens/s<br><b>{f(u['completion_tokens_per_s'],1)}</b></div>
<div class="k">MFU<br><b>{f((u['mfu'] or 0)*100,2)}%</b></div>
<div class="k">memory-bandwidth utilisation<br><b>{f((u['bandwidth'] or 0)*100,1)}%</b><br>assumed step {f((u['step_s_assumed'] or 0)*1000,1)} ms</div>
<h2>Per tenant</h2>
<table><tr><th>tenant</th><th>requests</th><th>errors</th><th>prompt tokens</th><th>completion tokens</th><th>share (loaded)</th><th>$ loaded</th><th>$ marginal</th><th>TTFT p50 ms</th><th>p95 ms</th></tr>{''.join(rows)}</table>
<h2>Tokens over time</h2><img src="tokens_over_time.png" alt="tokens over time">
<p><small>Costs use wall-clock GPU hours (idle time is billed); "marginal" uses busy time only. MFU = tokens/s × 2 × params ÷ peak FLOPS; bandwidth utilisation assumes the weights are read once per decode step.</small></p>
</body></html>"""
