"""`tokenmeter proxy ...` and `tokenmeter report ...`."""

from __future__ import annotations

import argparse
import sys

from tokenmeter.dashboard import build_report
from tokenmeter.proxy import main as proxy_main


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="tokenmeter", description="per-tenant token accounting: metering proxy and cost dashboard")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("proxy", help="run the metering proxy (see `tokenmeter proxy --help`)", add_help=False)
    r = sub.add_parser("report", help="build the dashboard from the database")
    r.add_argument("--db", default="results/tokenmeter.db")
    r.add_argument("--out", default="results/dashboard")
    r.add_argument("--gpu-usd-per-hour", type=float, required=True, help="what the GPU costs per wall-clock hour")
    r.add_argument("--params", type=float, required=True, help="model parameter count, e.g. 1.54e9")
    r.add_argument("--peak-tflops", type=float, default=65.0, help="dense FP16 tensor-core peak (T4: 65)")
    r.add_argument("--weight-gib", type=float, required=True, help="weights in GiB as loaded (from vLLM's startup log)")
    r.add_argument("--peak-gb-s", type=float, default=320.0, help="memory bandwidth (T4: 320)")
    r.add_argument("--prompt-weight", type=float, default=0.25)
    r.add_argument("--hosted-in", type=float, default=0.15, help="hosted API $/M input tokens for comparison")
    r.add_argument("--hosted-out", type=float, default=0.60, help="hosted API $/M output tokens for comparison")
    r.add_argument("--gpu-hours", type=float, default=None, help="billed GPU hours (default: span of the records)")
    r.add_argument("--step-ms", type=float, default=None, help="decode step time for bandwidth utilisation (default: estimated)")
    if argv and argv[0] == "proxy":
        proxy_main(argv[1:])
        return
    a = p.parse_args(argv)
    rep = build_report(a.db, a.out, a.gpu_usd_per_hour, a.params, a.peak_tflops, a.weight_gib * 2**30, a.peak_gb_s,
                       a.prompt_weight, a.hosted_in, a.hosted_out, a.gpu_hours, a.step_ms / 1000 if a.step_ms else None)
    pr, u = rep["prices"], rep["utilisation"]
    print(f"{rep['totals']['requests']} requests, {rep['totals']['completion_tokens']:,} completion tokens over {rep['span_s']/60:.1f} min")
    print(f"$/M completion: {pr['usd_per_m_completion']:.3f} loaded, {pr['marginal_usd_per_m_completion']:.3f} marginal | MFU {u['mfu']*100:.2f}% | wrote {a.out}/dashboard.html")
