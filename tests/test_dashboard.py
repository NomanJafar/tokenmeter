from tokenmeter.dashboard import build_report
from tokenmeter.store import Store


def _seed(db):
    s = Store(db)
    for i in range(20):
        s.record(ts=1000 + i * 5, tenant="chat", model="m", upstream="u", prompt_tokens=50, completion_tokens=200, usage_estimated=False, ttft_ms=40 + i, e2e_ms=3000, status=200)
    for i in range(5):
        s.record(ts=1010 + i * 20, tenant="rag", model="m", upstream="u", prompt_tokens=4000, completion_tokens=60, usage_estimated=False, ttft_ms=400, e2e_ms=1500, status=200)
    return s


def test_build_report_writes_html_and_json_with_both_attributions(tmp_path):
    _seed(tmp_path / "m.db")
    out = build_report(tmp_path / "m.db", tmp_path / "out", gpu_usd_per_hour=0.35, params=1.54e9, peak_tflops=65,
                       weight_bytes=2.98 * 2**30, peak_gb_s=320, prompt_weight=0.25, hosted_in=0.15, hosted_out=0.60)
    assert (tmp_path / "out" / "dashboard.html").exists() and (tmp_path / "out" / "report.json").exists()
    assert out["totals"]["completion_tokens"] == 4000 + 300 and out["prices"]["gpu_hours"] > 0
    assert set(out["attribution"]["loaded"]) == {"chat", "rag"} and set(out["attribution"]["marginal"]) == {"chat", "rag"}
    assert abs(sum(t["usd"] for t in out["attribution"]["loaded"].values()) - out["prices"]["total_usd"]) < 1e-9
    assert 0 < out["utilisation"]["mfu"] < 1 and 0 < out["utilisation"]["bandwidth"] < 5
    html = (tmp_path / "out" / "dashboard.html").read_text()
    assert "chat" in html and "rag" in html and "$/M" in html and "MFU" in html
    assert (tmp_path / "out" / "tokens_over_time.png").exists()


def test_cli_report_builds_dashboard(tmp_path, capsys):
    from tokenmeter.cli import main
    _seed(tmp_path / "m.db")
    main(["report", "--db", str(tmp_path / "m.db"), "--out", str(tmp_path / "o"), "--gpu-usd-per-hour", "0.35", "--params", "1.54e9", "--weight-gib", "2.98"])
    assert (tmp_path / "o" / "dashboard.html").exists() and "$/M completion" in capsys.readouterr().out
