import pytest

from tokenmeter.costs import (
    attribute,
    bandwidth_utilisation,
    hosted_equivalent,
    mfu,
    unit_prices,
)


def test_unit_prices_split_gpu_bill_by_weighted_tokens():
    # 1 GPU-hour at $0.35; 1M completion tokens and 2M prompt tokens weighted 1/4 -> 1.5M weighted tokens
    p = unit_prices(gpu_usd_per_hour=0.35, gpu_hours=1.0, prompt_tokens=2_000_000, completion_tokens=1_000_000, prompt_weight=0.25)
    assert p["usd_per_m_completion"] == pytest.approx(0.35 / 1.5)
    assert p["usd_per_m_prompt"] == pytest.approx(0.35 / 1.5 * 0.25)
    assert p["total_usd"] == pytest.approx(0.35)
    assert p["weighted_tokens"] == 1_500_000


def test_unit_prices_with_no_tokens_is_all_idle():
    p = unit_prices(0.35, 2.0, 0, 0, 0.25)
    assert p["total_usd"] == 0.7 and p["usd_per_m_completion"] is None


def test_attribute_marginal_vs_fully_loaded():
    tenants = {"chat": (100_000, 400_000), "rag": (1_600_000, 100_000)}   # (prompt, completion)
    prices = unit_prices(0.35, 1.0, 1_700_000, 500_000, prompt_weight=0.25)
    m = attribute(tenants, prices, mode="marginal")
    f = attribute(tenants, prices, mode="loaded")
    # marginal: each tenant pays for its weighted tokens at the unit price; sums to the whole bill when nothing is idle
    assert sum(t["usd"] for t in m.values()) == pytest.approx(0.35)
    assert m["chat"]["weighted_tokens"] == 100_000 * 0.25 + 400_000
    # with idle time, marginal undercharges and loaded recovers the full bill
    idle = unit_prices(0.35, 4.0, 1_700_000, 500_000, prompt_weight=0.25, busy_hours=1.0)   # billed 4 h, served 1 h
    assert sum(t["usd"] for t in attribute(tenants, idle, "marginal").values()) == pytest.approx(0.35)
    assert sum(t["usd"] for t in attribute(tenants, idle, "loaded").values()) == pytest.approx(1.40)
    assert f["rag"]["share"] == pytest.approx((1_600_000 * 0.25 + 100_000) / (1_700_000 * 0.25 + 500_000))


def test_prompt_weight_moves_money_between_tenants():
    tenants = {"chat": (100_000, 400_000), "rag": (1_600_000, 100_000)}
    p1 = attribute(tenants, unit_prices(0.35, 1.0, 1_700_000, 500_000, 1.0), "loaded")
    p4 = attribute(tenants, unit_prices(0.35, 1.0, 1_700_000, 500_000, 0.25), "loaded")
    assert p1["rag"]["share"] > 0.7 and p4["rag"]["share"] < 0.6


def test_mfu_and_bandwidth_utilisation_match_hand_calculation():
    assert mfu(tokens_per_s=1637, params=1.54e9, peak_tflops=65) == pytest.approx(1637 * 2 * 1.54e9 / 65e12, rel=1e-6)
    # 2.98 GiB of weights read once per 14.7 ms step against 320 GB/s
    assert bandwidth_utilisation(weight_bytes=2.98 * 2**30, step_s=0.0147, peak_gb_s=320) == pytest.approx(2.98 * 2**30 / 0.0147 / 320e9, rel=1e-6)


def test_hosted_equivalent_prices_tokens_at_api_rates():
    assert hosted_equivalent(prompt_tokens=1_000_000, completion_tokens=500_000, usd_per_m_in=0.15, usd_per_m_out=0.60) == pytest.approx(0.15 + 0.30)
