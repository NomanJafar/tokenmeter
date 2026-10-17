"""Unit economics (CONCEPTS.md sections 1-3): GPU hours -> $/M tokens, tenant attribution, MFU."""

from __future__ import annotations

M = 1_000_000


def unit_prices(gpu_usd_per_hour: float, gpu_hours: float, prompt_tokens: int, completion_tokens: int,
                prompt_weight: float = 0.25, busy_hours: float | None = None) -> dict:
    """Spread the GPU bill over weighted tokens (a prompt token counts `prompt_weight` of a completion token).

    `gpu_hours` is wall-clock time billed; `busy_hours` (default: same) is time actually spent serving.
    The fully-loaded price divides the whole bill by the tokens; the marginal price divides only the
    busy time's cost, which is what a token costs when the GPU is not idle."""
    busy = gpu_hours if busy_hours is None else busy_hours
    total = gpu_usd_per_hour * gpu_hours
    weighted = prompt_tokens * prompt_weight + completion_tokens
    loaded = total / weighted if weighted else None
    marginal = gpu_usd_per_hour * busy / weighted if weighted else None
    return {
        "gpu_usd_per_hour": gpu_usd_per_hour, "gpu_hours": gpu_hours, "busy_hours": busy, "total_usd": total,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "prompt_weight": prompt_weight,
        "weighted_tokens": weighted,
        "usd_per_m_completion": loaded * M if loaded is not None else None,
        "usd_per_m_prompt": loaded * M * prompt_weight if loaded is not None else None,
        "marginal_usd_per_m_completion": marginal * M if marginal is not None else None,
        "marginal_usd_per_m_prompt": marginal * M * prompt_weight if marginal is not None else None,
    }


def attribute(tenants: dict[str, tuple[int, int]], prices: dict, mode: str = "loaded") -> dict[str, dict]:
    """tenants: name -> (prompt_tokens, completion_tokens).
    marginal: tokens x unit price (idle time belongs to nobody);
    loaded: the whole bill split in proportion to weighted tokens (idle time shared)."""
    w = prices["prompt_weight"]
    weighted = {t: p * w + c for t, (p, c) in tenants.items()}
    total_w = sum(weighted.values())
    out = {}
    for t, (p, c) in tenants.items():
        share = weighted[t] / total_w if total_w else 0.0
        if mode == "marginal":
            usd = (weighted[t] / M) * prices["marginal_usd_per_m_completion"] if prices["marginal_usd_per_m_completion"] else 0.0
        elif mode == "loaded":
            usd = share * prices["total_usd"]
        else:
            raise ValueError(mode)
        out[t] = {"prompt_tokens": p, "completion_tokens": c, "weighted_tokens": weighted[t], "share": share, "usd": usd}
    return out


def mfu(tokens_per_s: float, params: float, peak_tflops: float) -> float:
    return tokens_per_s * 2 * params / (peak_tflops * 1e12)


def bandwidth_utilisation(weight_bytes: float, step_s: float, peak_gb_s: float) -> float:
    """Fraction of peak memory bandwidth used if the weights are read once per decode step."""
    return weight_bytes / step_s / (peak_gb_s * 1e9)


def hosted_equivalent(prompt_tokens: int, completion_tokens: int, usd_per_m_in: float, usd_per_m_out: float) -> float:
    return prompt_tokens / M * usd_per_m_in + completion_tokens / M * usd_per_m_out
