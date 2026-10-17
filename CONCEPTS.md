# Concepts: the unit economics of serving tokens

`tokenmeter` is a metering proxy and a dashboard: every request through it is attributed to a tenant and a model, its prompt and completion tokens and latency are recorded, and the records are turned into cost per million tokens and a utilisation figure (MFU). This primer is about what those numbers mean, how to compute them honestly, and why a serving team that cannot produce them is flying blind.

## 1. The only price that matters is $ per million tokens

Hosted APIs are priced per million tokens, split into input (prompt) and output (completion) rates, with output typically 3–5x the input price. Self-hosting has no price list; it has a GPU bill. To compare the two, or to charge internal tenants, the GPU bill must be converted into the same unit:

```
$ per million output tokens = GPU $/hour ÷ (output tokens per hour / 1,000,000)
```

Everything interesting hides in "output tokens per hour". A T4 that costs $0.35/h and sustains 1,600 tokens/s produces 5.76 M tokens/hour: $0.06 per million. The same card serving one chatty user at 68 tokens/s produces 0.24 M tokens/hour: $1.43 per million, 24x more expensive per token for the same hardware. **Utilisation, not hardware, decides the unit price.** This is the quantitative reason behind every other project here: batching, quantization, prefix caching and autoscaling all move the denominator.

Two refinements the meter needs:

- **Prompt tokens are not free, but they are cheaper.** Prefill is compute-bound and fast per token; decode is memory-bound and slow per token. A common split is to price prompt tokens at a fraction of completion tokens (hosted APIs use 1/3 to 1/5). The meter records both counts and applies a configurable ratio, and the README shows how much the ratio changes a tenant's bill.
- **Idle time is a real cost.** A GPU that serves 1,600 tok/s for 10 minutes and sits idle for 50 produces one sixth of the tokens for the full hourly price. The meter computes cost from wall-clock GPU hours, not from busy time, because that is what the invoice does.

## 2. Attributing cost to tenants

A tenant is whoever the request is billed to: a product, a team, a customer. The proxy reads it from a header (`x-tenant`), falls back to an API key prefix or "anonymous", and stores it with every record. Per-tenant cost is then the tenant's share of tokens times the unit price, with two choices to make explicit:

- **Marginal vs fully loaded.** Marginal cost charges only the tokens (idle time is nobody's); fully loaded divides the whole GPU bill among tenants in proportion to their tokens, so idle time is shared. The second is what finance wants; the first is what a tenant can be told to optimise. The dashboard shows both.
- **Weighting prompt tokens.** With a 1:4 prompt:completion ratio, a RAG tenant that sends 4,000-token prompts and gets 50-token answers costs a tenth of what raw token counts suggest. Getting this wrong misallocates money between teams, which is how metering projects lose credibility.

## 3. MFU: how much of the GPU's arithmetic is actually used

**Model FLOPs Utilisation** compares the useful floating-point work done to the hardware's peak:

```
useful FLOPs per token ≈ 2 × parameters          (one multiply-add per weight per token, both phases)
MFU = (tokens per second × 2 × parameters) / peak FLOPS
```

For Qwen2.5-1.5B (1.54B parameters) at 1,637 tokens/s on a T4 (65 TFLOPS FP16 tensor-core peak):

```
MFU = 1,637 × 2 × 1.54e9 / 65e12 ≈ 7.8%
```

At one user (68 tokens/s): 0.3%. Those numbers look terrible and are normal: decode is memory-bound, so arithmetic units idle by design. MFU is the right lens for prefill-heavy or large-batch work and for training; for interactive decode the honest companion metric is **memory-bandwidth utilisation** (bytes of weights read per step over peak bandwidth), which the servellm results put near 68%. The dashboard shows both and labels which regime the server is in.

Caveats that belong next to any MFU figure: the "2 × parameters" rule ignores attention FLOPs (grows with context length, matters for long prompts), peak FLOPS is the dense tensor-core number that no real kernel reaches, and for a quantized model "parameters" still counts the logical weights.

## 4. What to record per request

Minimum: timestamp, tenant, model, prompt tokens, completion tokens, TTFT, end-to-end latency, status, which upstream served it. The token counts come from the upstream's `usage` field (present in the final streaming chunk when `stream_options.include_usage` is set, and in every non-streaming response); when it is missing, the proxy falls back to counting content chunks and marks the record as estimated. SQLite is enough: one row per request, indexed by tenant and time, queryable with plain SQL, no server to run.

What *not* to record: prompt or completion text. Metering does not need it, and storing it turns a cost tool into a privacy problem.

## 5. From records to a dashboard

The dashboard is one static HTML file built from the database: a per-tenant table (requests, prompt and completion tokens, share of cost under both attribution rules, p50/p95 TTFT), a cost summary (GPU hours, $/M prompt and completion tokens at the configured prices, hosted-API equivalent for comparison), and two charts (tokens per hour over time; MFU and bandwidth utilisation over time). Rebuilt by a command, not served live: the point is the numbers, and a file is reproducible.

## 6. What the experiment will show

1. Route a mixed load through the proxy to a vLLM replica on the T4: three tenants with different shapes (short chat, long-prompt RAG, bulk generation), via tokenbench with `x-tenant` headers.
2. The dashboard from that run: per-tenant cost under marginal and fully-loaded attribution, with and without prompt-token weighting; the resulting $/M tokens at the T4's hourly price; MFU and bandwidth utilisation for the run.
3. The same numbers at an assumed 30% average utilisation (the "GPUs idling at 30%" case from project 11's motivation), to show what idle time does to the unit price.

## Further reading

- Hosted API price pages (OpenAI, Anthropic, Together) for the input/output ratio convention.
- Chowdhery et al., *PaLM* (2022), appendix B, for the MFU definition.
- The servellm and quantlab results in this portfolio for the tokens/s figures the costs are built from.
