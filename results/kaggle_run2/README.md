Run 2 (2026-10-17), Kaggle T4, vLLM 0.31.0, Qwen2.5-1.5B-Instruct fp16.

- `kernel.log`, `startup.log`: Kaggle kernel output and vLLM startup log (weights 2.98 GiB, KV cache 340,064 tokens, TRITON_ATTN backend).
- `load_{chat,rag,bulk}.{json,csv,md,log}`: the three tokenbench open-loop sweeps, 480 s each, run in parallel through the proxy.
- `proxy.log`: metering proxy log (empty: no errors, no estimated usage).
- `tokenmeter.db`: every request record the proxy wrote; `tokenmeter report` reads this.
- `dashboard/`: `report.json`, `dashboard.html` and `tokens_over_time.png` built from the database.
- `run_meta.json`: wall hours and the startup numbers passed to the report.
