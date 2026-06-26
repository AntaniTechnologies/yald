YALD (Yet Another Llama Dashboard) v1.1.1 is a real-time terminal UI for monitoring llama-server (https://github.com/ggerganov/llama.cpp) instances.

Copyright 2026 Antani Technologies BV — MIT License

## Core function

Spins up a background thread that polls a llama-server every 500 ms, fetching data from `/health`, `/slots`, `/metrics`, and `/props` endpoints, then renders the results in a flicker-free terminal dashboard using the Rich (https://github.com/Textualize/rich) library at 10 FPS.

## Dashboard panels

- **Header** — connection status (ONLINE/OFFLINE), server address (stripped of scheme), and timestamp
- **Metrics (left)** — state badge (IDLE / PREFILL / ACTIVE), context usage bar with colour-coded thresholds (green ≤ 69.9 %, orange 70–85 %, red > 85 %, blinks > 95 %), token counts (max tokens, processed, generated, prompt total, predicted total, decode calls), concurrency stats (processing, deferred), reasoning format, and model name
- **Performance (right)** — prompt speed and generation speed displayed side-by-side as `X.X tok/s`
- **Slots (right column)** — 2×2 grid with per-slot quadrants (Slot 0–3), each showing state badge, generation progress `(n/n_predict)`, prompt progress `(processed/total)`, KV cache bar with percentage, and optional remaining-token counter
- **Footer** — activity log of the last 5 state transitions

## Key technical details

- State detection uses `is_processing` (bool, top-level `/slots` field) combined with `n_decoded` nested at `slot["next_token"][0]["n_decoded"]`:
  - `is_processing=False` → IDLE
  - `is_processing=True, n_decoded=0` → PREFILL (evaluating prompt)
  - `is_processing=True, n_decoded>0` → INFERENCE/ACTIVE (generating tokens)
  - Note: `slot["state"]` integer does NOT exist in the `/slots` JSON schema
- Calculates token speeds from cumulative counter deltas (`prompt_tokens_total`, `tokens_predicted_total`) with 10-sample moving-average smoothing
- Clears stale speed samples when the server goes idle to prevent old values from lingering in the graph
- Anomaly filter: prefill speeds > 499 tok/s and inference speeds > 99 tok/s are replaced with the previous value
- Context usage bar uses `slot_capacity` (n_ctx, the true KV-cache capacity) rather than `n_tokens_max` (which is the max-generation budget and a semantic mismatch for context sizing)
- Context safeguard: when `n_prompt` drops to 0 (slot just assigned, task not yet started), YALD falls back through three layers — last-seen effective `n_prompt` → historical maximum → `n_ctx` — saving the *effective* (post-fallback) value so stale zeros are never persisted
- KV cache occupancy per slot: `(n_cache + n_processed + n_decoded) / n_ctx`
- Prompt progress per slot: `(n_cache + n_processed) / n_prompt` (includes KV-cache prefix reuse, matching `server-context.cpp` logic)
- High-water marks for context capacity, max token counts, and reasoning format across idle periods
- Parses both Prometheus exposition format and llama.cpp internal metric naming conventions (`:` vs `_` separators)
- Model name via `/props` endpoint (prefers `model_alias`, falls back to `model_path` basename), with Prometheus `llama_model_name` as fallback
- Prompt speed via delta method takes precedence; Prometheus `prompt_tokens_seconds` / `predicted_tokens_seconds` metrics act as fallback when delta is zero
- Accepts `--server` / `-s` flag to specify the llama-server URL (default http://127.0.0.1:8080); auto-prepends `http://` if scheme is omitted
- Optional `--debug` flag writes raw `/slots` and `/metrics` responses as JSONL to the specified file
- Runs with Ctrl+C or SIGTERM graceful shutdown
- Runs with `screen=True` for full-screen terminal UI
