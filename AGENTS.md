YALD (Yet Another Llama Dashboard) is a real-time terminal UI for monitoring llama.cpp (https://github.com/ggerganov/llama.cpp) server instances. Here's what it does:
Core function: Spins up a background thread that polls a llama-server every 100 ms, fetching data from /health, /slots, and /metrics endpoints, then renders the results in a flicker-free terminal dashboard using the Rich (https://github.com/Textualize/rich) library at 10 FPS.
Dashboard panels:
- Header — connection status (ONLINE/OFFLINE) and timestamp
- Metrics (left) — current state (IDLE/PREFILL/INFERENCE), context usage bar, prompt progress bar, token counts (prompt, processed, evaluated, predicted, decoded), queue/concurrency stats, KV cache usage, and reasoning format
- Performance (right) — prefill speed (tok/s) and generation speed (tok/s) with sparkline bar graphs showing ~6 seconds of speed history
- Footer — activity log of recent state transitions
Key technical details:
- Uses slot["state"] integer (0=Idle, 1=Prefill, 2=Decode) for reliable state detection with fallback heuristics for older llama.cpp builds
- Calculates token speeds from cumulative counter deltas with moving-average smoothing (10-sample window)
- Preserves high-water marks for context capacity, max token counts, and reasoning format across idle periods
- Parses both Prometheus exposition format and llama.cpp internal metric naming conventions (: vs _ separators)
- Runs with Ctrl+C or SIGTERM graceful shutdown
- Accepts --server / -s flag to specify the llama-server URL (default http://127.0.0.1:8080)