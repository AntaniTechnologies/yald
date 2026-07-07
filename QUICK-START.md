# YALD v2 — Quick Start

## TL;DR

```bash
# 1. Start llama-server with the three flags YALD v2 depends on:
llama-server \
  --port 8080 \
  --metrics \
  -m models/your-model.gguf

# 2. Run YALD v2, pointing it at the server and the log file:
python yald_v2.py -s http://127.0.0.1:8080 -l /tmp/llama-run.log
```

---

## Why these flags?

YALD v2 combines **two data sources** that both need to be enabled explicitly in `llama-server`:

| Data source | Required flag | What YALD v2 gets |
|---|---|---|
| **API metrics** (token counts, KV cache, speed deltas) | `--metrics` | `prefill_speed`, `inference_speed`, `prompt_tokens_total`, `n_decode_total`, concurrency |
| **Log stream** (real-time parse of stdout/stderr) | `--log-level` or default verbosity | MTP acceptance stats, VRAM buffer breakdown, instantaneous prompt/gen speed, model name & quant |

Without `--metrics`, YALD v2 falls back to slot-level estimates and loses historical smoothing.
Without the log stream (see below), MTP diagnostics and memory breakdown panels are empty.

---

## The log file

`yald_v2.py` reads from a **log file** written by `monitor.sh` (or any process that mirrors llama-server output to disk). The default log file path is `/tmp/llama-run.log`, but you can override it:

```bash
# In monitor.sh, LOG_FILE controls where logs go:
export LOG_FILE=/tmp/llama-run.log   # default
python yald_v2.py -l "$LOG_FILE"
```

If you skip `monitor.sh` and run `llama-server` directly, you can still feed YALD v2 the log by redirecting output:

```bash
# Direct capture (no monitor.sh)
llama-server --port 8080 --metrics -m models/your-model.gguf \
  2>&1 | tee /tmp/llama-run.log
```

Then in another terminal:

```bash
python yald_v2.py -s http://127.0.0.1:8080 -l /tmp/llama-run.log
```

> **Note:** When using `tee`, the log file stays open and grows indefinitely. For production use, prefer `monitor.sh` which manages file rotation and parsing efficiently.

---

## Full llama-server command

```bash
llama-server \
  --port 8080 \
  --metrics \
  -m models/llama-3.1-8b-instruct-Q8_0.gguf \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  -t 8
```

| Flag | Required for YALD v2? | Purpose |
|---|---|---|
| `--port` | Yes (connect to it) | API server port |
| `--metrics` | **Yes** | Exposes Prometheus `/metrics` endpoint for speed deltas and token counters |
| `-m` | Yes (obviously) | Model file |
| `--ctx-size` | No (but recommended) | Sets slot capacity; affects KV cache sizing |
| `--n-gpu-layers` | No | GPU offload |
| `-t` | No | CPU threads |

---

## YALD v2 CLI options

```
usage: yald_v2.py [-h] [--server SERVER] [--log-file LOG_FILE] [--debug DEBUG]

YALD v2 — Yet Another Llama Dashboard (with monitor.sh integration).

options:
  --server, -s    llama-server URL (default: http://127.0.0.1:8080)
  --log-file, -l  Path to monitor.sh log file (enables MTP, memory, log-sourced metrics)
  --debug         Path to a JSONL debug log file
```

Example with all options:

```bash
python yald_v2.py \
  --server http://127.0.0.1:8080 \
  --log-file /tmp/llama-run.log \
  --debug /tmp/yald-debug.jsonl
```

---

## Verifying it works

After starting both, you should see:

1. **Header** — `YALD v2 ● ONLINE` with the server URL
2. **Metrics** — State, context bar, token counts, concurrency
3. **MTP Diagnostics** — Acceptance rate, gen/acc tokens, mean acceptance length
4. **Memory Breakdown** — KV / Compute / Scratch buffer sizes in GiB
5. **Performance** — Prompt & generation speed with source indicators (`log` or `api`)
6. **Slots** — Up to 4 slot quadrants with KV cache bars
7. **Activity Log** — State transitions at the bottom

If MTP or Memory panels show `-` / `0`, verify that:
- `monitor.sh` is running and writing to the log file
- The log file path matches between `monitor.sh` (`LOG_FILE`) and `yald_v2.py` (`--log-file`)
