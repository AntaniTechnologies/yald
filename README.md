# YALD - Yet Another Llama Dashboard

A real-time terminal UI for monitoring llama-server instances.
One file. One command. One screen. Zero fuss.

## Prerequisites

- Python 3.10+
- A running llama-server instance with `--metrics --slot` flags (e.g., from llama.cpp)

## Installation

```bash
pip install rich aiohttp
python yald.py
```

## Usage

```bash
python yald.py                       # connect to localhost:8080
python yald.py --server 192.168.1.10 # connect to a remote server
python yald.py --server 192.168.1.10 --debug dump.jsonl  # log raw server responses
```

## What it shows

The dashboard is split into a few panels:

- **Header** -- connection status, current time
- **Metrics** -- state (IDLE / PREFILL / INFERENCE), context usage bar, token counts, concurrency info
- **Performance** -- prompt speed and generation speed side by side
- **Slots** -- 2x2 grid of per-slot state with generation, prompt, and KV cache progress
- **Footer** -- last few state transitions in an activity log

## Dependencies

- [rich](https://github.com/Textualize/rich) -- terminal rendering
- [aiohttp](https://docs.aiohttp.org/) -- async HTTP client

## License

MIT License. See [LICENSE](LICENSE) for the full text.

YALD is copyright (C) 2026 Antani Technologies BV.
