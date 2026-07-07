# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.5.4] - 2026-07-07

### Fixed

- **Slot quadrant state label:** Slot in prefill state (`is_processing=True`, `n_decoded=0`) now correctly displays "PREFILL" with magenta styling instead of incorrectly showing "IDLE" with blue styling, matching the Metrics panel's state detection logic
- **Empty slot fallback:** Removed misleading single-slot `SlotData` creation when no active slots exist — `snapshot.slots` now correctly remains empty, preventing the slots panel from rendering a phantom slot cell
- **Performance panel cleanup:** Removed unused local variables (`total_eval_tokens`, `total_decoded_tokens`) that were computed but never displayed; the panel now cleanly uses the collector's aggregated smoothed speed values

## [1.5.3] - 2026-07-07

### Changed

- **Activity log state label:** Renamed "INFERENCE" to "ACTIVE" for consistency
  with the metrics panel state badge — both now use the same label for the
  `is_processing && n_decoded > 0` state

## [1.5.2] - 2026-07-06

### Fixed

- **KV cache watermark display:** High watermark is now applied whenever a slot is idle
  (`is_processing=False`) instead of only when the computed KV token count is exactly zero,
  fixing a corner case where the KV cache progress bar showed stale values after a slot
  transitioned from active to idle

## [1.5.1] - 2026-07-06

### Fixed

- **KV cache high watermark:** Per-slot KV cache peak is now preserved when slots go idle, so the KV cache progress bar no longer drops to 0 between generations
- **Tokens Max display:** "Tokens Max" now shows the correct value instead of always displaying 0 — the effective `n_prompt` (after safeguard fallbacks) is used for the slot data
- **Model name layout:** Model name is now printed on a separate line to avoid truncation in narrow terminals

## [1.5.0] - 2026-07-06

### Changed

- **Major rewrite:** Replaced `threading.Thread` + `threading.Lock` polling with
  `asyncio` — the event loop is single-threaded, eliminating all lock contention
  and race conditions (`_history.append`, `_state_log` TOCTOU, etc.) entirely
- **Concurrent endpoint fetching:** `/health`, `/slots`, `/metrics`, `/props` are
  now fetched in parallel via `asyncio.gather()` instead of sequentially, cutting
  worst-case poll latency from ~8s to ~2s
- **Connection pooling:** Single `aiohttp.ClientSession` reused across polls
- **Dependency:** Removed `requests`; `aiohttp` (already listed) is now the sole
  HTTP client
- `MetricsCollector.start()` and `stop()` are now `async`
- `YALDApplication.run()` is now `async`; `main()` uses `asyncio.run()`

## [1.4.0] - 2026-06-29

### Changed

- Renamed `yald_v2.py` → `yald.py`
- Slots panel: only paint slots that are actually reported by `llama-server /slots` — any slot index beyond the server-reported count is skipped entirely instead of drawing a placeholder

## [1.3.0] - 2026-06-27

### Changed

- Speed calculation: only add samples to moving average if within threshold
  (<= 499 tok/s prefill, <= 99 tok/s inference) instead of post-hoc freezing
  to the previous frame's value
- Debug file handle: open lazily in `run()` instead of `__init__`, preventing
  handle leaks when the app is instantiated but never executed

## [1.1.2] - 2026-06-26

### Fixed

- Context Usage progress bar no longer shows non-zero values on startup before the first request; added a guard so the high-water fallback only activates when `_last_slot_capacity > 0` (i.e. after the slot has been activated by an actual request)

## [1.1.1] - 2026-06-25

### Fixed

- Context Usage progress bar no longer jumps to unrelated values when llama-server goes idle; safeguard now saves the effective `n_prompt` value (after fallbacks) instead of the raw pre-fallback slot value, preventing stale 0 from being persisted as the "last known good"

## [1.1] - 2026-06-25

### Added

- Model name display via `/props` endpoint with Prometheus `/metrics` fallback
- Server address shown in the header bar

### Changed

- Align endpoint constant spacing in `MetricsCollector`

## [1.0.0] - 2026-06-25

### Added

- Real-time terminal UI for monitoring llama-server instances
- Metrics panel: state detection (IDLE, PREFILL, INFERENCE), context usage bar, token counts, concurrency info
- Performance panel: prompt speed and generation speed display
- Slots panel: 2x2 grid with per-slot generation, prompt, and KV cache progress bars
- Activity log: state transition history in the footer
- Prometheus `/metrics` endpoint parsing
- Debug logging mode with JSONL output
- Remote server support via `--server` flag
- Thread-safe metrics collection with background polling
- Moving-average smoothing for speed metrics
- High-water mark fallbacks to prevent UI values from blinking to 0
- Graceful shutdown on SIGINT / SIGTERM
- MIT License
