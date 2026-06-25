# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
