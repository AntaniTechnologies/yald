"""
YALD - Yet Another Llama Dashboard

A real-time terminal UI for monitoring llama-server instances.

Copyright 2026 Antani Technologies BV

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MetricSnapshot:
    """Immutable snapshot of all metrics at a point in time."""

    timestamp: float
    # Slot / context
    context_tokens: int = 0        # n_prompt_tokens_processed + n_decoded
    slot_capacity: int = 1         # n_ctx for the active slot
    n_prompt_tokens: int = 0       # full prompt length (including KV-cached prefix)
    n_prompt_tokens_processed: int = 0  # tokens actually evaluated this turn
    n_prompt_tokens_cache: int = 0      # prefix tokens reused from KV cache
    n_decoded: int = 0                  # tokens generated so far this turn
    prompt_progress: float = 0.0        # 0.0 – 1.0
    # State (derived from is_processing + n_decoded)
    is_prefill: bool = False
    is_inference: bool = False
    # Speed
    prefill_speed: float = 0.0     # tok/s during prefill
    inference_speed: float = 0.0   # tok/s during generation
    # Prometheus cumulative counters (used for delta-based speed)
    prompt_tokens_total: int = 0
    tokens_predicted_total: int = 0
    n_tokens_max: int = 0
    n_tokens_predicted: int = 0        # params["n_predict"] — max generation budget for this slot
    # Concurrency
    requests_processing: int = 0
    requests_deferred: int = 0
    # KV cache: derived from /slots fields (cache + processed + decoded) / n_ctx
    kv_cache_usage: float = 0.0    # 0.0 – 1.0
    kv_cache_tokens: int = 0       # total tokens occupying KV cache
    # Totals
    n_decode_total: int = 0
   # Reasoning (model-specific)
    reasoning_format: str = ""
    reasoning_in_content: bool = False
    # Model identification from /metrics
    model_name: str = ""


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Thread-safe background collector for llama-server metrics."""

    HEALTH_ENDPOINT   = "/health"
    SLOTS_ENDPOINT    = "/slots"
    METRICS_ENDPOINT  = "/metrics"
    PROPS_ENDPOINT    = "/props"

    def __init__(self, server_url: str = "http://127.0.0.1:8080",
                 poll_interval: float = 0.5):
        self.server_url    = server_url
        self.poll_interval = poll_interval

        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._current: MetricSnapshot = MetricSnapshot(timestamp=time.time())
        self._history: list[MetricSnapshot] = []
        self._history_max = 60          # ~30 s of history at 500 ms poll

        self._connected      = False
        self._last_error: Optional[str] = None
        self._consecutive_failures: int = 0
        self._success_after_offline: int = 0

        self._state_log: list[tuple[float, str]] = []
        self._log_max   = 10
        self._last_state: str = ""      # FIX 9: empty so first IDLE is recorded

        # Raw /slots list for per-slot UI (thread-safe)
        self._raw_slots: list[dict] = []

        # Persistent "high-water" values so the UI doesn't reset to 0 mid-session
        self._max_prompt_tokens: int = 0
        self._max_processed: int     = 0
        self._max_context: int       = 0
        self._last_slot_capacity: int = 0

        # Last-seen runtime values (not maximums) — used as safeguard when a
        # value drops to 0 mid-request (e.g. prompt consumed but slot not yet
        # cleared by llama.cpp).  The high-water marks above are secondary
        # fallbacks when no previous value has been recorded yet.
        self._prev_context_tokens: int = 0
        self._prev_prompt_tokens: int  = 0
        self._last_reasoning_format: str  = ""
        self._last_reasoning_in_content: bool = False
        self._last_prompt_progress: float = 0.0

        # Moving averages to smooth out 500 ms batch-sampling spikes
        self._prefill_avg: deque[float]   = deque(maxlen=10)
        self._inference_avg: deque[float] = deque(maxlen=10)

        # Raw /metrics text for debug logging (thread-safe)
        self._raw_metrics_text: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_slot_data(self, slot: dict, snapshot: MetricSnapshot) -> None:
        """Pull fields from a /slots JSON slot object into *snapshot*.

        Field layout (from llama.cpp tools/server/server-context.cpp):
          slot["n_ctx"]                      int   KV-cache capacity
          slot["is_processing"]              bool
          slot["n_prompt_tokens"]            int   full prompt length
          slot["n_prompt_tokens_processed"]  int   tokens evaluated this turn
          slot["n_prompt_tokens_cache"]      int   prefix reused from prior KV cache
          slot["next_token"]                 list  [{..., "n_decoded": int, ...}]
            next_token[0]["n_decoded"]        int   tokens generated so far this turn
          slot["params"]["n_predict"]         int   max-generation token budget
          slot["params"]["reasoning_format"] str
          slot["params"]["reasoning_in_content"] bool

        Fields that do NOT exist (common misconceptions):
          "state", "n_past", "prompt_progress"
        """
        n_ctx       = slot.get("n_ctx", 1)
        n_processed = slot.get("n_prompt_tokens_processed", 0)
        n_cache     = slot.get("n_prompt_tokens_cache", 0)
        n_prompt    = slot.get("n_prompt_tokens", 0)
        # n_tokens_predicted does NOT exist as a top-level /slots field in the
        # current llama.cpp schema (server-context.cpp server_slot::to_json()).
        # The max-generation budget is params["n_predict"] (same as max_tokens).
        params_raw  = slot.get("params", {})
        n_predicted = params_raw.get("n_predict", 0)

        # n_decoded lives in next_token[0]["n_decoded"], NOT at the top level.
        next_token = slot.get("next_token", [])
        n_decoded = next_token[0].get("n_decoded", 0) if next_token else 0

        snapshot.slot_capacity             = n_ctx
        snapshot.n_prompt_tokens           = n_prompt
        snapshot.n_prompt_tokens_processed = n_processed
        snapshot.n_prompt_tokens_cache     = n_cache
        snapshot.n_decoded                 = n_decoded

        # Guard: slot is_processing=True but n_prompt==0 (task just finished,
        # prompt consumed, slot hasn't been cleared by llama.cpp yet).
        # Use the last-seen runtime value so Context Usage doesn't blink to 0%.
        if n_prompt == 0 and self._prev_prompt_tokens > 0:
            n_prompt = self._prev_prompt_tokens
        # No previous value yet (first request) — fall back to the historical
        # maximum, which is better than 0 but still illogical when the prior
        # request was much larger than the current one.
        if n_prompt == 0 and self._max_context > 0:
            n_prompt = self._max_context
        # Startup edge case: neither previous nor high-water values exist, and
        # the slot has been activated by a real request (i.e. n_ctx was seen at
        # least once and _last_slot_capacity > 0), but llama-server reports
        # n_prompt==0 after prompt consumption. Fall back to n_ctx as the
        # best available signal that context is in use.
        # When _last_slot_capacity is still 0 (true first tick, no request sent
        # yet), leave n_prompt at 0 so Context Usage displays 0% on startup.
        if n_prompt == 0 and n_ctx > 0 and self._last_slot_capacity > 0:
            n_prompt = n_ctx

        snapshot.context_tokens            = n_prompt

        # KV cache occupancy: prefix reused + tokens evaluated this turn +
        # tokens generated so far.  All three live in the KV cache simultaneously.
        kv_tokens = n_cache + n_processed + n_decoded
        snapshot.kv_cache_tokens = kv_tokens
        snapshot.kv_cache_usage  = kv_tokens / n_ctx if n_ctx > 0 else 0.0

        # Prompt progress: fraction of the full prompt that has been loaded into
        # the KV cache (either reused from a prior request or freshly evaluated).
        #
        # Correct formula (matches server-context.cpp log: progress = n_past / n_tokens):
        #   progress = (n_cache + n_processed) / n_prompt
        #
        # Using only n_processed / n_prompt was WRONG: when KV-cache prefix reuse is
        # active the bar would plateau at (n_prompt - n_cache)/n_prompt and never
        # reach 100 %, because the cached tokens are never re-evaluated.
        #
        # Guard against n_prompt==0 (slot just assigned, task not yet started).
        if n_prompt > 0:
            snapshot.prompt_progress = min((n_cache + n_processed) / n_prompt, 1.0)
        else:
            snapshot.prompt_progress = 0.0

        # params_raw was already extracted above for n_predicted
        snapshot.reasoning_format     = params_raw.get("reasoning_format", "")
        snapshot.reasoning_in_content = params_raw.get("reasoning_in_content", False)
        snapshot.n_tokens_predicted   = n_predicted

        # Update high-water marks
        if n_ctx > self._last_slot_capacity:
            self._last_slot_capacity = n_ctx
        if n_prompt > self._max_prompt_tokens:
            self._max_prompt_tokens = n_prompt
        if n_processed > self._max_processed:
            self._max_processed = n_processed
        if snapshot.context_tokens > self._max_context:
            self._max_context = snapshot.context_tokens
        if snapshot.prompt_progress > 0:
            self._last_prompt_progress = snapshot.prompt_progress
        if snapshot.reasoning_format:
            self._last_reasoning_format     = snapshot.reasoning_format
            self._last_reasoning_in_content = snapshot.reasoning_in_content

        # Update last-seen values for the safeguard logic (always write; the
        # safeguard at the top only uses them when n_prompt==0, i.e. after
        # the prompt was consumed, so the *previous* iteration's value is what
        # matters).
        # BUG FIX: save the *effective* n_prompt (post-fallback), not the raw
        # slot value stored in snapshot.n_prompt_tokens (which is the pre-fallback
        # zero).  snapshot.context_tokens holds the post-fallback value (set at
        # line 222 from the corrected n_prompt), so mirror that here.
        # Using snapshot.n_prompt_tokens was wrong: on the very first tick that
        # n_prompt drops to 0 the raw value would overwrite the good previous
        # value, making _prev_prompt_tokens==0 and causing the safeguard to
        # fall through to _max_context (an unrelated high-water mark) on the
        # next tick — producing the visible discontinuity in Context Usage.
        self._prev_context_tokens = snapshot.context_tokens
        self._prev_prompt_tokens  = n_prompt   # n_prompt is the effective value after all fallbacks above

    def _apply_cached_slot_data(self, snapshot: MetricSnapshot) -> None:
        """Restore high-water / last-seen values when no active slot is found."""
        snapshot.context_tokens            = self._max_context
        snapshot.slot_capacity             = self._last_slot_capacity if self._last_slot_capacity > 0 else 1
        snapshot.n_prompt_tokens           = self._max_prompt_tokens
        snapshot.n_prompt_tokens_processed = self._max_processed
        # prompt_progress, kv_cache_tokens, kv_cache_usage left at 0 when idle —
        # there is no meaningful cached value to show between requests.

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------

    def _collect_loop(self) -> None:
        while self._running:
            try:
                snapshot = self._fetch_and_parse()
                self._update_state(snapshot)

                with self._lock:
                    self._current = snapshot
                    self._history.append(snapshot)
                    if len(self._history) > self._history_max:
                        self._history.pop(0)
                    self._consecutive_failures = 0
                    if not self._connected:
                        self._success_after_offline += 1
                        if self._success_after_offline >= 2:
                            self._connected = True
                            self._success_after_offline = 0
                            self._last_error = None
                    else:
                        self._last_error = None

            except requests.ConnectionError:
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 2:
                        self._connected  = False
                        self._last_error = "Connection refused"
            except requests.Timeout:
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 2:
                        self._connected  = False
                        self._last_error = "Request timeout"
            except requests.HTTPError as e:
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 2:
                        self._connected  = False
                        self._last_error = f"HTTP {e.response.status_code}"
            except Exception as e:
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 2:
                        self._connected  = False
                        self._last_error = str(e)

            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Fetch + parse
    # ------------------------------------------------------------------

    def _fetch_and_parse(self) -> MetricSnapshot:
        """Fetch /slots and /metrics; build and return a MetricSnapshot."""
        now      = time.time()
        snapshot = MetricSnapshot(timestamp=now)

        # --- /health (optional; just to prime connection check) --------
        try:
            r = requests.get(f"{self.server_url}{self.HEALTH_ENDPOINT}", timeout=2.0)
            r.raise_for_status()
        except requests.RequestException:
            pass  # health is informational; failure handled by /slots below

        # --- /slots (primary; failure marks server offline) ------------
        # Re-raise so _collect_loop sets connected=False.
        slots_resp = requests.get(
            f"{self.server_url}{self.SLOTS_ENDPOINT}", timeout=2.0
        )
        slots_resp.raise_for_status()
        slots: list[dict] = slots_resp.json()
        self._raw_slots = slots

        # State detection: "state" integer is NOT in the /slots JSON.
        # Use is_processing (bool, top-level) combined with n_decoded which is
        # nested at slot["next_token"][0]["n_decoded"].
        #   is_processing=False            → IDLE
        #   is_processing=True, n_decoded=0 → PREFILL  (evaluating prompt)
        #   is_processing=True, n_decoded>0 → INFERENCE (generating tokens)
        def _slot_n_decoded(s: dict) -> int:
            nt = s.get("next_token", [])
            return nt[0].get("n_decoded", 0) if nt else 0

        if slots:
            any_processing = any(s.get("is_processing", False) for s in slots)
            any_decoding   = any(_slot_n_decoded(s) > 0 for s in slots)
            snapshot.is_prefill   = any_processing and not any_decoding
            snapshot.is_inference = any_processing and any_decoding
        else:
            snapshot.is_prefill   = False
            snapshot.is_inference = False

        # Reset prompt_progress when not prefilling — there's nothing to show.
        if not snapshot.is_prefill:
            self._last_prompt_progress = 0.0

        # Extract slot context data
        if slots:
            active_slots = [s for s in slots if s.get("is_processing", False)]
            if active_slots:
                self._extract_slot_data(active_slots[0], snapshot)
            else:
                slots_with_tokens = [
                    s for s in slots
                    if s.get("n_prompt_tokens_processed", 0) > 0
                    or _slot_n_decoded(s) > 0
                ]
                if slots_with_tokens:
                    self._extract_slot_data(slots_with_tokens[0], snapshot)
                else:
                    self._apply_cached_slot_data(snapshot)
        else:
            self._apply_cached_slot_data(snapshot)

        # --- /metrics (optional; failure does NOT mark server offline) -
        raw_metrics_text: Optional[str] = None
        try:
            metrics_resp = requests.get(
                f"{self.server_url}{self.METRICS_ENDPOINT}", timeout=2.0
            )
            metrics_resp.raise_for_status()
            raw_metrics_text = metrics_resp.text
            self._parse_prometheus(raw_metrics_text, snapshot)
        except requests.RequestException:
            # /metrics may not be enabled (needs --metrics flag).
            # If /slots succeeded we are still online.
            pass

        # --- /props (optional; model name, overrides /metrics) ----------
        # /props is a newer llama.cpp endpoint.  /metrics `llama_model_name`
        # is a fallback when /props is unavailable.
        if not snapshot.model_name:
            try:
                props_resp = requests.get(
                    f"{self.server_url}{self.PROPS_ENDPOINT}", timeout=2.0
                )
                props_resp.raise_for_status()
                props_data = props_resp.json()
                # Prefer model_alias (short name), fall back to model_path
                model_alias = props_data.get("model_alias", "")
                model_path  = props_data.get("model_path", "")
                if model_alias:
                    snapshot.model_name = model_alias
                elif model_path:
                    snapshot.model_name = os.path.basename(model_path).replace(".gguf", "")
            except requests.RequestException:
                pass  # /props not enabled — model_name stays from /metrics

        self._calculate_speeds(snapshot)
        # Store raw metrics text for debug logging (thread-safe)
        with self._lock:
            self._raw_metrics_text = raw_metrics_text
        return snapshot

    # ------------------------------------------------------------------
    # Prometheus parser
    # ------------------------------------------------------------------

    def _parse_prometheus(self, text: str, snapshot: MetricSnapshot) -> None:
        """Parse Prometheus exposition format into *snapshot*."""
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            raw_name = parts[0]
            label_idx = raw_name.find('{')
            metric_name = raw_name[:label_idx] if label_idx != -1 else raw_name

            try:
                value = float(parts[1])
            except ValueError:
                continue

            # Cumulative token counters (used for delta-speed in _calculate_speeds)
            if metric_name in ("llamacpp:prompt_tokens_total",
                               "llamacpp_prompt_tokens_total"):
                snapshot.prompt_tokens_total = int(value)

            elif metric_name in ("llamacpp:n_tokens_max",
                                 "llamacpp_n_tokens_max"):
                snapshot.n_tokens_max = int(value)

            elif metric_name in ("llamacpp:tokens_predicted_total",
                                 "llamacpp_tokens_predicted_total"):
                snapshot.tokens_predicted_total = int(value)

            # FIX 6: Direct speed metrics as fallback (populated even when the
            # delta is zero because we haven't seen a new token yet this poll)
            elif metric_name in ("llamacpp:prompt_tokens_seconds",
                                 "llamacpp_prompt_tokens_seconds"):
                if snapshot.prefill_speed == 0.0:
                    snapshot.prefill_speed = value

            elif metric_name in ("llamacpp:predicted_tokens_seconds",
                                 "llamacpp_predicted_tokens_seconds"):
                if snapshot.inference_speed == 0.0:
                    snapshot.inference_speed = value

            # KV cache usage/ratio metrics are NOT emitted by llama.cpp /metrics.
            # KV cache is computed from /slots fields in _extract_slot_data.
            # (Matchers for kv_cache_usage_ratio / kv_cache_tokens_count removed.)

            elif metric_name in ("llamacpp:n_decode_total",
                                 "llamacpp_n_decode_total"):
                snapshot.n_decode_total = int(value)

            elif metric_name in ("llamacpp:requests_processing",
                                 "llamacpp_requests_processing"):
                snapshot.requests_processing = int(value)

            elif metric_name in ("llamacpp:requests_deferred",
                                 "llamacpp_requests_deferred"):
                snapshot.requests_deferred = int(value)

            # Model name from /props (reliable fallback when /metrics has no llama_model_name)
            elif metric_name == "llama_model_name":
                if 'filename="' in line:
                    snapshot.model_name = line.split('filename="')[1].split('"')[0]

   # ------------------------------------------------------------------
    # Speed calculation
    # ------------------------------------------------------------------

    def _calculate_speeds(self, current: MetricSnapshot) -> None:
        """Calculate smoothed speeds from token-counter deltas."""
        is_active = current.is_prefill or current.is_inference

        # FIX 5: Clear stale deque values when the server is idle so that
        # leftover samples from the previous generation don't appear in graphs.
        if not is_active:
            self._prefill_avg.clear()
            self._inference_avg.clear()
            # Keep whatever the /metrics endpoint reported (may be 0)
            return

        if not self._history:
            return

        prev       = self._history[-1]
        time_delta = current.timestamp - prev.timestamp

        if time_delta > 0:
            eval_delta = current.prompt_tokens_total - prev.prompt_tokens_total
            if eval_delta > 0:
                # FIX 3: Only add a sample to the moving average if it's
                # within a reasonable bound (<= 499 tok/s prefill). This
                # prevents single-sample outliers from corrupting the
                # displayed trend before the average is even computed.
                delta_speed = eval_delta / time_delta
                if delta_speed <= 499:
                    self._prefill_avg.append(delta_speed)

            pred_delta = current.tokens_predicted_total - prev.tokens_predicted_total
            if pred_delta > 0:
                delta_speed = pred_delta / time_delta
                if delta_speed <= 99:
                    self._inference_avg.append(delta_speed)

        # Override with smoothed moving-average when we have samples
        if self._prefill_avg:
            # Use average of the last N samples to avoid single-sample spikes
            current.prefill_speed = sum(self._prefill_avg) / len(self._prefill_avg)
        if self._inference_avg:
            current.inference_speed = sum(self._inference_avg) / len(self._inference_avg)

        # Double-check: if the smoothed moving average itself is still above
        # threshold (e.g., all recent samples were bad), fall back to the
        # average value rather than using a raw sample that could itself be
        # anomalous.
        if self._prefill_avg and current.prefill_speed > 499:
            current.prefill_speed = sum(self._prefill_avg) / len(self._prefill_avg)
        if self._inference_avg and current.inference_speed > 99:
            current.inference_speed = sum(self._inference_avg) / len(self._inference_avg)

    # ------------------------------------------------------------------
    # State log
    # ------------------------------------------------------------------

    def _update_state(self, snapshot: MetricSnapshot) -> None:
        if snapshot.is_prefill:
            current_state = "PREFILL"
        elif snapshot.is_inference:
            current_state = "INFERENCE"
        else:
            current_state = "IDLE"

        if current_state != self._last_state:
            with self._lock:
                self._state_log.append((snapshot.timestamp, current_state))
                if len(self._state_log) > self._log_max:
                    self._state_log.pop(0)
            self._last_state = current_state

    # ------------------------------------------------------------------
    # Public accessors (all thread-safe)
    # ------------------------------------------------------------------

    def get_snapshot(self) -> MetricSnapshot:
        with self._lock:
            return self._current

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def get_state_log(self) -> list[tuple[float, str]]:
        with self._lock:
            return list(self._state_log)

    def get_speed_history(self) -> tuple[list[float], list[float]]:
        with self._lock:
            return (
                [s.prefill_speed   for s in self._history],
                [s.inference_speed for s in self._history],
            )

    def get_raw_slots(self) -> list[dict]:
        """Return the most recent raw /slots list (thread-safe)."""
        with self._lock:
            return list(self._raw_slots)

    def get_raw_metrics_text(self) -> Optional[str]:
        """Return the most recent raw /metrics text (thread-safe)."""
        with self._lock:
            return self._raw_metrics_text


# ---------------------------------------------------------------------------
# Graph renderer
# ---------------------------------------------------------------------------

_GRAPH_CHARS = "▁▂▃▄▅▆▇█"


def _render_graph(speeds: list[float], width: int, style: str) -> Text:
    """Render a Unicode bar graph.  Always renders exactly *width* characters."""
    text = Text()

    if not speeds or all(v == 0 for v in speeds):
        text.append("─" * width, style="dim")
        return text

    max_val = max(speeds)
    if max_val <= 0:
        text.append("─" * width, style="dim")
        return text

    # FIX 10: Trim / pad so the graph always fills exactly *width* columns.
    # Take the *most recent* `width` samples when we have more.
    display = speeds[-width:] if len(speeds) >= width else speeds
    # Left-pad with dim dashes when fewer samples exist
    pad = width - len(display)
    if pad:
        text.append("─" * pad, style="dim")

    for v in display:
        if v == 0:
            text.append("▁", style="dim")
        else:
            # FIX in zai-yald.py already present; kept: clamp to valid index
            level = min(len(_GRAPH_CHARS) - 1, max(0, int(v / max_val * len(_GRAPH_CHARS)) - 1))
            text.append(_GRAPH_CHARS[level], style=style)

    return text


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def create_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left",       ratio=2),
        Layout(name="right",      ratio=3),
    )
    layout["left"].split(
        Layout(name="metrics"),
        Layout(name="performance", size=5),
    )
    layout["right"].split(
        Layout(name="slots"),
    )
    return layout


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def make_header(
    connected: bool,
    server_url: str = "",
    error: Optional[str] = None,
) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    title = Text()
    title.append("YALD", style="bold cyan")
    title.append(" Yet Another Llama Dashboard", style="dim white")

    if connected:
        status = Text("● ONLINE", style="bold green")
    else:
        status = Text("● OFFLINE / DISCONNECTED", style="bold red")
        if error:
            status.append(f" ({error})", style="dim red")

    # Right side: server address (stripped of scheme), then status icon
    if server_url:
        addr = server_url.removeprefix("https://").removeprefix("http://")
        right = Text()
        right.append(f"{addr}  ", style="dim yellow")
        right.append(status)
    else:
        right = status

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="center", ratio=1)
    header.add_column(justify="right")
    header.add_row(title, Text(now, style="bold yellow"), right)

    return Panel(header, style="bold white on blue")


def make_metrics_panel(snapshot: MetricSnapshot, _frame: int = 0) -> Panel:
    """Left panel: state, context, progress, token counts, queue, KV cache."""

    # --- State badge ---------------------------------------------------
    if snapshot.is_prefill:
        state_text = Text("PREFILL", style="bold white on magenta")
    elif snapshot.is_inference:
        state_text = Text("ACTIVE", style="bold white on green")
    else:
        state_text = Text("IDLE", style="bold white on blue")

    table = Table.grid(expand=True)
    table.add_column(style="cyan", width=22)
    table.add_column(style="bold white")

    table.add_row("State:", state_text)
    table.add_row("")

     # --- Context usage bar -----------------------------------------------
    # Use slot_capacity (n_ctx, the true KV-cache capacity) rather than
    # n_tokens_max (the Prometheus gauge for max-generation budget / n_predict).
    # n_tokens_max is a semantic mismatch for context-window sizing and can
    # trivially saturate the ratio to 1.0 even when the KV cache is mostly empty.
    # Colour coding: green <= 69.9%, yellow 70–80%, red > 80%.
    max_ctx       = snapshot.slot_capacity if snapshot.slot_capacity > 0 else 1
    ctx_ratio     = min(snapshot.context_tokens / max_ctx, 1.0)
    # Guard: never display exactly 100.0% — cap at 99.5% to leave a visual
    # safety buffer.  This also covers the idle-cached-state edge case.
    if ctx_ratio >= 1.0:
        ctx_ratio = 0.995
    ctx_filled    = int(ctx_ratio * 20)
    ctx_pct       = round(ctx_ratio * 100, 1)

    # Colour-coded bar: green <= 69.9%, orange 70–85%, red > 85%.
    if ctx_pct > 85:
        bar_style = 'red'
    elif ctx_pct >= 70:
        bar_style = 'dark_orange'
    else:
        bar_style = 'green'

    # Blink when > 95 % — toggle visibility every 10 frames ≈ 1 s at 10 FPS.
    _blink_off = (ctx_pct > 95) and (_frame % 20 < 10)
    if _blink_off:
        context_bar = "░" * 20
        context_row = f"[dim][{context_bar}] {ctx_ratio:.1%}[/dim]"
    else:
        context_bar = f'[{bar_style}]{"█" * ctx_filled}{"░" * (20 - ctx_filled)}[/{bar_style}]'
        context_row = f"[{context_bar}] {ctx_ratio:.1%} "

    table.add_row("Max Context:",   f"{snapshot.slot_capacity:,}")
    table.add_row("Context Usage:", context_row)
    table.add_row("")

    # Removed: Prompt Progress bar — each slot now has its own prompt progress.

    # --- Token counts ----------------------------------------------------
    table.add_row("Tokens Max:",    f"{snapshot.n_prompt_tokens:,}")
    table.add_row("Processed:",        f"{snapshot.n_prompt_tokens_processed:,}")
    table.add_row("Generated:",        f"{snapshot.n_decoded:,}")
    table.add_row("Prompt Total:", f"{snapshot.prompt_tokens_total:,}")
    table.add_row("Predict Total:",     f"{snapshot.tokens_predicted_total:,}")
    table.add_row("Decode Calls:",     f"{snapshot.n_decode_total:,}")

    table.add_row("")

    # --- Concurrency -------------------------------------------------------
    table.add_row("Processing:",   f"{snapshot.requests_processing}")
    table.add_row("Deferred:",     f"{snapshot.requests_deferred}")

    # --- Reasoning (model-specific) --------------------------------------
    if snapshot.reasoning_format:
        r_style = "green" if snapshot.reasoning_in_content else "yellow"
        table.add_row("")
        table.add_row("Reasoning:", f"[{r_style}]{snapshot.reasoning_format}[/{r_style}]")

    # --- Model identification -------------------------------------------
    if snapshot.model_name:
        table.add_row("")
        table.add_row("Model:", f"[dark_orange]{snapshot.model_name}[/dark_orange]")

    return Panel(table, title="[bold]Metrics[/bold]", border_style="cyan")


def make_performance_panel(snapshot: MetricSnapshot,
                           collector: MetricsCollector) -> Panel:
    """Right panel: prompt & generation speeds side by side on one line."""

    def _speed_text(value: float, active_style: str, unit_style: str) -> Text:
        t = Text(f"{value:.1f}", style=active_style if value > 0 else "dim")
        t.append(" tok/s", style=unit_style if value > 0 else "dim")
        return t

    # Speed values side-by-side: no outlines, no graphs — minimal height.
    prefill_text = _speed_text(snapshot.prefill_speed, "bold magenta", "dim magenta")
    infer_text   = _speed_text(snapshot.inference_speed, "bold green", "dim green")

    # 3-column grid: label | divider | value — mirrors Metrics panel width.
    perf_table = Table.grid(expand=True)
    perf_table.add_column(style="cyan", width=22)
    perf_table.add_column(style="dim", width=1)
    perf_table.add_column(style="bold white")

    # Top row: labels with vertical divider
    perf_table.add_row("PROMPT SPEED:", " ", "GENERATION SPEED:")
    # Bottom row: values
    perf_table.add_row(prefill_text, " ", infer_text)
    # Spacer rows to reach 5 total
    perf_table.add_row("", " ", "")
    perf_table.add_row("", " ", "")

    return Panel(perf_table, title="[bold]Performance[/bold]", border_style="green", padding=(0, 0))


def _build_slot_cell(slot: dict, slot_idx: int) -> Panel:
    """Build a single cell for one slot quadrant with state, progress bars, tokens.

    Fixed inner height of 11 lines, yielding exactly 13 character rows
    including the top/bottom panel borders — never expands regardless of
    content length.
    """
    inner_height = 10
    is_processing = slot.get("is_processing", False)
    n_ctx         = slot.get("n_ctx", 0)
    n_prompt      = slot.get("n_prompt_tokens", 0)
    n_processed   = slot.get("n_prompt_tokens_processed", 0)
    n_cache       = slot.get("n_prompt_tokens_cache", 0)
    params_raw    = slot.get("params", {})
    n_predict     = params_raw.get("n_predict", 0)

    next_token = slot.get("next_token", [])
    n_decoded  = next_token[0].get("n_decoded", 0) if next_token else 0
    n_remain   = next_token[0].get("n_remain", 0) if next_token else 0

    # Determine state and color
    if not is_processing and n_decoded == 0:
        state_label = f"Slot {slot_idx} ○"
        state_style = "dim"
        border_style = "blue"
        state_text = ""
    elif is_processing and n_decoded == 0:
        state_label = f"Slot {slot_idx} ●"
        state_style = "bold white on blue"
        border_style = "blue"
        state_text = "IDLE"
    elif is_processing and n_decoded > 0:
        state_label = f"Slot {slot_idx} ●"
        state_style = "bold white on green"
        border_style = "green"
        state_text = "ACTIVE"
    else:
        # Processing but n_decoded==0 yet — still prefill
        state_label = f"Slot {slot_idx} ●"
        state_style = "bold white on blue"
        border_style = "blue"
        state_text = "IDLE"

    gen_label = f"Gen: ({n_decoded}/{n_predict})"

    prog_label = f"Prompt: ({n_processed}/{n_prompt})" if n_prompt > 0 else "Prompt: (0/0)"

    # --- KV cache per slot (n_cache + n_processed + n_decoded) / n_ctx ------
    kv_tokens = n_cache + n_processed + n_decoded
    kv_ratio  = min(kv_tokens / n_ctx, 1.0) if n_ctx > 0 else 0.0
    kv_filled = int(kv_ratio * 20)
    kv_bar    = "█" * kv_filled + "░" * (20 - kv_filled)
    kv_label  = f"KV: [{kv_bar}] {kv_ratio:.1%} ({kv_tokens}/{n_ctx})"

    # --- KV cache token detail (shown only when slot is active) -------------
    kv_detail = ""
    if is_processing:
        kv_detail = f"  KV: cache={n_cache:,} processed={n_processed:,} decoded={n_decoded:,}"

    # --- Remaining tokens -----------------------------------------------
    rem_label = ""
    if n_predict > 0:
        rem = n_remain
        rem_label = f"  Remaining: {rem:,} tok"

    # --- Assemble cell content, always *inner_height* lines ----------------
    # State label + state badge on the same line; then generation, prompt, KV.
    # Extras added when height allows; remaining rows are blank spacers so
    # each quadrant fills its allocated vertical space evenly.  Extra lines
    # are silently dropped so the cell never expands beyond 13 rows total.
    lines: list[Text] = [
        Text(f"{state_label}  {state_text}", style=state_style),
        Text(gen_label, style="bold white"),
        Text(prog_label, style="bold white"),
        Text(kv_label, style="cyan"),
    ]
    if kv_detail:
        lines.append(Text(kv_detail, style="dim"))
    if rem_label:
        lines.append(Text(rem_label, style="dim"))
    # Pad or truncate to exactly *inner_height* lines
    while len(lines) < inner_height:
        lines.append(Text(style=""))
    lines = lines[:inner_height]

    inner = Table.grid(expand=False)
    inner.add_column()
    for line in lines:
        inner.add_row(line)

    return Panel(inner, title=None, border_style=border_style, expand=False, height=13)


def make_slots_panel(connected: bool, raw_slots: list[dict], term_height: int) -> Panel:
    """Slots panel: 4 quadrants (Slot 0-3) with state, generation/prompt/KV progress bars.

    Each quadrant is a fixed 13-row panel (11 inner lines + top/bottom borders),
    independent of terminal size or content length.
    """

    table = Table.grid(expand=True)
    table.add_column(justify="center", ratio=1)
    table.add_column(justify="center", ratio=1)

    # Top row: Slot 0 & Slot 1
    top_cells = [None, None]
    for slot_idx in range(0, 2):
        slot = raw_slots[slot_idx] if slot_idx < len(raw_slots) else None
        if slot:
            top_cells[slot_idx - 0] = _build_slot_cell(slot, slot_idx)
        else:
            label = Text(f"Slot {slot_idx} ○", style="dim")
            top_cells[slot_idx - 0] = Panel(label, title=None, border_style="blue", expand=False, height=13)
    table.add_row(*top_cells)

    # Bottom row: Slot 2 & Slot 3
    bottom_cells = [None, None]
    for slot_idx in range(2, 4):
        slot = raw_slots[slot_idx] if slot_idx < len(raw_slots) else None
        if slot:
            bottom_cells[slot_idx - 2] = _build_slot_cell(slot, slot_idx)
        else:
            label = Text(f"Slot {slot_idx} ○", style="dim")
            bottom_cells[slot_idx - 2] = Panel(label, title=None, border_style="blue", expand=False, height=13)
    table.add_row(*bottom_cells)

    if not connected or not raw_slots:
        return Panel(table, title="[bold]Slots[/bold]", border_style="dim", expand=True)

    return Panel(table, title="[bold]Slots[/bold]", border_style="yellow", expand=True)


def make_footer(collector: MetricsCollector) -> Panel:
    """Activity log: last N state transitions."""
    state_log = collector.get_state_log()

    if not state_log:
        log_text = Text("Waiting for state transitions…", style="dim")
    else:
        log_text = Text()
        for i, (ts, state) in enumerate(state_log[-5:]):
            if i > 0:
                log_text.append(" → ", style="dim")
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            if state == "PREFILL":
                log_text.append(f"{time_str} PREFILL",   style="magenta")
            elif state == "INFERENCE":
                log_text.append(f"{time_str} INFERENCE", style="green")
            else:
                log_text.append(f"{time_str} IDLE",      style="blue")

    return Panel(log_text, title="[bold]Activity Log[/bold]", border_style="dim")


def make_offline_panel(error: str) -> Panel:
    text = Text()
    text.append("⚠  SERVER OFFLINE\n\n",         style="bold red")
    text.append(f"Error: {error}\n\n",            style="yellow")
    text.append("Ensure llama-server is running with:\n", style="dim")
    text.append("  llama-server --metrics --port 8080\n", style="cyan")
    return Panel(text, title="[bold red]CONNECTION LOST[/bold red]", border_style="red")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class YALDApplication:
    def __init__(self, server_url: str = "http://127.0.0.1:8080",
                 debug_file: Optional[str] = None):
        # Auto-prepend http:// if the user omitted the scheme
        if not server_url.startswith(("http://", "https://")):
            server_url = "http://" + server_url
        self.collector = MetricsCollector(server_url=server_url, poll_interval=0.5)
        self.layout    = create_layout()
        self.console   = Console()
        self._running  = True
        self._frame    = 0

        # Debug logging path (file handle opened lazily in run(), closed in finally)
        self._debug_path = debug_file
        self._debug_fp: Optional[any] = None  # type: ignore[assignment]

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _log_debug(self, raw_slots: list[dict], metrics_text: str | None) -> None:
        """Append a single JSONL record with raw server responses."""
        if self._debug_fp is None:
            return
        record = {
            "ts": datetime.now().isoformat(),
            "slots": raw_slots,
        }
        if metrics_text:
            record["metrics"] = metrics_text
        self._debug_fp.write(json.dumps(record, default=str) + "\n")

    def _handle_signal(self, signum, frame) -> None:
        self._running = False

    def _update_layout(self) -> None:
        connected = self.collector.is_connected()
        error     = self.collector.get_last_error()
        snapshot  = self.collector.get_snapshot()
        raw_slots = self.collector.get_raw_slots()
        height    = self.console.size.height
        self._frame += 1

        # Debug: write raw server responses to JSONL file
        if self._debug_fp is not None:
            self._log_debug(raw_slots, self.collector.get_raw_metrics_text())

        self.layout["header"].update(make_header(connected, self.collector.server_url, error))

        if connected:
            self.layout["metrics"].update(make_metrics_panel(snapshot, self._frame))
            self.layout["performance"].update(
                make_performance_panel(snapshot, self.collector)
            )
            self.layout["slots"].update(
                make_slots_panel(connected, raw_slots, height)
            )
        else:
            self.layout["body"].update(make_offline_panel(error or "Unknown error"))

        self.layout["footer"].update(make_footer(self.collector))

    def run(self) -> None:
        # Open debug file lazily here so the handle is scoped to run()'s
        # lifecycle — even if run() is never called the file is never
        # opened, and if it is closed abnormally the finally below
        # guarantees cleanup.
        self._debug_fp: Optional[any] = None  # type: ignore[assignment]
        if self._debug_path:
            self._debug_fp = open(self._debug_path, "w", buffering=1)  # line-buffered
            print(f"[YALD] Debug log: {self._debug_path}")

        self.collector.start()
        try:
            with Live(
                self.layout,
                console=self.console,
                refresh_per_second=10,
                screen=True,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                while self._running:
                    self._update_layout()
                    live.update(self.layout)
                    time.sleep(0.01)
        finally:
            self.collector.stop()
            if self._debug_fp is not None:
                self._debug_fp.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="YALD - Yet Another Llama Dashboard. \u00A9 2026 Antani Technologies BV.")
    parser.add_argument(
        "--server", "-s",
        default="http://127.0.0.1:8080",
        help="llama-server URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--debug",
        help="Path to a JSONL debug log file; writes raw /slots and /metrics responses each frame",
    )
    args = parser.parse_args()

    YALDApplication(server_url=args.server, debug_file=args.debug).run()


if __name__ == "__main__":
    main()