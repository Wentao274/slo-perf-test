#!/usr/bin/env python3
"""Long-context full-load SLO test: max sustainable arrival rate search.

Simulates production long-context workload with Poisson arrival (open loop).
For each input length, finds the maximum arrival rate (req/s) that satisfies
TTFT and TPOT SLO over a sustained time window (default 10 minutes).

Methodology:
  - Two-stage search per case: pre (short duration) → formal (full duration).
  - Pre stage: coarse rate search with shorter duration to locate boundary.
  - Formal stage: full-duration verification at the boundary rate.
  - Each test point: evalscope perf with --open-loop --rate N --duration T.

Result metrics per case:
  到达 req/s, 每卡总吞吐 tok/s, 每卡输出 tok/s, TTFT p50/p99,
  TPOT p50/p90, 单用户生成速度, 单机月收入.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import secrets
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Standard test cases (aligned with PDF Section 3.3)
# ---------------------------------------------------------------------------
STANDARD_CASES: list[dict[str, Any]] = [
    {"input": 64 * 1024, "output": 600, "cache_ratio": 0.90},
    {"input": 77 * 1024, "output": 600, "cache_ratio": 0.90},
    {"input": 128 * 1024, "output": 600, "cache_ratio": 0.90},
    {"input": 160 * 1024, "output": 900, "cache_ratio": 0.91},
]

# ---------------------------------------------------------------------------
# Default SLO thresholds (aligned with PDF Table 3)
# ---------------------------------------------------------------------------
DEFAULT_SLO: dict[str, float] = {
    "p50_ttft_ms": 2500.0,   # TTFT p50 <= 2.5 s
    "p99_ttft_ms": 5000.0,   # TTFT p99 <= 5 s
    "p90_tpot_ms": 20.0,     # TPOT p90 <= 20 ms
}

# ---------------------------------------------------------------------------
# Default pricing (aligned with GLM-5.3 public catalog pricing)
# ---------------------------------------------------------------------------
DEFAULT_PRICING: dict[str, float] = {
    "input_per_mtok": 8.0,           # 元/百万 tokens (new input)
    "cached_input_per_mtok": 2.0,    # 元/百万 tokens (cache-hit input)
    "output_per_mtok": 28.0,         # 元/百万 tokens (output)
    "gpus_per_server": 8,
    "hours_per_month": 720,
}

# ---------------------------------------------------------------------------
# Default connection
# ---------------------------------------------------------------------------
DEFAULT_CONNECTION: dict[str, Any] = {
    "url": "http://127.0.0.1:8000/v1/chat/completions",
    "api": "openai",
    "api_key": None,
    "model": "glm5",
    "tokenizer_path": "/model",
    "evalscope_bin": "evalscope",
    "warmup_num": 4,
    "gpu_count": 32,
    "extra_evalscope_args": [],
}

# ---------------------------------------------------------------------------
# Search defaults
# ---------------------------------------------------------------------------
DEFAULT_SEARCH: dict[str, Any] = {
    "pre_duration": 120,       # pre-stage duration in seconds
    "formal_duration": 600,    # formal-stage duration in seconds (10 min)
    "pre_rate_start": 1,       # initial rate for pre-stage search
    "pre_rate_max": 256,       # max rate for pre-stage search
    "rate_step": 1,            # minimum rate step for binary search
}


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class Case:
    input_tokens: int
    output_tokens: int
    cache_ratio: float

    @property
    def label(self) -> str:
        inp = self.input_tokens
        return f"{inp // 1024}K" if inp % 1024 == 0 else str(inp)


@dataclass
class PointResult:
    case_label: str
    stage: str                    # "pre" or "formal"
    rate: float                   # target arrival rate (req/s)
    actual_rate: Optional[float] = None  # measured request throughput
    duration_seconds: Optional[float] = None
    total_requests: Optional[int] = None
    success_requests: Optional[int] = None
    failed_requests: Optional[int] = None
    success_rate: Optional[float] = None
    avg_input_tokens: Optional[float] = None
    avg_output_tokens: Optional[float] = None
    request_throughput_rps: Optional[float] = None
    output_token_throughput_tps: Optional[float] = None
    total_token_throughput_tps: Optional[float] = None
    # Latency (seconds) — total request time, used for Little's Law
    avg_latency_seconds: Optional[float] = None
    # TTFT metrics (seconds)
    avg_ttft_seconds: Optional[float] = None
    p50_ttft_seconds: Optional[float] = None
    p90_ttft_seconds: Optional[float] = None
    p99_ttft_seconds: Optional[float] = None
    # TPOT metrics (milliseconds)
    avg_tpot_ms: Optional[float] = None
    p50_tpot_ms: Optional[float] = None
    p90_tpot_ms: Optional[float] = None
    p99_tpot_ms: Optional[float] = None
    # SLO evaluation
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    # Per-GPU derived metrics
    per_gpu_total_tps: Optional[float] = None
    per_gpu_output_tps: Optional[float] = None
    single_user_speed_tps: Optional[float] = None
    monthly_revenue_wan: Optional[float] = None
    # Little's Law: avg_concurrency = actual_rate × avg_latency
    avg_concurrency: Optional[float] = None
    run_directories: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case_label: str
    input_tokens: int
    output_tokens: int
    cache_ratio: float
    pre_max_rate: Optional[float] = None
    formal_max_rate: Optional[float] = None
    final_max_rate: Optional[float] = None
    final_point: Optional[PointResult] = None
    pre_points: list[PointResult] = field(default_factory=list)
    formal_points: list[PointResult] = field(default_factory=list)


# ===================================================================
# Utility functions
# ===================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_token_length(value: Any) -> int:
    """Parse token length like '64K', '64k', '128', '128K tokens'."""
    if isinstance(value, int):
        result = value
    else:
        text = str(value).strip().upper().replace("TOKENS", "").replace("TOKEN", "")
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(K|KI|M|MI)?", text)
        if not match:
            raise ValueError(f"Invalid token length: {value!r}")
        number = float(match.group(1))
        suffix = match.group(2)
        multiplier = 1024 if suffix in {"K", "KI"} else 1024 * 1024 if suffix in {"M", "MI"} else 1
        result = int(number * multiplier)
    if result <= 0:
        raise ValueError(f"Token length must be positive: {value!r}")
    return result


def display_length(tokens: int) -> str:
    if tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return str(tokens)


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def find_value(data: dict[str, Any], aliases: list[str]) -> Any:
    by_norm = {normalized_key(str(key)): value for key, value in data.items()}
    for alias in aliases:
        key = normalized_key(alias)
        if key in by_norm:
            return by_norm[key]
    return None


def as_finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> Optional[int]:
    number = as_finite_float(value)
    return int(round(number)) if number is not None else None


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def format_s(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def format_ms(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def format_rate(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.0f}"


def _fmt(value: Optional[float], fmt: str = ".0f", default: str = "-") -> str:
    """Safe format that returns default for None."""
    if value is None:
        return default
    return f"{value:{fmt}}"


# ===================================================================
# Percentile parsing (extended from prefill_slo.py to support TPOT)
# ===================================================================

def percentile_rows(data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    elif isinstance(data, dict):
        nested = next(
            (v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None
        )
        if nested is not None:
            rows = nested
        else:
            labels_key = next(
                (k for k in data if normalized_key(k) in {"percentiles", "percentile"}), None
            )
            if labels_key and isinstance(data[labels_key], list):
                for index, label in enumerate(data[labels_key]):
                    row = {"Percentiles": label}
                    for key, values in data.items():
                        if isinstance(values, list) and index < len(values):
                            row[key] = values[index]
                    rows.append(row)
    return rows


def parse_percentile(data: Any, family: str) -> dict[str, Optional[float]]:
    """Parse p50/p90/p99 for a metric family ('ttft' or 'tpot')."""
    result: dict[str, Optional[float]] = {"p50": None, "p90": None, "p99": None}
    for row in percentile_rows(data):
        label = str(find_value(row, ["Percentiles", "Percentile"]) or "").strip().lower()
        match = re.search(r"(?:^|\b)p?(50|90|99)(?:\s*%|\b)", label)
        if not match:
            continue
        key = f"p{match.group(1)}"
        column = next(
            (k for k in row if family in normalized_key(k)),
            None,
        )
        if not column:
            continue
        value = as_finite_float(row[column])
        if value is None:
            continue
        # Normalize to consistent units
        if family == "ttft":
            # TTFT columns: may be "TTFT (ms)" or "TTFT (s)"
            if "ms" in column.lower() or "millisecond" in column.lower():
                value /= 1000.0  # convert to seconds
        elif family == "tpot":
            # TPOT columns: should be "TPOT (ms)"
            if "ms" not in column.lower() and "millisecond" not in column.lower():
                value *= 1000.0  # convert to ms
        result[key] = value
    return result


# ===================================================================
# Logging
# ===================================================================

class MasterLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ===================================================================
# Seed management
# ===================================================================

class SeedRegistry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.used: set[int] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    self.used.add(int(line.strip()))
                except ValueError:
                    pass

    def next(self) -> int:
        while True:
            value = secrets.randbelow(2_147_483_646) + 1
            if value not in self.used:
                self.used.add(value)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{value}\n")
                return value


# ===================================================================
# Process execution with idle timeout
# ===================================================================

def run_and_tee(command: list[str], log_path: Path, idle_timeout: Optional[float] = None) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shell_join(command)}\n\n")
        handle.flush()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            message = f"Unable to start command: {exc}"
            print(message, flush=True)
            handle.write(message + "\n")
            return 127
        assert process.stdout is not None

        last_output_at = time.monotonic()

        def tee_stream() -> None:
            nonlocal last_output_at
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                    last_output_at = time.monotonic()
            except Exception:
                pass

        reader = threading.Thread(target=tee_stream, daemon=True)
        reader.start()
        try:
            exit_code: Optional[int] = None
            while exit_code is None:
                try:
                    exit_code = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if idle_timeout is not None and time.monotonic() - last_output_at > idle_timeout:
                        idle_secs = time.monotonic() - last_output_at
                        message = (
                            f"\n# IDLE TIMEOUT: no output for {idle_secs:.0f}s "
                            f"(limit {idle_timeout:.0f}s); terminating PID {process.pid}\n"
                        )
                        print(message, end="", flush=True)
                        handle.write(message)
                        handle.flush()
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        reader.join(timeout=5)
                        return 124
            reader.join(timeout=5)
            return exit_code
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


# ===================================================================
# Configuration loading
# ===================================================================

def load_config(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    return data


def connection_from(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    result = dict(DEFAULT_CONNECTION)
    configured = config.get("connection", {})
    if configured:
        if not isinstance(configured, dict):
            raise ValueError("connection must be a JSON object")
        result.update(configured)
    for name in ("url", "api", "api_key", "model", "tokenizer_path", "evalscope_bin", "warmup_num"):
        value = getattr(args, name, None)
        if value is not None:
            result[name] = value
    if getattr(args, "gpu_count", None) is not None:
        result["gpu_count"] = args.gpu_count
    extras = list(result.get("extra_evalscope_args", []))
    for raw in getattr(args, "evalscope_extra", []) or []:
        extras.extend(shlex.split(raw))
    result["extra_evalscope_args"] = [str(x) for x in extras]
    result["warmup_num"] = int(result["warmup_num"])
    result["gpu_count"] = int(result["gpu_count"])
    return result


def slo_from(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, float]:
    result = dict(DEFAULT_SLO)
    configured = config.get("slo", {})
    if configured:
        if not isinstance(configured, dict):
            raise ValueError("slo must be a JSON object")
        result.update({str(k): float(v) for k, v in configured.items()})
    # CLI overrides
    if getattr(args, "slo_p50_ttft", None) is not None:
        result["p50_ttft_ms"] = args.slo_p50_ttft
    if getattr(args, "slo_p99_ttft", None) is not None:
        result["p99_ttft_ms"] = args.slo_p99_ttft
    if getattr(args, "slo_p90_tpot", None) is not None:
        result["p90_tpot_ms"] = args.slo_p90_tpot
    return result


def pricing_from(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, float]:
    result = dict(DEFAULT_PRICING)
    configured = config.get("pricing", {})
    if configured:
        if not isinstance(configured, dict):
            raise ValueError("pricing must be a JSON object")
        result.update({str(k): float(v) for k, v in configured.items()})
    return result


def search_from(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    result = dict(DEFAULT_SEARCH)
    configured = config.get("search", {})
    if configured:
        if not isinstance(configured, dict):
            raise ValueError("search must be a JSON object")
        result.update(configured)
    if getattr(args, "pre_duration", None) is not None:
        result["pre_duration"] = args.pre_duration
    if getattr(args, "formal_duration", None) is not None:
        result["formal_duration"] = args.formal_duration
    if getattr(args, "pre_rate_start", None) is not None:
        result["pre_rate_start"] = args.pre_rate_start
    if getattr(args, "pre_rate_max", None) is not None:
        result["pre_rate_max"] = args.pre_rate_max
    if getattr(args, "rate_step", None) is not None:
        result["rate_step"] = args.rate_step
    return result


def cases_from(args: argparse.Namespace, config: dict[str, Any]) -> list[Case]:
    raw_cases: list[Any]
    if args.scenario:
        raw_cases = []
        for text in args.scenario:
            parts = text.split(":", 2)
            if len(parts) < 2:
                raise ValueError(f"--scenario must be INPUT:OUTPUT[:CACHE_RATIO], got {text!r}")
            entry: dict[str, Any] = {"input": parts[0], "output": parts[1]}
            if len(parts) == 3:
                entry["cache_ratio"] = float(parts[2])
            raw_cases.append(entry)
    else:
        raw_cases = config.get("cases", STANDARD_CASES)

    cases: list[Case] = []
    for raw in raw_cases:
        if isinstance(raw, Case):
            cases.append(raw)
            continue
        if not isinstance(raw, dict) or "input" not in raw or "output" not in raw:
            raise ValueError(f"Invalid case: {raw!r}; need input and output")
        ratio = float(raw.get("cache_ratio", 0.9))
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"cache_ratio must be in [0, 1), got {ratio}")
        cases.append(
            Case(parse_token_length(raw["input"]), parse_token_length(raw["output"]), ratio)
        )
    if not cases:
        raise ValueError("No test cases resolved")
    return cases


# ===================================================================
# Command builder
# ===================================================================

def build_command(
    connection: dict[str, Any],
    case: Case,
    *,
    rate: float,
    duration: float,
    seed: int,
    outputs_dir: Path,
) -> list[str]:
    """Build evalscope perf command for open-loop Poisson arrival test.

    Open-loop mode: --open-loop --rate N --number M --duration T
    - rate: Poisson arrival rate (req/s)
    - number: total requests = ceil(rate * duration) + buffer
    - duration: max wall-clock seconds
    - parallel: forced to -1 (unbounded) by evalscope
    """
    prefix_length = int(case.input_tokens * case.cache_ratio)
    suffix_length = case.input_tokens - prefix_length
    # Number of requests: rate * duration, with 20% buffer to ensure
    # duration is the limiting factor, not number
    number = max(1, math.ceil(rate * duration * 1.2))

    command = shlex.split(str(connection["evalscope_bin"])) + [
        "perf",
        "--url", str(connection["url"]),
        "--api", str(connection["api"]),
    ]
    if connection.get("api_key"):
        command.extend(["--api-key", str(connection["api_key"])])
    command.extend([
        "--model", str(connection["model"]),
        "--tokenizer-path", str(connection["tokenizer_path"]),
        "--seed", str(seed),
        "--dataset", "random",
        "--warmup-num", str(connection["warmup_num"]),
        "--no-test-connection",
        "--open-loop",
        "--rate", str(rate),
        "--number", str(number),
        "--duration", str(duration),
        "--prefix-length", str(prefix_length),
        "--min-prompt-length", str(suffix_length),
        "--max-prompt-length", str(suffix_length),
        "--min-tokens", str(case.output_tokens),
        "--max-tokens", str(case.output_tokens),
        "--stream",
        "--outputs-dir", str(outputs_dir),
    ])
    command.extend(connection["extra_evalscope_args"])
    return command


# ===================================================================
# Result parsing
# ===================================================================

def read_point(
    run_dir: Path, stage: str, case: Case, rate: float,
    gpu_count: int, pricing: dict[str, float],
) -> Optional[PointResult]:
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None

    point = PointResult(
        case_label=case.label,
        stage=stage,
        rate=rate,
        total_requests=as_int(find_value(summary, ["Total Requests"])),
        success_requests=as_int(find_value(summary, ["Success Requests", "Succeed requests"])),
        failed_requests=as_int(find_value(summary, ["Failed Requests"])),
        duration_seconds=as_finite_float(
            find_value(summary, ["Test Duration (s)", "Time taken for tests (s)"])
        ),
        request_throughput_rps=as_finite_float(
            find_value(summary, ["Req Throughput (req/s)", "Request throughput (req/s)"])
        ),
        output_token_throughput_tps=as_finite_float(
            find_value(summary, ["Output Throughput (tok/s)", "Output token throughput (tok/s)"])
        ),
        total_token_throughput_tps=as_finite_float(
            find_value(summary, ["Total Throughput (tok/s)", "Total token throughput (tok/s)"])
        ),
        avg_input_tokens=as_finite_float(find_value(summary, ["Avg Input Tokens"])),
        avg_output_tokens=as_finite_float(find_value(summary, ["Avg Output Tokens"])),
        run_directories=[str(run_dir.resolve())],
    )

    # Actual rate
    point.actual_rate = point.request_throughput_rps

    # Success rate
    rate_val = as_finite_float(find_value(summary, ["Success Rate"]))
    if rate_val is not None:
        point.success_rate = rate_val if rate_val <= 1.0 else rate_val / 100.0
    elif point.total_requests and point.success_requests is not None:
        point.success_rate = point.success_requests / point.total_requests

    # Avg Latency from summary (total request time in seconds)
    by_norm = {normalized_key(str(key)): key for key in summary}
    latency_key = by_norm.get(normalized_key("Avg Latency (s)")) or by_norm.get(normalized_key("Avg Latency"))
    if latency_key is not None:
        value = as_finite_float(summary[latency_key])
        if value is not None:
            if "ms" in str(latency_key).lower():
                value /= 1000.0
            point.avg_latency_seconds = value

    # Avg TTFT from summary
    ttft_key = by_norm.get(normalized_key("Avg TTFT (ms)")) or by_norm.get(normalized_key("Avg TTFT"))
    if ttft_key is not None:
        value = as_finite_float(summary[ttft_key])
        if value is not None:
            if "ms" not in str(ttft_key).lower():
                value *= 1000.0
            point.avg_ttft_seconds = value / 1000.0

    # Avg TPOT from summary
    tpot_key = by_norm.get(normalized_key("Avg TPOT (ms)")) or by_norm.get(normalized_key("Avg TPOT"))
    if tpot_key is not None:
        value = as_finite_float(summary[tpot_key])
        if value is not None:
            if "ms" not in str(tpot_key).lower():
                value *= 1000.0
            point.avg_tpot_ms = value

    # Percentile metrics from benchmark_percentile.json
    percentile_path = run_dir / "benchmark_percentile.json"
    if percentile_path.is_file():
        try:
            percentiles = json.loads(percentile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            percentiles = None
        if percentiles is not None:
            # TTFT percentiles (convert to seconds)
            ttft = parse_percentile(percentiles, "ttft")
            point.p50_ttft_seconds = ttft["p50"]
            point.p90_ttft_seconds = ttft["p90"]
            point.p99_ttft_seconds = ttft["p99"]
            # TPOT percentiles (in ms)
            tpot = parse_percentile(percentiles, "tpot")
            point.p50_tpot_ms = tpot["p50"]
            point.p90_tpot_ms = tpot["p90"]
            point.p99_tpot_ms = tpot["p99"]

    # Derived per-GPU metrics
    if point.actual_rate and point.actual_rate > 0 and gpu_count > 0:
        inp = point.avg_input_tokens or case.input_tokens
        out = point.avg_output_tokens or case.output_tokens
        point.per_gpu_total_tps = (inp + out) * point.actual_rate / gpu_count
        point.per_gpu_output_tps = out * point.actual_rate / gpu_count

    # Single user generation speed: 1000 / TPOT p50 (ms)
    if point.p50_tpot_ms and point.p50_tpot_ms > 0:
        point.single_user_speed_tps = 1000.0 / point.p50_tpot_ms

    # Monthly revenue calculation
    if point.actual_rate and point.actual_rate > 0:
        inp = point.avg_input_tokens or case.input_tokens
        out = point.avg_output_tokens or case.output_tokens
        cached_tokens = inp * case.cache_ratio
        new_input_tokens = inp * (1.0 - case.cache_ratio)
        # Revenue per request (元)
        rev_per_req = (
            cached_tokens * pricing["cached_input_per_mtok"] / 1e6
            + new_input_tokens * pricing["input_per_mtok"] / 1e6
            + out * pricing["output_per_mtok"] / 1e6
        )
        # Per-GPU per-hour revenue
        per_gpu_per_hour = rev_per_req * point.actual_rate * 3600 / gpu_count
        # Monthly revenue per server (万元)
        point.monthly_revenue_wan = (
            per_gpu_per_hour * pricing["gpus_per_server"] * pricing["hours_per_month"] / 10000
        )

    # Little's Law: avg_concurrency = actual_rate × avg_latency
    if point.actual_rate and point.actual_rate > 0 and point.avg_latency_seconds and point.avg_latency_seconds > 0:
        point.avg_concurrency = point.actual_rate * point.avg_latency_seconds

    return point


# ===================================================================
# SLO evaluation
# ===================================================================

def evaluate_point(point: PointResult, slo: dict[str, float]) -> None:
    """Evaluate a test point against SLO thresholds."""
    # Gate 1: request success
    if point.total_requests is None or point.success_requests is None:
        point.failure_reasons.append("request counts missing")
    if point.failed_requests not in (None, 0):
        point.failure_reasons.append(f"failed requests {point.failed_requests}")
    if point.success_rate is not None and point.success_rate < 1.0:
        point.failure_reasons.append(f"success rate {point.success_rate:.2%} < 100%")

    # Gate 2: SLO metrics
    # TTFT p50
    if "p50_ttft_ms" in slo:
        actual = point.p50_ttft_seconds
        if actual is None:
            point.failure_reasons.append("p50_ttft missing")
        elif actual * 1000 > slo["p50_ttft_ms"]:
            point.failure_reasons.append(
                f"p50_ttft {actual * 1000:.0f}ms > {slo['p50_ttft_ms']:.0f}ms"
            )
    # TTFT p99
    if "p99_ttft_ms" in slo:
        actual = point.p99_ttft_seconds
        if actual is None:
            point.failure_reasons.append("p99_ttft missing")
        elif actual * 1000 > slo["p99_ttft_ms"]:
            point.failure_reasons.append(
                f"p99_ttft {actual * 1000:.0f}ms > {slo['p99_ttft_ms']:.0f}ms"
            )
    # TPOT p90
    if "p90_tpot_ms" in slo:
        actual = point.p90_tpot_ms
        if actual is None:
            point.failure_reasons.append("p90_tpot missing")
        elif actual > slo["p90_tpot_ms"]:
            point.failure_reasons.append(
                f"p90_tpot {actual:.1f}ms > {slo['p90_tpot_ms']:.1f}ms"
            )

    point.passed = not point.failure_reasons


# ===================================================================
# Artifact parsing for a stage
# ===================================================================

def parse_stage_artifacts(
    outputs_dir: Path, stage: str, case: Case, rate: float,
    gpu_count: int, pricing: dict[str, float], slo: dict[str, float],
) -> Optional[PointResult]:
    """Parse all benchmark artifacts from a single rate test point."""
    # Find all benchmark_summary.json files
    summaries = list(outputs_dir.rglob("benchmark_summary.json"))
    if not summaries:
        return None

    # For open-loop single-point tests, there should be exactly one run dir
    # Use the most recent one
    summaries.sort(key=lambda p: p.stat().st_mtime_ns)
    run_dir = summaries[-1].parent

    point = read_point(run_dir, stage, case, rate, gpu_count, pricing)
    if point is not None:
        evaluate_point(point, slo)
    return point


# ===================================================================
# Rate search (single stage)
# ===================================================================

def run_single_rate(
    case: Case,
    connection: dict[str, Any],
    *,
    rate: float,
    duration: float,
    stage: str,
    session_dir: Path,
    seed_registry: SeedRegistry,
    logger: MasterLogger,
    idle_timeout: Optional[float],
    gpu_count: int,
    pricing: dict[str, float],
    slo: dict[str, float],
) -> PointResult:
    """Run a single rate test point and return the result."""
    seed = seed_registry.next()
    run_dir = session_dir / "runs" / f"{case.label}_{stage}_rate{rate:g}"
    outputs_dir = run_dir / "evalscope_output"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(
        connection, case,
        rate=rate, duration=duration, seed=seed,
        outputs_dir=outputs_dir.resolve(),
    )
    (run_dir / "command.txt").write_text(shell_join(command) + "\n", encoding="utf-8")

    logger.write(
        f"START POINT case={case.label} stage={stage} rate={rate:g} "
        f"duration={duration:.0f}s seed={seed}"
    )
    logger.write(f"COMMAND {shell_join(command)}")

    exit_code = run_and_tee(command, run_dir / "console.log", idle_timeout=idle_timeout)

    # Parse results
    point = parse_stage_artifacts(outputs_dir, stage, case, rate, gpu_count, pricing, slo)

    # Cleanup heavy artifacts
    for heavy in ("benchmark_data.db", "workload_timeline.json"):
        for path in outputs_dir.rglob(heavy):
            path.unlink(missing_ok=True)

    if point is None:
        point = PointResult(
            case_label=case.label, stage=stage, rate=rate,
            failure_reasons=[f"no results parsed (exit_code={exit_code})"],
        )

    status = "PASS" if point.passed else "FAIL"
    p50_ttft = format_s(point.p50_ttft_seconds)
    p99_ttft = format_s(point.p99_ttft_seconds)
    p90_tpot = format_ms(point.p90_tpot_ms)
    actual = f"{point.actual_rate:.1f}" if point.actual_rate else "N/A"
    conc = f"{point.avg_concurrency:.0f}" if point.avg_concurrency else "N/A"
    logger.write(
        f"POINT {status} case={case.label} stage={stage} rate={rate:g} "
        f"actual_rate={actual} avg_concurrency={conc} "
        f"P50_TTFT={p50_ttft}s P99_TTFT={p99_ttft}s "
        f"P90_TPOT={p90_tpot}ms reasons={'; '.join(point.failure_reasons)}"
    )
    return point


def search_max_rate(
    case: Case,
    connection: dict[str, Any],
    *,
    duration: float,
    stage: str,
    rate_start: float,
    rate_max: float,
    rate_step: float,
    session_dir: Path,
    seed_registry: SeedRegistry,
    logger: MasterLogger,
    idle_timeout: Optional[float],
    gpu_count: int,
    pricing: dict[str, float],
    slo: dict[str, float],
) -> tuple[Optional[float], list[PointResult]]:
    """Search for max arrival rate satisfying SLO.

    Algorithm:
      1. Exponential probing: start at rate_start, double until FAIL
      2. Binary search: between last PASS and first FAIL
      3. Returns (max_passing_rate, all_tested_points)
    """
    points: list[PointResult] = []

    def probe(r: float) -> PointResult:
        return run_single_rate(
            case, connection,
            rate=r, duration=duration, stage=stage,
            session_dir=session_dir, seed_registry=seed_registry,
            logger=logger, idle_timeout=idle_timeout,
            gpu_count=gpu_count, pricing=pricing, slo=slo,
        )

    # Phase 1: Exponential probing to find bounds
    logger.write(f"PHASE 1 case={case.label} exponential probing from rate={rate_start:g}")
    rate = rate_start
    last_pass: Optional[float] = None
    first_fail: Optional[float] = None

    while rate <= rate_max:
        point = probe(rate)
        points.append(point)
        if point.passed:
            last_pass = rate
            next_rate = rate * 2
            if next_rate > rate_max:
                next_rate = rate_max
                if next_rate == rate:
                    break
            rate = next_rate
        else:
            first_fail = rate
            break

    if last_pass is None:
        logger.write(f"NO_PASS case={case.label} rate={rate_start:g} already fails")
        return None, points

    if first_fail is None:
        logger.write(f"REACHED_MAX case={case.label} all rates up to {last_pass:g} pass")
        return last_pass, points

    # Phase 2: Binary search between last_pass and first_fail
    logger.write(
        f"PHASE 2 case={case.label} binary search in [{last_pass:g}, {first_fail:g}]"
    )
    lower = last_pass
    upper = first_fail
    while upper - lower > rate_step:
        mid = (lower + upper) / 2
        # Round to rate_step precision
        mid = round(mid / rate_step) * rate_step
        if mid == lower or mid == upper:
            break
        point = probe(mid)
        points.append(point)
        if point.passed:
            lower = mid
        else:
            upper = mid

    logger.write(f"SEARCH_DONE case={case.label} max_rate={lower:g}")
    return lower, points


# ===================================================================
# Output formatting
# ===================================================================

RESULT_HEADERS = [
    "输入长度", "到达 req/s", "平均并发数",
    "每卡总吞吐 tok/s", "其中每卡输出 tok/s",
    "TTFT p50 (s)", "TTFT p99 (s)", "TPOT p50 (ms)", "TPOT p90 (ms)",
    "单用户生成速度 (tok/s)", "单机月收入 （万元）",
]

DETAIL_HEADERS = [
    "case", "stage", "rate", "actual_rate", "avg_concurrency", "duration_s",
    "total_req", "success_req", "failed_req", "success_rate",
    "avg_input_tokens", "avg_output_tokens",
    "req_throughput_rps", "output_tps", "total_tps",
    "avg_latency_s", "avg_ttft_s", "p50_ttft_s", "p90_ttft_s", "p99_ttft_s",
    "avg_tpot_ms", "p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
    "per_gpu_total_tps", "per_gpu_output_tps",
    "single_user_speed_tps", "monthly_revenue_wan",
    "passed", "failure_reasons",
]

CSV_FIELDS = [
    "case_label", "stage", "rate", "actual_rate", "avg_concurrency", "duration_seconds",
    "total_requests", "success_requests", "failed_requests", "success_rate",
    "avg_input_tokens", "avg_output_tokens",
    "request_throughput_rps", "output_token_throughput_tps", "total_token_throughput_tps",
    "avg_latency_seconds", "avg_ttft_seconds", "p50_ttft_seconds", "p90_ttft_seconds", "p99_ttft_seconds",
    "avg_tpot_ms", "p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
    "per_gpu_total_tps", "per_gpu_output_tps",
    "single_user_speed_tps", "monthly_revenue_wan",
    "passed", "failure_reasons", "run_directories",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_result_table(results: list[CaseResult]) -> str:
    """Format results as PDF Table 7 style markdown table."""
    lines = [
        "| " + " | ".join(RESULT_HEADERS) + " |",
        "| " + " | ".join("---" for _ in RESULT_HEADERS) + " |",
    ]
    for r in results:
        p = r.final_point
        if p is None:
            lines.append(
                f"| {r.case_label} | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {r.case_label} "
            f"| {_fmt(p.actual_rate, '.0f')} "
            f"| {_fmt(p.avg_concurrency, '.0f')} "
            f"| {_fmt(p.per_gpu_total_tps, ',.0f')} "
            f"| {_fmt(p.per_gpu_output_tps, ',.0f')} "
            f"| {_fmt(p.p50_ttft_seconds, '.2f')} "
            f"| {_fmt(p.p99_ttft_seconds, '.2f')} "
            f"| {_fmt(p.p50_tpot_ms, '.2f')} "
            f"| {_fmt(p.p90_tpot_ms, '.2f')} "
            f"| {_fmt(p.single_user_speed_tps, '.0f')} "
            f"| {_fmt(p.monthly_revenue_wan, '.1f')} |"
        )
    return "\n".join(lines)


def format_detail_table(points: list[PointResult]) -> str:
    """Format all test points as detailed markdown table."""
    lines = [
        "| " + " | ".join(DETAIL_HEADERS) + " |",
        "| " + " | ".join("---" for _ in DETAIL_HEADERS) + " |",
    ]
    for p in points:
        lines.append(
            f"| {p.case_label} "
            f"| {p.stage} "
            f"| {p.rate:g} "
            f"| {_fmt(p.actual_rate, '.1f')} "
            f"| {_fmt(p.avg_concurrency, '.0f')} "
            f"| {_fmt(p.duration_seconds, '.1f')} "
            f"| {_fmt(p.total_requests)} "
            f"| {_fmt(p.success_requests)} "
            f"| {_fmt(p.failed_requests)} "
            f"| {_fmt(p.success_rate, '.2%')} "
            f"| {_fmt(p.avg_input_tokens, '.0f')} "
            f"| {_fmt(p.avg_output_tokens, '.0f')} "
            f"| {_fmt(p.request_throughput_rps, '.2f')} "
            f"| {_fmt(p.output_token_throughput_tps, '.0f')} "
            f"| {_fmt(p.total_token_throughput_tps, '.0f')} "
            f"| {format_s(p.avg_latency_seconds)} "
            f"| {format_s(p.avg_ttft_seconds)} "
            f"| {format_s(p.p50_ttft_seconds)} "
            f"| {format_s(p.p90_ttft_seconds)} "
            f"| {format_s(p.p99_ttft_seconds)} "
            f"| {format_ms(p.avg_tpot_ms)} "
            f"| {format_ms(p.p50_tpot_ms)} "
            f"| {format_ms(p.p90_tpot_ms)} "
            f"| {format_ms(p.p99_tpot_ms)} "
            f"| {_fmt(p.per_gpu_total_tps, ',.0f')} "
            f"| {_fmt(p.per_gpu_output_tps, ',.0f')} "
            f"| {_fmt(p.single_user_speed_tps, '.0f')} "
            f"| {_fmt(p.monthly_revenue_wan, '.1f')} "
            f"| {'PASS' if p.passed else 'FAIL'} "
            f"| {'; '.join(p.failure_reasons)} |"
        )
    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Long-context full-load SLO test: max sustainable arrival rate search (open-loop Poisson)"
    )
    # Connection
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--url")
    parser.add_argument("--api")
    parser.add_argument("--api-key", dest="api_key", help="API key for the inference service")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer-path", dest="tokenizer_path")
    parser.add_argument("--evalscope-bin", dest="evalscope_bin")
    parser.add_argument("--warmup-num", type=int, dest="warmup_num")
    parser.add_argument("--gpu-count", type=int, dest="gpu_count",
                        help="Total GPU count for per-GPU throughput calculation")
    parser.add_argument(
        "--evalscope-extra", action="append", default=[],
        help="Additional evalscope arguments as one quoted string; repeat as needed",
    )
    # Test cases
    parser.add_argument(
        "--scenario", action="append",
        help="INPUT:OUTPUT[:CACHE_RATIO], e.g. 64K:600 or 128K:600:0.9; repeatable",
    )
    # SLO
    parser.add_argument("--slo-p50-ttft", type=float, dest="slo_p50_ttft",
                        help="TTFT p50 threshold in ms (default 2500)")
    parser.add_argument("--slo-p99-ttft", type=float, dest="slo_p99_ttft",
                        help="TTFT p99 threshold in ms (default 5000)")
    parser.add_argument("--slo-p90-tpot", type=float, dest="slo_p90_tpot",
                        help="TPOT p90 threshold in ms (default 20)")
    # Search parameters
    parser.add_argument("--pre-duration", type=int, dest="pre_duration",
                        help="Pre-stage duration in seconds (default 120)")
    parser.add_argument("--formal-duration", type=int, dest="formal_duration",
                        help="Formal-stage duration in seconds (default 600)")
    parser.add_argument("--pre-rate-start", type=float, dest="pre_rate_start",
                        help="Initial rate for pre-stage search (default 1)")
    parser.add_argument("--pre-rate-max", type=float, dest="pre_rate_max",
                        help="Max rate for pre-stage search (default 256)")
    parser.add_argument("--rate-step", type=float, dest="rate_step",
                        help="Minimum rate step for binary search (default 1)")
    # Execution
    parser.add_argument("--log-root", default="longcontext_slo_logs", dest="log_root")
    parser.add_argument(
        "--run-idle-timeout", type=float, default=600.0, dest="run_idle_timeout",
        help="Seconds without evalscope output before killing (0 disables, default 600)",
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--skip-pre", action="store_true", dest="skip_pre",
        help="Skip pre-stage, use formal-duration for all search points",
    )
    parser.add_argument(
        "--fixed-rate", type=float, default=None, dest="fixed_rate",
        help="Run at a fixed rate instead of searching (skip binary search)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        connection = connection_from(args, config)
        slo = slo_from(args, config)
        pricing = pricing_from(args, config)
        search = search_from(args, config)
        cases = cases_from(args, config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    idle_timeout = args.run_idle_timeout if args.run_idle_timeout and args.run_idle_timeout > 0 else None
    log_root = Path(args.log_root).expanduser().resolve()
    session_dir = log_root / f"longcontext_{safe_stamp()}_{os.getpid()}"
    gpu_count = connection["gpu_count"]

    # Dry run
    if args.dry_run:
        for case in cases:
            print(f"# case {case.label} input={case.input_tokens} output={case.output_tokens} "
                  f"cache_ratio={case.cache_ratio}")
            if args.fixed_rate is not None:
                cmd = build_command(
                    connection, case,
                    rate=args.fixed_rate,
                    duration=search["formal_duration"],
                    seed=0,
                    outputs_dir=Path("<session>/runs/.../evalscope_output"),
                )
                print(f"FIXED  {shell_join(cmd)}")
            else:
                pre_cmd = build_command(
                    connection, case,
                    rate=search["pre_rate_start"],
                    duration=search["pre_duration"],
                    seed=0,
                    outputs_dir=Path("<session>/runs/.../evalscope_output"),
                )
                formal_cmd = build_command(
                    connection, case,
                    rate=search["pre_rate_start"],
                    duration=search["formal_duration"],
                    seed=0,
                    outputs_dir=Path("<session>/runs/.../evalscope_output"),
                )
                print(f"PRE    {shell_join(pre_cmd)}")
                print(f"FORMAL {shell_join(formal_cmd)}")
            print()
        return 0

    # Create session
    session_dir.mkdir(parents=True, exist_ok=False)
    logger = MasterLogger(session_dir / "master.log")
    registry = SeedRegistry(log_root / "seed_history.txt")
    write_json(session_dir / "resolved_config.json", {
        "connection": connection,
        "slo": slo,
        "pricing": pricing,
        "search": search,
        "gpu_count": gpu_count,
        "cases": [asdict(c) for c in cases],
    })
    logger.write(f"SESSION_DIR {session_dir}")
    logger.write(f"SLO {json.dumps(slo, sort_keys=True)}")
    logger.write(f"GPU_COUNT {gpu_count}")

    all_points: list[PointResult] = []
    case_results: list[CaseResult] = []

    try:
        for case in cases:
            logger.write(
                f"START CASE {case.label} input={case.input_tokens} "
                f"output={case.output_tokens} cache_ratio={case.cache_ratio}"
            )
            case_result = CaseResult(
                case_label=case.label,
                input_tokens=case.input_tokens,
                output_tokens=case.output_tokens,
                cache_ratio=case.cache_ratio,
            )

            if args.fixed_rate is not None:
                # Fixed rate mode: just run at the specified rate
                point = run_single_rate(
                    case, connection,
                    rate=args.fixed_rate,
                    duration=search["formal_duration"],
                    stage="formal",
                    session_dir=session_dir, seed_registry=registry,
                    logger=logger, idle_timeout=idle_timeout,
                    gpu_count=gpu_count, pricing=pricing, slo=slo,
                )
                case_result.formal_points.append(point)
                case_result.formal_max_rate = args.fixed_rate if point.passed else None
                case_result.final_max_rate = case_result.formal_max_rate
                case_result.final_point = point if point.passed else None
                all_points.append(point)
            else:
                # Pre stage: coarse search with shorter duration
                if not args.skip_pre:
                    pre_max, pre_points = search_max_rate(
                        case, connection,
                        duration=search["pre_duration"],
                        stage="pre",
                        rate_start=search["pre_rate_start"],
                        rate_max=search["pre_rate_max"],
                        rate_step=search["rate_step"],
                        session_dir=session_dir, seed_registry=registry,
                        logger=logger, idle_timeout=idle_timeout,
                        gpu_count=gpu_count, pricing=pricing, slo=slo,
                    )
                    case_result.pre_points.extend(pre_points)
                    case_result.pre_max_rate = pre_max
                    all_points.extend(pre_points)
                else:
                    pre_max = None

                # Formal stage: verify with full duration
                if pre_max is not None or args.skip_pre:
                    if args.skip_pre:
                        # Without pre-stage, search with full duration
                        formal_max, formal_points = search_max_rate(
                            case, connection,
                            duration=search["formal_duration"],
                            stage="formal",
                            rate_start=search["pre_rate_start"],
                            rate_max=search["pre_rate_max"],
                            rate_step=search["rate_step"],
                            session_dir=session_dir, seed_registry=registry,
                            logger=logger, idle_timeout=idle_timeout,
                            gpu_count=gpu_count, pricing=pricing, slo=slo,
                        )
                        case_result.formal_points.extend(formal_points)
                        case_result.formal_max_rate = formal_max
                        case_result.final_max_rate = formal_max
                        all_points.extend(formal_points)
                    else:
                        # Verify pre-stage result with full duration
                        assert pre_max is not None
                        formal_point = run_single_rate(
                            case, connection,
                            rate=pre_max,
                            duration=search["formal_duration"],
                            stage="formal",
                            session_dir=session_dir, seed_registry=registry,
                            logger=logger, idle_timeout=idle_timeout,
                            gpu_count=gpu_count, pricing=pricing, slo=slo,
                        )
                        case_result.formal_points.append(formal_point)
                        all_points.append(formal_point)

                        if formal_point.passed:
                            case_result.formal_max_rate = pre_max
                            case_result.final_max_rate = pre_max
                            case_result.final_point = formal_point
                        else:
                            # Pre-stage rate failed formal verification, try lower
                            logger.write(
                                f"FORMAL_FAIL case={case.label} pre_rate={pre_max:g} "
                                f"fails at full duration, retrying lower"
                            )
                            retry_rate = pre_max - search["rate_step"]
                            if retry_rate >= search["rate_step"]:
                                retry_point = run_single_rate(
                                    case, connection,
                                    rate=retry_rate,
                                    duration=search["formal_duration"],
                                    stage="formal",
                                    session_dir=session_dir, seed_registry=registry,
                                    logger=logger, idle_timeout=idle_timeout,
                                    gpu_count=gpu_count, pricing=pricing, slo=slo,
                                )
                                case_result.formal_points.append(retry_point)
                                all_points.append(retry_point)
                                if retry_point.passed:
                                    case_result.formal_max_rate = retry_rate
                                    case_result.final_max_rate = retry_rate
                                    case_result.final_point = retry_point
                            else:
                                case_result.formal_max_rate = None
                                case_result.final_max_rate = None

                # Set final point if not already set
                if case_result.final_point is None and case_result.formal_points:
                    passing = [p for p in case_result.formal_points if p.passed]
                    if passing:
                        case_result.final_point = passing[-1]
                        if case_result.final_max_rate is None:
                            case_result.final_max_rate = case_result.final_point.rate

            case_results.append(case_result)
            logger.write(
                f"END CASE {case.label} pre_max={case_result.pre_max_rate} "
                f"formal_max={case_result.formal_max_rate} "
                f"final_max_rate={case_result.final_max_rate}"
            )
    except KeyboardInterrupt:
        logger.write("INTERRUPTED by user; completed run artifacts preserved")
        return 130

    # Write results
    write_json(session_dir / "all_points.json", [asdict(p) for p in all_points])
    with (session_dir / "all_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in all_points:
            row = asdict(point)
            row["failure_reasons"] = " | ".join(point.failure_reasons)
            row["run_directories"] = " | ".join(point.run_directories)
            writer.writerow(row)

    # Case results
    write_json(session_dir / "case_results.json", [
        {k: v for k, v in asdict(r).items() if k not in ("pre_points", "formal_points", "final_point")}
        | {"final_point": asdict(r.final_point) if r.final_point else None}
        for r in case_results
    ])

    # Summary tables
    result_table = format_result_table(case_results)
    detail_table = format_detail_table(all_points)

    summary_md = (
        "# 长上下文满载 SLO 测试结果\n\n"
        f"测试时间: {safe_stamp()}\n"
        f"SLO: TTFT p50 <= {slo['p50_ttft_ms']:.0f}ms, "
        f"TTFT p99 <= {slo['p99_ttft_ms']:.0f}ms, "
        f"TPOT p90 <= {slo['p90_tpot_ms']:.1f}ms\n"
        f"GPU 数量: {gpu_count}\n\n"
        "## 结果汇总\n\n"
        + result_table + "\n\n"
        "## 全部测试点\n\n"
        + detail_table + "\n"
    )
    (session_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    # Print results
    print()
    print("=" * 80)
    print("长上下文满载 SLO 测试结果")
    print("=" * 80)
    print()
    print(result_table)
    print()

    for r in case_results:
        parts = [
            f"RESULT case={r.case_label}",
            f"pre_max={r.pre_max_rate}",
            f"formal_max={r.formal_max_rate}",
            f"final_rate={r.final_max_rate}",
        ]
        if r.final_point:
            p = r.final_point
            parts.extend([
                f"avg_concurrency={p.avg_concurrency:.0f}" if p.avg_concurrency else "avg_concurrency=N/A",
                f"per_gpu_tps={p.per_gpu_total_tps:,.0f}" if p.per_gpu_total_tps else "per_gpu_tps=N/A",
                f"revenue={p.monthly_revenue_wan:.1f}万" if p.monthly_revenue_wan else "revenue=N/A",
            ])
        logger.write(" ".join(parts))

    logger.write(f"COMPLETE results={session_dir}")
    print(f"\nResults: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
