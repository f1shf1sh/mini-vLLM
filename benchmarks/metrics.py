"""Recompute benchmark statistics from raw request events and a measured interval."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


SCHEMA_VERSION = 1
TERMINAL = {"FINISHED", "FAILED", "CANCELLED"}
STATUSES = TERMINAL | {"NOT_SUBMITTED", "WAITING", "PREFILLING", "DECODING"}


def percentile(values: list[float], quantile: float) -> float | None:
    """Linear interpolation at (n - 1) * quantile; empty samples return None."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_record(record: dict, start_ns: int, end_ns: int) -> None:
    """Reject inconsistent events rather than silently producing plausible metrics."""
    status = record["status"]
    if status not in STATUSES:
        raise ValueError(f"Unknown request status: {status!r}")
    input_ids, generated_ids = record["input_ids"], record["generated_ids"]
    if not isinstance(input_ids, list) or not isinstance(generated_ids, list):
        raise ValueError("input_ids and generated_ids must be lists")
    for token in input_ids + generated_ids:
        _integer(token, "token ID")
    prompt_tokens = _integer(record["prompt_tokens"], "prompt_tokens")
    output_tokens = _integer(record["output_tokens"], "output_tokens")
    maximum = _integer(record["max_new_tokens"], "max_new_tokens")
    stop_ids = record["stop_token_ids"]
    if not isinstance(stop_ids, list):
        raise ValueError("stop_token_ids must be a list")
    for token in stop_ids:
        _integer(token, "stop token ID")
    if prompt_tokens != len(input_ids) or output_tokens != len(generated_ids):
        raise ValueError("Token counts must match recorded token IDs")
    if output_tokens > maximum:
        raise ValueError("Output token count exceeds max_new_tokens")
    if any(token in stop_ids for token in generated_ids[:-1]):
        raise ValueError("A stop token cannot occur before the final output token")
    if status == "FINISHED" and output_tokens < maximum:
        if not generated_ids or generated_ids[-1] not in stop_ids:
            raise ValueError("A successful request must reach max_new_tokens or end with a stop token")
    timing = record["timing"]
    if timing is None:
        if status != "NOT_SUBMITTED" or generated_ids:
            raise ValueError("Only an unsubmitted request may have no timing")
        return
    if status == "NOT_SUBMITTED":
        raise ValueError("An unsubmitted request cannot have timing events")

    def timestamp(value, name):
        value = _integer(value, name)
        if not start_ns <= value <= end_ns:
            raise ValueError(f"{name} is outside the measured interval")
        return value

    submitted = timestamp(timing["submitted_at_ns"], "submitted_at_ns")
    admitted = timing["admitted_at_ns"]
    finished = timing["finished_at_ns"]
    tokens = timing["token_ready_at_ns"]
    if not isinstance(tokens, list) or len(tokens) != output_tokens:
        raise ValueError("Each output token must have exactly one timestamp")
    events = [submitted]
    if admitted is not None:
        events.append(timestamp(admitted, "admitted_at_ns"))
    if tokens and admitted is None:
        raise ValueError("Token events require an admission event")
    events.extend(timestamp(t, "token_ready_at_ns") for t in tokens)
    if status in TERMINAL and finished is None:
        raise ValueError("A terminal request must have a finished timestamp")
    if status not in TERMINAL and finished is not None:
        raise ValueError("An unfinished request cannot have a finished timestamp")
    if finished is not None:
        events.append(timestamp(finished, "finished_at_ns"))
    if events != sorted(events):
        raise ValueError("Request timestamps must be monotonically nondecreasing")


def summarize_requests(records: list[dict], start_ns: int, end_ns: int) -> dict:
    start_ns = _integer(start_ns, "start_ns")
    end_ns = _integer(end_ns, "end_ns")
    if end_ns <= start_ns:
        raise ValueError("The measured interval must have positive duration")
    seen = set()
    for record in records:
        request_id = record["request_id"]
        if not isinstance(request_id, str) or request_id in seen:
            raise ValueError("Request IDs must be unique strings")
        seen.add(request_id)
        validate_record(record, start_ns, end_ns)
    successful = [r for r in records if r["status"] == "FINISHED"]
    seconds = (end_ns - start_ns) / 1e9
    samples = {key: [] for key in ("queue", "ttft", "tpot_per_request", "itl", "e2e")}
    for record in successful:
        timing = record["timing"]
        submitted, admitted = timing["submitted_at_ns"], timing["admitted_at_ns"]
        tokens, finished = timing["token_ready_at_ns"], timing["finished_at_ns"]
        samples["e2e"].append((finished - submitted) / 1e6)
        if admitted is not None:
            samples["queue"].append((admitted - submitted) / 1e6)
        if tokens:
            samples["ttft"].append((tokens[0] - submitted) / 1e6)
        if len(tokens) > 1:
            samples["tpot_per_request"].append((tokens[-1] - tokens[0]) / (len(tokens) - 1) / 1e6)
            samples["itl"].extend((b - a) / 1e6 for a, b in zip(tokens, tokens[1:]))
    successful_tokens = sum(r["output_tokens"] for r in successful)
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement": {
            "clock": "time.perf_counter_ns",
            "start_ns": start_ns,
            "end_ns": end_ns,
            "elapsed_ns": end_ns - start_ns,
            "elapsed_seconds": seconds,
        },
        "counts": {
            "planned_requests": len(records),
            "submitted_requests": sum(r["timing"] is not None for r in records),
            "successful_requests": len(successful),
            "failed_requests": sum(r["status"] == "FAILED" for r in records),
            "cancelled_requests": sum(r["status"] == "CANCELLED" for r in records),
            "not_submitted_requests": sum(r["status"] == "NOT_SUBMITTED" for r in records),
            "unfinished_requests": sum(r["status"] not in TERMINAL | {"NOT_SUBMITTED"} for r in records),
            "successful_output_tokens": successful_tokens,
            "unsuccessful_output_tokens": sum(r["output_tokens"] for r in records if r["status"] != "FINISHED"),
        },
        "throughput": {
            "successful_requests_per_second": len(successful) / seconds,
            "successful_output_tokens_per_second": successful_tokens / seconds,
        },
        "latency_ms": {key: distribution(values) for key, values in samples.items()},
        "definitions": {
            "latency_population": "Successful requests only; failed/cancelled output is excluded.",
            "throughput_denominator": "Entire measured wall interval, including initial submission/ramp-up, drain, failures and final cleanup/synchronization.",
            "ttft": "first token_ready - submitted",
            "tpot_per_request": "(last token_ready - first token_ready) / (output_tokens - 1); undefined for 0/1 token",
            "itl": "All adjacent token_ready differences from successful requests (token-weighted population).",
            "e2e": "finished after cleanup - submitted",
            "percentiles": "Linear interpolation at (n - 1) * quantile; empty samples are null.",
        },
    }


def summarize_run(directory: str | Path) -> dict:
    directory = Path(directory)
    measurement = json.loads((directory / "measurement.json").read_text())
    records = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
    summary = summarize_requests(records, measurement["start_ns"], measurement["end_ns"])
    if measurement["run_status"] not in {"success", "failed"}:
        raise ValueError("Unknown run_status")
    if measurement["run_status"] == "success":
        if (any(record["status"] != "FINISHED" for record in records)
                or measurement["errors"] != []
                or measurement["cache_before"].get("fully_released") is not True
                or measurement["cache_after"].get("fully_released") is not True):
            raise ValueError("A successful run must finish every request without errors and release all cache resources")
    for key in ("run_status", "errors", "cache_before", "cache_after", "peak_outstanding"):
        summary[key] = measurement[key]
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Write a new JSON file; an existing file is never overwritten.")
    args = parser.parse_args(argv)
    result = json.dumps(summarize_run(args.run_dir), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(result)
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
