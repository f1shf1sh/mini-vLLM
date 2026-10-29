import copy
import json

import pytest

from benchmarks.metrics import percentile, summarize_requests, summarize_run


NS = 1_000_000_000


def record(name="r", tokens=(0.2, 0.4, 0.8), status="FINISHED", submitted=0.1, admitted=0.15, finished=0.9):
    return {
        "request_id": name,
        "status": status,
        "input_ids": [4, 5],
        "generated_ids": [6] * len(tokens),
        "prompt_tokens": 2,
        "output_tokens": len(tokens),
        "max_new_tokens": len(tokens) if status == "FINISHED" else 8,
        "stop_token_ids": [],
        "timing": {
            "submitted_at_ns": int(submitted * NS),
            "admitted_at_ns": None if admitted is None else int(admitted * NS),
            "token_ready_at_ns": [int(value * NS) for value in tokens],
            "finished_at_ns": None if finished is None else int(finished * NS),
        },
    }


def test_known_latency_and_throughput_values():
    summary = summarize_requests([record()], 0, NS)
    latency = summary["latency_ms"]
    assert latency["queue"]["mean"] == 50
    assert latency["ttft"]["mean"] == 100
    assert latency["tpot_per_request"]["mean"] == 300
    assert latency["itl"]["count"] == 2
    assert latency["itl"]["p95"] == 390
    assert latency["e2e"]["mean"] == 800
    assert summary["throughput"] == {
        "successful_requests_per_second": 1,
        "successful_output_tokens_per_second": 3,
    }


def test_empty_workload_and_zero_token_completion_have_null_undefined_metrics():
    empty = summarize_requests([], 0, NS)
    assert empty["counts"]["successful_requests"] == 0
    assert empty["latency_ms"]["e2e"]["p99"] is None
    zero = summarize_requests([record(tokens=(), admitted=None)], 0, NS)
    assert zero["counts"]["successful_requests"] == 1
    assert zero["throughput"]["successful_output_tokens_per_second"] == 0
    for key in ("queue", "ttft", "tpot_per_request", "itl"):
        assert zero["latency_ms"][key]["count"] == 0
        assert zero["latency_ms"][key]["mean"] is None
    assert zero["latency_ms"]["e2e"]["mean"] == 800


def test_single_token_has_ttft_but_no_tpot_or_itl():
    summary = summarize_requests([record(tokens=(0.2,))], 0, NS)
    assert summary["latency_ms"]["ttft"]["mean"] == 100
    assert summary["latency_ms"]["tpot_per_request"]["p50"] is None
    assert summary["latency_ms"]["itl"]["mean"] is None


def test_failures_are_excluded_from_numerators_but_not_wall_time():
    records = [
        record("ok"),
        record("failed", status="FAILED", tokens=(0.25, 0.3)),
        record("cancelled", status="CANCELLED", tokens=(), admitted=None),
    ]
    unsubmitted = record("later", tokens=(), status="NOT_SUBMITTED")
    unsubmitted["timing"] = None
    records.append(unsubmitted)
    summary = summarize_requests(records, 0, 2 * NS)
    assert summary["counts"] == {
        "planned_requests": 4,
        "submitted_requests": 3,
        "successful_requests": 1,
        "failed_requests": 1,
        "cancelled_requests": 1,
        "not_submitted_requests": 1,
        "unfinished_requests": 0,
        "successful_output_tokens": 3,
        "unsuccessful_output_tokens": 2,
    }
    assert summary["throughput"]["successful_requests_per_second"] == 0.5
    assert summary["throughput"]["successful_output_tokens_per_second"] == 1.5
    assert summary["latency_ms"]["e2e"]["count"] == 1


def test_tpot_population_is_request_weighted_and_itl_is_token_weighted():
    summary = summarize_requests([
        record("long", tokens=(0.2, 0.4, 0.8)),
        record("short", tokens=(0.2, 0.3)),
    ], 0, NS)
    assert summary["latency_ms"]["tpot_per_request"]["mean"] == 200
    assert summary["latency_ms"]["itl"]["mean"] == pytest.approx(700 / 3)


def test_percentiles_use_documented_linear_interpolation():
    assert percentile([], 0.95) is None
    assert percentile([7], 0.99) == 7
    assert percentile([100, 0, 20], 0.50) == 20
    assert percentile([100, 0, 20], 0.95) == pytest.approx(92)
    with pytest.raises(ValueError):
        percentile([1], 1.1)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(output_tokens=2),
    lambda r: r.update(prompt_tokens=3),
    lambda r: r.update(max_new_tokens=1),
    lambda r: r["timing"].update(token_ready_at_ns=[200_000_000]),
    lambda r: r["timing"].update(token_ready_at_ns=[400_000_000, 200_000_000, 800_000_000]),
    lambda r: r["timing"].update(admitted_at_ns=None),
    lambda r: r["timing"].update(admitted_at_ns=50_000_000),
    lambda r: r["timing"].update(finished_at_ns=500_000_000),
    lambda r: r["timing"].update(finished_at_ns=None),
    lambda r: r["timing"].update(submitted_at_ns=-1),
    lambda r: r["timing"].update(finished_at_ns=2 * NS),
    lambda r: r["timing"].update(submitted_at_ns=True),
    lambda r: r.update(status="UNKNOWN"),
    lambda r: r.update(timing=None),
])
def test_inconsistent_raw_records_are_rejected(mutation):
    value = record()
    mutation(value)
    with pytest.raises(ValueError):
        summarize_requests([value], 0, NS)


def test_duplicate_ids_and_invalid_intervals_are_rejected():
    value = record()
    with pytest.raises(ValueError, match="unique"):
        summarize_requests([value, copy.deepcopy(value)], 0, NS)
    with pytest.raises(ValueError, match="positive duration"):
        summarize_requests([], NS, NS)


def test_raw_run_recomputes_exactly_and_preserves_failure_information(tmp_path):
    measurement = {
        "start_ns": 0, "end_ns": NS, "run_status": "failed",
        "errors": [{"stage": "execution", "message": "intentional"}],
        "cache_before": {"fully_released": True},
        "cache_after": {"fully_released": True}, "peak_outstanding": 1,
    }
    value = record(status="FAILED")
    (tmp_path / "measurement.json").write_text(json.dumps(measurement))
    (tmp_path / "requests.jsonl").write_text(json.dumps(value) + "\n")
    expected = summarize_requests([value], 0, NS)
    expected.update({key: measurement[key] for key in ("run_status", "errors", "cache_before", "cache_after", "peak_outstanding")})
    assert summarize_run(tmp_path) == expected


def test_early_success_requires_final_eos_and_eos_cannot_be_nonfinal():
    value = record(tokens=(0.2,))
    value["max_new_tokens"] = 4
    with pytest.raises(ValueError, match="reach max_new_tokens"):
        summarize_requests([value], 0, NS)
    value["stop_token_ids"] = [6]
    assert summarize_requests([value], 0, NS)["counts"]["successful_requests"] == 1
    value = record(tokens=(0.2, 0.3))
    value["stop_token_ids"] = [6]
    with pytest.raises(ValueError, match="before the final"):
        summarize_requests([value], 0, NS)
    value = record()
    value["stop_token_ids"] = [True]
    with pytest.raises(ValueError, match="stop token ID"):
        summarize_requests([value], 0, NS)


@pytest.mark.parametrize("fault", ["request", "errors", "cache_before", "cache_after"])
def test_successful_run_rejects_inconsistent_completion_and_resources(tmp_path, fault):
    value = record(status="FAILED" if fault == "request" else "FINISHED")
    measurement = {
        "start_ns": 0, "end_ns": NS, "run_status": "success", "errors": [],
        "cache_before": {"fully_released": True},
        "cache_after": {"fully_released": True}, "peak_outstanding": 1,
    }
    if fault == "errors":
        measurement["errors"].append({"message": "failure"})
    elif fault in {"cache_before", "cache_after"}:
        measurement[fault]["fully_released"] = False
    (tmp_path / "measurement.json").write_text(json.dumps(measurement))
    (tmp_path / "requests.jsonl").write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="successful run"):
        summarize_run(tmp_path)
