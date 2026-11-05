import pytest

from llm.inference.request import Request, RequestStatus
from llm.inference.sampler import SamplingParams


def test_request_defaults_to_waiting() -> None:
    request = Request(
        request_id="req-1",
        input_ids=[1, 2, 3],
        max_new_tokens=2,
    )

    assert request.status is RequestStatus.WAITING
    assert request.prompt_length == 3
    assert request.num_generated_tokens == 0
    assert request.num_computed_tokens == 0
    assert request.output_ids == [1, 2, 3]


def test_request_can_transition_to_prefill_and_decode() -> None:
    request = Request(
        request_id="req-1",
        input_ids=[1, 2, 3],
        max_new_tokens=2,
    )

    request.mark_prefilling()
    assert request.status is RequestStatus.PREFILLING

    request.mark_decoding()
    assert request.status is RequestStatus.DECODING


def test_request_appends_output_without_committing_progress_or_terminal_state() -> None:
    request = Request(
        request_id="req-1",
        input_ids=[1, 2, 3],
        max_new_tokens=2,
    )

    request.mark_decoding()
    request.num_computed_tokens = 3
    request.append_token(4)

    assert request.status is RequestStatus.DECODING
    assert request.generated_ids == [4]
    assert request.output_ids == [1, 2, 3, 4]
    assert not request.is_finished
    assert not request.should_stop
    assert request.num_computed_tokens == 3

    request.append_token(5)

    assert request.status is RequestStatus.DECODING
    assert request.generated_ids == [4, 5]
    assert request.output_ids == [1, 2, 3, 4, 5]
    assert request.should_stop
    assert not request.is_finished
    assert request.num_computed_tokens == 3


def test_request_keeps_sampling_params() -> None:
    request = Request(
        request_id="req-1",
        input_ids=[1, 2, 3],
        max_new_tokens=2,
        sampling_params=SamplingParams(temperature=0.8, top_k=5),
    )

    assert request.sampling_params.temperature == 0.8
    assert request.sampling_params.top_k == 5

@pytest.mark.parametrize("limit,stop_ids,output", [
    (0, (), []),
    (5, (7,), [2, 7]),
])
def test_stop_condition_does_not_assign_terminal_state(limit, stop_ids, output):
    request = Request("a", [1], limit, stop_token_ids=stop_ids)
    for token in output:
        request.append_token(token)
    assert request.should_stop
    assert not request.is_finished
    assert request.status is RequestStatus.WAITING


@pytest.mark.parametrize("status", [
    RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED,
])
def test_terminal_state_does_not_require_a_stop_condition(status):
    request = Request("a", [1], 5, status=status)
    assert request.is_finished
    assert not request.should_stop
