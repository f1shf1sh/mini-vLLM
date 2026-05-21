from lmcore.inference.request import Request, RequestStatus
from lmcore.inference.sampler import SamplingParams


def test_request_defaults_to_waiting() -> None:
    request = Request(
        request_id="req-1",
        input_ids=[1, 2, 3],
        max_new_tokens=2,
    )

    assert request.status is RequestStatus.WAITING
    assert request.prompt_length == 3
    assert request.num_generated_tokens == 0
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


def test_request_appends_tokens_and_finishes() -> None:
    request = Request(
        request_id="req-1",
        input_ids=[1, 2, 3],
        max_new_tokens=2,
    )

    request.mark_decoding()
    request.append_token(4)

    assert request.status is RequestStatus.DECODING
    assert request.generated_ids == [4]
    assert request.output_ids == [1, 2, 3, 4]
    assert not request.is_finished

    request.append_token(5)

    assert request.status is RequestStatus.FINISHED
    assert request.generated_ids == [4, 5]
    assert request.output_ids == [1, 2, 3, 4, 5]
    assert request.is_finished


def test_request_keeps_sampling_params() -> None:
    request = Request(
        request_id="req-1",
        input_ids=[1, 2, 3],
        max_new_tokens=2,
        sampling_params=SamplingParams(temperature=0.8, top_k=5),
    )

    assert request.sampling_params.temperature == 0.8
    assert request.sampling_params.top_k == 5