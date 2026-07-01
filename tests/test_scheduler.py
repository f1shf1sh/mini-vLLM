import torch

from lmcore.config import ModelConfig
from lmcore.inference.block_manager import BlockManager, BlockAllocator
from lmcore.inference.kv_pool import KVPool
from lmcore.inference.request import Request, RequestStatus
from lmcore.inference.sampler import Sampler, SamplingParams
from lmcore.inference.scheduler import Scheduler
from lmcore.model import GPTMini


def make_model() -> GPTMini:
    config = ModelConfig(
        vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=8,
        dropout=0.0, bias=False,
    )
    model = GPTMini(config)
    model.eval()
    return model


def _make_scheduler(model=None, sampler=None):
    if model is None:
        model = make_model()
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    pool = KVPool(
        num_blocks=50, num_layers=model.config.n_layer,
        num_heads=n_head, block_size=4, head_dim=head_dim,
    )
    allocator = BlockAllocator(num_blocks=50)
    bm = BlockManager(allocator=allocator, block_size=4)
    return Scheduler(model, kv_pool=pool, block_manager=bm, sampler=sampler)


# ── is_done ────────────────────────────────────────────────────────

def test_is_done_empty_scheduler():
    assert _make_scheduler().is_done is True


def test_is_done_false_when_waiting():
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    assert s.is_done is False


def test_is_done_false_when_running():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))
    s.step()
    assert s.is_done is False


# ── single-request ─────────────────────────────────────────────────

def test_single_request_completes():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))
    while not s.is_done:
        s.step()
    req = s.finished[0]
    assert req.status is RequestStatus.FINISHED
    assert req.num_generated_tokens == 3


def test_step_returns_empty_list_when_prefill_only():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))
    assert s.step() == []


def test_step_returns_request_when_decode_finishes_it():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    s.step()
    finished = s.step()
    assert len(finished) == 1
    assert finished[0].request_id == "r1"
    assert finished[0].is_finished


# ── multiple requests ──────────────────────────────────────────────

def test_multiple_requests_all_finish():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    s.add_request(Request("r2", [4, 5], max_new_tokens=2))
    s.add_request(Request("r3", [6], max_new_tokens=3))
    while not s.is_done:
        s.step()
    assert {r.request_id for r in s.finished} == {"r1", "r2", "r3"}


def test_short_request_finishes_before_long_request():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("short", [1, 2, 3], max_new_tokens=1,
                   sampling_params=SamplingParams(temperature=1.0)))
    s.add_request(Request("long", [4, 5, 6], max_new_tokens=5,
                   sampling_params=SamplingParams(temperature=1.0)))
    order = []
    while not s.is_done:
        for r in s.step():
            order.append(r.request_id)
    assert order[0] == "short"
    assert order[1] == "long"


# ── determinism ────────────────────────────────────────────────────

def test_scheduler_is_deterministic():
    def run():
        torch.manual_seed(99)
        s = _make_scheduler()
        s.add_request(Request("a", [1, 2], max_new_tokens=3))
        s.add_request(Request("b", [3, 4], max_new_tokens=2))
        while not s.is_done:
            s.step()
        return [r.output_ids for r in s.finished]
    assert run() == run()


# ── step semantics ─────────────────────────────────────────────────

def test_step_does_exactly_one_operation():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=5))
    s.step()
    assert s.running[0].num_generated_tokens == 1
    s.step()
    assert s.running[0].num_generated_tokens == 2
    s.step()
    assert s.running[0].num_generated_tokens == 3


def test_finished_request_not_processed_again():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("short", [1, 2, 3], max_new_tokens=1))
    s.add_request(Request("long", [4, 5, 6], max_new_tokens=5))
    while not s.is_done:
        s.step()
    short = [r for r in s.finished if r.request_id == "short"][0]
    assert short.num_generated_tokens == 1
    assert len(short.generated_ids) == 1


# ── state transitions ──────────────────────────────────────────────

def test_request_transitions_waiting_to_prefilling_to_decoding_to_finished():
    torch.manual_seed(0)
    s = _make_scheduler()
    req = Request("r1", [1, 2, 3], max_new_tokens=2)
    assert req.status is RequestStatus.WAITING
    s.add_request(req)
    s.step()
    assert req.status is RequestStatus.DECODING
    assert req.num_generated_tokens == 1
    s.step()
    assert req.status is RequestStatus.FINISHED


# ── cleanup ────────────────────────────────────────────────────────

def test_block_manager_cleaned_after_finish():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=1))
    while not s.is_done:
        s.step()
    assert "r1" not in s.block_manager.block_tables


# ── FIFO / round-robin ─────────────────────────────────────────────

def test_waiting_is_fifo():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("first", [1, 2], max_new_tokens=1))
    s.add_request(Request("second", [3, 4], max_new_tokens=1))
    order = []
    while not s.is_done:
        for r in s.step():
            order.append(r.request_id)
    assert order == ["first", "second"]


def test_running_decode_interleaves_round_robin():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("a", [1, 2], max_new_tokens=3))
    s.add_request(Request("b", [3, 4], max_new_tokens=3))
    s.step(); s.step()
    assert {r.request_id for r in s.running} == {"a", "b"}
    s.step()
    by_id = {r.request_id: r for r in s.running}
    assert by_id["a"].num_generated_tokens == 2
    assert by_id["b"].num_generated_tokens == 1
    s.step()
    by_id = {r.request_id: r for r in s.running}
    assert by_id["a"].num_generated_tokens == 2
    assert by_id["b"].num_generated_tokens == 2


# ── edge cases ─────────────────────────────────────────────────────

def test_max_new_tokens_zero_generates_nothing():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r", [1, 2, 3], max_new_tokens=0))
    finished = s.step()
    assert len(finished) == 1
    assert finished[0].num_generated_tokens == 0
    assert finished[0].output_ids == [1, 2, 3]


def test_duplicate_request_id_rejected():
    import pytest
    s = _make_scheduler()
    s.add_request(Request("dup", [1, 2], max_new_tokens=1))
    with pytest.raises(ValueError, match="Duplicate request_id"):
        s.add_request(Request("dup", [3, 4], max_new_tokens=1))


# ── per-request sampling params ────────────────────────────────────

def test_per_request_sampling_params_used():
    params = SamplingParams(temperature=1.0, top_k=1)
    torch.manual_seed(42)
    s = _make_scheduler(sampler=Sampler(SamplingParams(temperature=2.0)))
    s.add_request(Request("r", [1, 2, 3], max_new_tokens=3,
                   sampling_params=params))
    while not s.is_done:
        s.step()
    assert len(s.finished[0].output_ids) == 6
