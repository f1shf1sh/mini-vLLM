import torch

from lmcore.config import ModelConfig
from lmcore.inference.generator import CachedGenerator
from lmcore.inference.request import Request, RequestStatus
from lmcore.inference.sampler import Sampler, SamplingParams
from lmcore.inference.scheduler import Scheduler
from lmcore.model import GPTMini


def make_model() -> GPTMini:
    config = ModelConfig(
        vocab_size=16,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    model = GPTMini(config)
    model.eval()
    return model


# ── is_done ────────────────────────────────────────────────────────────

def test_is_done_empty_scheduler() -> None:
    """新创建的 scheduler 没有请求，is_done 应为 True。"""
    scheduler = Scheduler(make_model())
    assert scheduler.is_done is True


def test_is_done_false_when_waiting() -> None:
    """add_request 后 is_done 应为 False。"""
    scheduler = Scheduler(make_model())
    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    assert scheduler.is_done is False


def test_is_done_false_when_running() -> None:
    """prefill 完成后请求进入 running，is_done 应为 False。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())
    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))
    scheduler.step()  # prefill → running
    assert scheduler.is_done is False


# ── 单请求完整性 ──────────────────────────────────────────────────────

def test_single_request_completes() -> None:
    """一个请求从头跑到尾，状态变为 FINISHED，生成了正确数量的 token。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())
    max_tokens = 3
    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=max_tokens))

    while not scheduler.is_done:
        scheduler.step()

    assert len(scheduler.finished) == 1
    req = scheduler.finished[0]
    assert req.status is RequestStatus.FINISHED
    assert req.num_generated_tokens == max_tokens
    assert len(req.output_ids) == req.prompt_length + max_tokens


def test_single_request_matches_cached_generator() -> None:
    """Scheduler 和 CachedGenerator 用相同 seed/model/输入，输出应完全一致。"""
    torch.manual_seed(42)
    model = make_model()
    prompt_ids = [1, 2, 3]
    max_tokens = 5

    # CachedGenerator 作为基准
    torch.manual_seed(42)
    gen_model = make_model()
    gen = CachedGenerator(gen_model)
    gen_output = gen.generate(
        torch.tensor([prompt_ids], dtype=torch.long),
        max_new_tokens=max_tokens,
    )
    expected = gen_output[0].tolist()

    # Scheduler
    torch.manual_seed(42)
    sched_model = make_model()
    scheduler = Scheduler(sched_model)
    scheduler.add_request(Request("r1", prompt_ids, max_new_tokens=max_tokens))

    while not scheduler.is_done:
        scheduler.step()

    actual = scheduler.finished[0].output_ids
    assert actual == expected, (
        f"Scheduler output differs from CachedGenerator:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )


# ── step() 返回值 ─────────────────────────────────────────────────────

def test_step_returns_empty_list_when_prefill_only() -> None:
    """prefill 不产生完成请求，step() 应返回空列表。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())
    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))

    finished = scheduler.step()
    assert finished == []


def test_step_returns_request_when_decode_finishes_it() -> None:
    """最后一个 decode step 完成后，step() 应返回包含该请求的列表。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())
    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))

    scheduler.step()             # prefill → 生成第 1 个 token，未完成
    finished = scheduler.step()  # decode → 生成第 2 个 token，完成

    assert len(finished) == 1
    assert finished[0].request_id == "r1"
    assert finished[0].is_finished


# ── 多请求 ─────────────────────────────────────────────────────────────

def test_multiple_requests_all_finish() -> None:
    """多个请求全部最终完成。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())

    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    scheduler.add_request(Request("r2", [4, 5], max_new_tokens=2))
    scheduler.add_request(Request("r3", [6], max_new_tokens=3))

    while not scheduler.is_done:
        scheduler.step()

    assert len(scheduler.finished) == 3
    finished_ids = {req.request_id for req in scheduler.finished}
    assert finished_ids == {"r1", "r2", "r3"}
    for req in scheduler.finished:
        assert req.status is RequestStatus.FINISHED


def test_short_request_finishes_before_long_request() -> None:
    """max_new_tokens 小的请求应先完成。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())

    scheduler.add_request(Request(
        "short", [1, 2, 3], max_new_tokens=1,
        sampling_params=SamplingParams(temperature=1.0),
    ))
    scheduler.add_request(Request(
        "long", [4, 5, 6], max_new_tokens=5,
        sampling_params=SamplingParams(temperature=1.0),
    ))

    finished_order: list[str] = []
    while not scheduler.is_done:
        just_finished = scheduler.step()
        for req in just_finished:
            finished_order.append(req.request_id)

    # short 应该先完成
    assert finished_order[0] == "short"
    assert finished_order[1] == "long"


# ── 确定性 ─────────────────────────────────────────────────────────────

def test_scheduler_is_deterministic() -> None:
    """相同 seed 跑两次，输出应完全一致。"""

    def run() -> list[list[int]]:
        torch.manual_seed(99)
        scheduler = Scheduler(make_model())
        scheduler.add_request(Request("a", [1, 2], max_new_tokens=3))
        scheduler.add_request(Request("b", [3, 4], max_new_tokens=2))
        while not scheduler.is_done:
            scheduler.step()
        return [req.output_ids for req in scheduler.finished]

    run1 = run()
    run2 = run()
    assert run1 == run2


# ── step() 语义 ────────────────────────────────────────────────────────

def test_step_does_exactly_one_operation() -> None:
    """一次 step() 只做一件事：prefill 一个请求 或 decode 一个 token。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())
    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=5))

    # 第 1 步: prefill
    scheduler.step()
    assert len(scheduler.running) == 1
    assert scheduler.running[0].num_generated_tokens == 1  # prefill 后采了第一个 token

    # 第 2 步: decode 第 2 个 token
    scheduler.step()
    assert scheduler.running[0].num_generated_tokens == 2

    # 第 3 步: decode 第 3 个 token
    scheduler.step()
    assert scheduler.running[0].num_generated_tokens == 3


def test_finished_request_not_processed_again() -> None:
    """完成的请求不应再被推进，token 数量保持不变。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())

    # 短请求：只需要 1 个 token
    scheduler.add_request(Request("short", [1, 2, 3], max_new_tokens=1))
    # 长请求：需要 5 个 token
    scheduler.add_request(Request("long", [4, 5, 6], max_new_tokens=5))

    while not scheduler.is_done:
        scheduler.step()

    short_req = [r for r in scheduler.finished if r.request_id == "short"][0]
    assert short_req.num_generated_tokens == 1
    # 确认没有多余的 token
    assert len(short_req.generated_ids) == 1


# ── 状态转换 ───────────────────────────────────────────────────────────

def test_request_transitions_waiting_to_prefilling_to_decoding_to_finished() -> None:
    """验证请求按 WAITING → PREFILLING → DECODING → FINISHED 演进。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())
    req = Request("r1", [1, 2, 3], max_new_tokens=2)
    assert req.status is RequestStatus.WAITING

    scheduler.add_request(req)
    scheduler.step()  # prefill → 生成第 1 个 token，未完成（1 < 2）
    assert req.status is RequestStatus.DECODING
    assert req.num_generated_tokens == 1

    scheduler.step()  # decode → 生成第 2 个 token，完成
    assert req.status is RequestStatus.FINISHED


# ── 内部状态清理 ──────────────────────────────────────────────────────

def test_kv_cache_cleaned_after_finish() -> None:
    """请求完成后内部 _kv_caches 和 _next_inputs 应被清理。"""
    torch.manual_seed(0)
    scheduler = Scheduler(make_model())
    scheduler.add_request(Request("r1", [1, 2, 3], max_new_tokens=1))

    while not scheduler.is_done:
        scheduler.step()

    assert "r1" not in scheduler._kv_caches
    assert "r1" not in scheduler._next_inputs
