"""Schedule two chat prompts with one model and shared paged storage."""
import argparse

from transformers import AutoTokenizer

from llm.inference.request import Request
from llm.inference.sampler import SamplingParams
from llm.inference.scheduler import Scheduler
from llm.model.qwen2_5_0_5b_instruct.runner import QwenModelRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    runner = QwenModelRunner.from_pretrained(args.model, device=args.device)
    scheduler = Scheduler(runner)
    eos = runner.model.generation_config.eos_token_id
    stop_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)
    for index, prompt in enumerate(("What is 2 + 2?", "Name one planet.")):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        )
        scheduler.add_request(Request(
            str(index), ids, args.max_new_tokens, SamplingParams(do_sample=False), stop_token_ids=stop_ids,
        ))
    for request in scheduler.run():
        print(f"{request.request_id}: {tokenizer.decode(request.generated_ids, skip_special_tokens=True)}")
    print(f"Free KV blocks: {runner.cache.block_manager.allocator.num_free}/{runner.cache.capacity}")


if __name__ == "__main__":
    main()
