# Repository Guidelines

## Project Structure & Module Organization

`src/llm/` is the shipped inference package (`llm`). Shared storage lives in
`cache/`; `inference/` contains the Runner contract, request scheduling, and
sampling. `model/qwen2_5_0_5b_instruct/` adapts Qwen forward and Transformers Cache;
`runtime/generate.py` provides the real-model `llm-generate` CLI. Only inference
configuration belongs in `src/llm/config.py` and root `configs/`.

`toy/` is a repository-only GPT model and training module, with its own config,
checkpoint, dataset, direct generator, and train/generate entry points. Its
learning-oriented character tokenizer lives in `toy/tokenizer.py`. Concrete
Runners live in `src/llm/model/gpt_mini/` and
`src/llm/model/qwen2_5_0_5b_instruct/`; retain the public Runner abstraction. Model,
training, and checkpoint tests live in `toy/tests/`; public cache, inference,
and Qwen tests live in `tests/cache`, `tests/inference`, and `tests/models`.
Root `examples/` is for engine usage. Do not recreate toy examples/benchmarks.
Root `benchmarks/` is for shared cache analysis and future engine benchmarking;
toy dataset tools remain under `toy/tools/`. Model adapter directories use full
model names normalized to valid Python identifiers. New models implement Runner
inside their adapter directory; do not add model-specific Scheduler branches.
Do not create unused placeholder modules for future training or export work.

## Build, Test, and Development Commands

Use the existing GPU container through `./docker/dev.sh`. Its interpreter is
`/opt/venv/bin/python`; `.python-version` selects Python 3.12 for development
while package support remains Python 3.11+.

- `./docker/dev.sh python -m pytest tests toy/tests -q` runs all maintained tests.
- `./docker/dev.sh python -m pytest tests/cache -q` runs shared cache tests.
- `./docker/dev.sh python -m toy.train --config toy/configs/base.yaml` trains toy.
- `./docker/dev.sh python -m toy.generate --ckpt outputs/ckpt.pt --prompt hello`
  loads an existing character-model checkpoint.
- `./docker/dev.sh llm-generate --config configs/qwen.yaml --prompt hello`
  runs the real-model entry point using local mounted weights.
- `./docker/dev.sh python benchmarks/kv_memory.py` runs memory arithmetic.

Run modules from the repository root. The installed wheel contains only `src/llm`;
`toy` is intentionally available from the checkout, not the wheel.
`docker/Dockerfile` is the sole container installation recipe and owns version
pins. Reuse CUDA PyTorch; do not run `uv sync` over the active environment.
`pyproject.toml` and `uv.lock` describe standalone package workflows.
`docker/dev.sh` executes commands and can start the existing container; it does
not install dependencies or create a replacement environment. Keep environment
setup in `docker/`, never in benchmark scripts. Retain existing model mounts,
data, training artifacts, and unrelated changes.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and uppercase enum members. Follow existing type annotations and concise docstrings. Preserve one-parameter-per-line formatting in multiline model and runner signatures. No formatter or linter is configured; match surrounding code and avoid unrelated reformatting.

## Testing Guidelines

Use `pytest` functions named `test_<behavior>`. Prefer tiny CPU model configurations, disable dropout for numerical comparisons, and seed PyTorch when testing randomness. Cover tensor shapes, block boundaries, allocation/free behavior, request cleanup, and scheduler fairness. Add regression tests for behavior changes. No coverage percentage is configured. Run focused tests during development and the full suite before submitting; report failures and skips explicitly.

## Commit & Pull Request Guidelines

Use short imperative subjects following history, such as `Add KV block allocator and block table` or `Fix scheduler fairness, sampling, and edge cases`. Keep commits focused. PR descriptions should explain the problem, resulting behavior, relevant issues, and validation commands/results. Include benchmark comparisons for memory or scheduling changes.

## Architecture & Configuration

Keep this a PyTorch teaching implementation. Scheduler receives an injected
Runner and controls request admission, cache allocation/growth/release, and
sampling. Adapters parse model specifications and construct cache resources.
BlockManager executes block ownership changes, KVPool stores tensors, and
PagedKVCache maps logical positions and gathers standard KV pages. Shared cache
code must not import scheduling, model adapters, or toy. Shared scheduling must
not inspect model topology or KV tensor layouts. Qwen Cache adapts Transformers
forward and never owns a second request lifecycle.

Future MoE expert routing/execution belongs in the model layer. Standard-KV MoE
may reuse existing cache resources; MLA and other special cache layouts require
separate adapter/storage changes. Do not implement MoE operators, expert
parallelism, a registry, or a large model factory ahead of an actual requirement.
Paged storage plus gather is not a high-performance paged-attention kernel.
Keep generated checkpoints, logs, and datasets out of commits; honor .gitignore.

## Lab Tutoring

Follow `.local/learning/README.md`. Explain in Chinese for a learner with basic Python and little PyTorch. Provide interfaces, tests, and progressive hints; leave student TODOs unimplemented unless explicitly asked for solutions. Never weaken tests to produce a pass. Run exercises explicitly with `python -m pytest .local/learning/lab01_attention -q`; they are separate from production tests. Pretrained model loading is now a course objective. Preserve existing server work and record actual validation in `.local/learning/STATUS.md`; do not store credentials.

## Local-only Materials

Use the workspace selected for the current task as the authoritative checkout.
Preserve existing learner implementations; never replace them with a mirror.
Personal learning code/tests/handouts stay in `.local/learning/labNN_topic/`,
and detailed project records in `.local/docs/`. These directories and root
CLAUDE.md are ignored. Root AGENTS.md is versionable. README provides a short project overview and
links to detailed documentation in `docs/`. Public documentation must use
generic paths and must not include personal usernames or machine-specific locations.
The shipped llm package must not import toy or personal lessons. GPTMiniRunner
accepts an externally constructed model. Direct toy generation calls model
forward without constructing Runner/Scheduler/cache resources. Toy uses its
preserved educational `toy.tokenizer` implementation. Restrict default pytest
discovery to `tests/` and `toy/tests/`; run personal learning tests explicitly.
See `.local/README.md` for the local-materials layout.
