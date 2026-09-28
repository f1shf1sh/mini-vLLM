# Repository Guidelines

## Project Structure & Module Organization

`src/llm/` contains the Python package. The primary focus is `inference/`: paged KV storage, block allocation, request scheduling, sampling, and model execution. `model/` provides the toy GPT testbed; `runtime/` contains CLI entry points. Configuration, tokenization, datasets, and checkpoint helpers sit at the package root.

Keep tests in `tests/test_<module>.py`, configuration in `configs/base.yaml`, benchmark scripts in `benchmarks/`, and benchmark documentation in `.local/docs/benchmark.md`. Training text lives in `data/`; generated checkpoints belong in `outputs/`.

## Build, Test, and Development Commands

Development runs inside the existing GPU container. On the host, use
`./docker/dev.sh shell` or `./docker/dev.sh python -m pytest tests -q`.
The common interpreter is `/opt/venv/bin/python`. Python 3.12 is selected by
`.python-version`; package support remains Python 3.11+.

Inside the container:
- `python -m pytest tests -q` runs production tests.
- `python -m pytest tests/test_block_manager.py -v` runs allocator tests.
- `llm-train --config configs/base.yaml` runs toy-model training.
- `python benchmarks/kv_memory.py` and `python benchmarks/block_usage.py` run benchmarks.

docker/Dockerfile is the sole container installation recipe. It contains container
version pins and reuses the CUDA PyTorch base. Do not run uv sync over the active
container environment. pyproject.toml and uv.lock retain package/standalone
workflow metadata. See README.md for image build and editor attach instructions.
docker/dev.sh is a portable, versionable execution helper; it does not install
dependencies or create containers.

Container tooling lives in docker/, runnable demos in examples/, and dataset
preparation in tools/. Keep these separate from benchmarks/.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and uppercase enum members. Follow existing type annotations and concise docstrings. Preserve one-parameter-per-line formatting in multiline model and runner signatures. No formatter or linter is configured; match surrounding code and avoid unrelated reformatting.

## Testing Guidelines

Use `pytest` functions named `test_<behavior>`. Prefer tiny CPU model configurations, disable dropout for numerical comparisons, and seed PyTorch when testing randomness. Cover tensor shapes, block boundaries, allocation/free behavior, request cleanup, and scheduler fairness. Add regression tests for behavior changes. No coverage percentage is configured. Run focused tests during development and the full suite before submitting; report failures and skips explicitly.

## Commit & Pull Request Guidelines

Use short imperative subjects following history, such as `Add KV block allocator and block table` or `Fix scheduler fairness, sampling, and edge cases`. Keep commits focused. PR descriptions should explain the problem, resulting behavior, relevant issues, and validation commands/results. Include benchmark comparisons for memory or scheduling changes.

## Architecture & Configuration

Keep this a pure-PyTorch teaching implementation. Preserve dependency injection and separate block ownership (`BlockManager`) from tensor storage (`KVPool`). Keep generated checkpoints, logs, and local datasets out of commits; honor `.gitignore`.

## Lab Tutoring

Follow `.local/learning/README.md`. Explain in Chinese for a learner with basic Python and little PyTorch. Provide interfaces, tests, and progressive hints; leave student TODOs unimplemented unless explicitly asked for solutions. Never weaken tests to produce a pass. Run exercises explicitly with `python -m pytest .local/learning/lab01_attention -q`; they are separate from production tests. Pretrained model loading is now a course objective. Preserve existing server work and record actual validation in `.local/learning/STATUS.md`; do not store credentials.

## Local-only Materials

The authoritative workspace is `/home/yzl/mini-vLLM` on the Linux server. Preserve server learner implementations; do not replace them with the Windows mirror. Learning code, its tests, and handouts live together in `.local/learning/labNN_topic/`; detailed project docs live in `.local/docs/`. These directories and root CLAUDE.md are ignored by Git. Root AGENTS.md is versionable under the current ignore rules. The public README contains only the project overview and runnable commands. Production modules must not import learning modules. Keep default pytest discovery restricted to `tests/`; run learning tests explicitly. See `.local/README.md` for the layout and migration details.
