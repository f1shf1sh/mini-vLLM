"""Prevent model and cache layout details from leaking into the engine."""
import ast
from pathlib import Path


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield '.' * node.level + (node.module or '')


def test_public_cache_has_no_engine_or_model_dependencies():
    root = Path(__file__).resolve().parents[2] / 'src/llm/cache'
    for path in root.glob('*.py'):
        for module in imports(path):
            assert not any(part in module.split('.') for part in ('inference', 'models', 'model', 'toy'))


def test_scheduler_depends_only_on_runner_resource_contract():
    path = Path(__file__).resolve().parents[2] / 'src/llm/inference/scheduler.py'
    for module in imports(path):
        assert not any(part in module.split('.') for part in ('cache', 'models', 'model', 'toy'))
