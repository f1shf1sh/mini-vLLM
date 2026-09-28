#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=${MINIVLLM_CONTAINER:-mini-vllm-dev}

if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  printf 'Container %s does not exist. Follow the Docker setup in README.md.\n' "$CONTAINER_NAME" >&2
  exit 1
fi
if [[ ${1:-} == setup ]]; then
  printf 'Dependencies are installed by docker/Dockerfile. See README.md.\n' >&2
  exit 2
fi
if [[ $(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME") != true ]]; then
  docker start "$CONTAINER_NAME" >/dev/null
fi
if [[ $# -eq 0 || ${1:-} == shell ]]; then
  set -- bash
fi
TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_FLAGS=(-it)
fi

exec docker exec "${TTY_FLAGS[@]}" \
  --workdir /workspace/mini-vLLM \
  --env HOME=/home/dev \
  --env VIRTUAL_ENV=/opt/venv \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  "$CONTAINER_NAME" "$@"
