ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG DEV_UID=1001
ARG DEV_GID=1001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/mini-vllm-inductor \
    TRITON_CACHE_DIR=/tmp/mini-vllm-triton \
    XDG_CACHE_HOME=/tmp/mini-vllm-cache

WORKDIR /workspace/mini-vLLM
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Container versions live here. Reuse the image's CUDA PyTorch.
RUN python -m venv --without-pip --system-site-packages /opt/venv \
    && python -m pip --python /opt/venv install --no-cache-dir \
        -e . \
        "torch==2.11.0" \
        "PyYAML==6.0.3" \
        "tqdm==4.70.1" \
        "pytest==9.0.3" \
        "pluggy==1.6.0" \
        "iniconfig==2.3.0" \
    && /opt/venv/bin/python -c \
        "import torch; assert torch.__version__ == '2.11.0+cu128', torch.__version__"

# Match host ownership and provide the account/HOME required by editor attach.
RUN if ! getent group "$DEV_GID" >/dev/null; then \
        groupadd --gid "$DEV_GID" dev; \
    fi \
    && if getent passwd "$DEV_UID" >/dev/null; then \
        dev_user=$(getent passwd "$DEV_UID" | cut -d: -f1); \
        usermod --gid "$DEV_GID" --home /home/dev --shell /bin/bash "$dev_user"; \
    else \
        useradd --uid "$DEV_UID" --gid "$DEV_GID" \
            --home-dir /home/dev --shell /bin/bash dev; \
    fi \
    && mkdir -p /home/dev outputs \
    && chown -R "$DEV_UID:$DEV_GID" /home/dev /opt/venv /workspace/mini-vLLM

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/home/dev

COPY configs/ ./configs/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY examples/ ./examples/
COPY tools/ ./tools/
COPY data/input.txt ./data/input.txt

USER ${DEV_UID}:${DEV_GID}
CMD ["sleep", "infinity"]
