FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    x11-apps \
    vim \
    net-tools \
    pkg-config \
    libeigen3-dev \
    libopencv-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv python install 3.12
RUN uv venv /opt/venv --python 3.12

ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /workspace

COPY requirements.txt /workspace/rl_sandbox/requirements.txt
RUN uv pip install -r /workspace/rl_sandbox/requirements.txt

RUN uv pip install \
    -U "jax[cuda12_local]" \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    genesis-world \
    mujoco \
    mujoco_mjx \
    brax \
    playground \
    --index-url https://download.pytorch.org/whl/cu129 \
    --extra-index-url https://pypi.org/simple

RUN command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
RUN uv pip install -q mediapy 

RUN uv pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-cupti-cu12

ENV MUJOCO_GL="egl"

RUN apt-get update && apt-get install -y \   
    libglvnd0 libgl1 libegl1 libgles2 libdrm2 libgbm1

RUN uv pip install pygame pymunk shapely zarr numcodecs