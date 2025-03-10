ARG BASE_CONTAINER=quay.io/condaforge/miniforge3
FROM $BASE_CONTAINER

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    # (TODO) A recommendation that comes along with 
    # build-essential is required to pip install finufft
    # && apt-get install -y --no-install-recommends \
    && apt-get install -y \
        build-essential \
        ssh \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mamba install --yes \
    --channel conda-forge \
    'h5py' \
    'ipykernel' \
    'ipympl' \
    'matplotlib' \
    'numpy' \
    'pip' \
    'pre-commit' \
    'ruff' \
    'scipy' \
    && \
    mamba clean --all -f -y

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    finufft \
    torch