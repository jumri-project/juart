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

# Install finufft and torch
# RUN pip install --no-cache-dir \
#     --extra-index-url https://download.pytorch.org/whl/cpu
#     finufft \
#     torch

# Add non-root user
ARG USERNAME=nonroot
RUN groupadd --gid 1000 $USERNAME && \
    useradd --uid 1000 --gid 1000 -m $USERNAME

## Make sure to reflect new user in PATH
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
USER $USERNAME