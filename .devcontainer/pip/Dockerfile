ARG BASE_CONTAINER=quay.io/condaforge/miniforge3
# ARG BASE_CONTAINER=python:3.12
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

COPY pyproject.toml /tmp/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv

RUN uv pip compile /tmp/pyproject.toml \
    --output-file /tmp/requirements.txt \
    --all-extras

# uv pip install currently leads to compatibility issues with ipywidgets.
# As a result, we are using the standard pip installation method.
RUN uv pip install --no-cache-dir --system \
    --requirement /tmp/requirements.txt
# RUN pip install --no-cache-dir \
#     --requirement /tmp/requirements.
