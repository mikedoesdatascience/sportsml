FROM python:3.11-slim

ARG PLATFORM cuda

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=python/requirements.${PLATFORM}.lock,target=requirements.${PLATFORM}.lock \
    python -m pip install -U pip && \
    python -m pip install -U \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        -f https://data.dgl.ai/wheels/cu118/repo.html \
        -r requirements.${PLATFORM}.lock

COPY python /src/sportsml

RUN python -m pip install /src/sportsml && \
    rm -r /src/sportsml

ENV DGLBACKEND=pytorch