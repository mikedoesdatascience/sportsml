FROM python:3.11-slim

ARG PLATFORM cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=python/requirements.${PLATFORM}.lock,target=requirements.${PLATFORM}.lock \
    python -m pip install -U uv && \
    python -m uv pip install -U --system \
        --extra-index-url https://download.pytorch.org/whl/${PLATFORM} \
        -r requirements.${PLATFORM}.lock

COPY python /src/sportsml

RUN python -m pip install /src/sportsml && \
    rm -r /src/sportsml

ENV DGLBACKEND=pytorch