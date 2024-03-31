FROM python:3.11-slim as base

ARG PLATFORM cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=requirements.${PLATFORM}.lock,target=requirements.${PLATFORM}.lock \
    python -m pip install -U uv && \
    python -m uv pip install -U --system \
    --extra-index-url https://download.pytorch.org/whl/${PLATFORM} \
    -r requirements.${PLATFORM}.lock


FROM base

COPY . /src/sportsml

RUN python -m uv pip install --system --no-deps /src/sportsml && \
    rm -r /src/sportsml

ENV DGLBACKEND=pytorch