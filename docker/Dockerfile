FROM python:3.11-slim

ARG PLATFORM cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=python/requirements.${PLATFORM}.lock,target=requirements.${PLATFORM}.lock \
    python -m pip install -U pip && \
    python -m pip install -U \
        --extra-index-url https://download.pytorch.org/whl/${PLATFORM} \
        -f https://data.dgl.ai/wheels/${PLATFORM}/repo.html \
        -r requirements.${PLATFORM}.lock

COPY python /src/sportsml

RUN python -m pip install /src/sportsml && \
    rm -r /src/sportsml

ENV DGLBACKEND=pytorch