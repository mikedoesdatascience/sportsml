FROM python:3.11-slim

ARG PLATFORM cpu

RUN python -m pip install -U pip

COPY python/requirements.${PLATFORM}.lock /tmp/requirements.${PLATFORM}.lock

RUN python -m pip install -U --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -f https://data.dgl.ai/wheels/repo.html \
        -r /tmp/requirements.${PLATFORM}.lock && \
    rm /tmp/requirements.${PLATFORM}.lock

COPY python /src/sportsml

RUN python -m pip install --no-cache-dir /src/sportsml && \
    rm -r /src/sportsml

ENV DGLBACKEND=pytorch