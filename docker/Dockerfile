FROM python:3.11

RUN python -m pip install -U pip

COPY python/requirements.lock /tmp/requirements.lock

RUN python -m pip install -U --no-cache-dir -r /tmp/requirements.lock && \
    rm /tmp/requirements.lock

COPY python /src/sportsml

RUN python -m pip install --no-cache-dir /src/sportsml && \
    rm -r /src/sportsml
