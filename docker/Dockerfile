FROM python:3.10

COPY python/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY python /src/sportsml

RUN python -m pip install --no-cache-dir /src/sportsml && \
    rm -r /src/sportsml
