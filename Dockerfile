# syntax=docker/dockerfile:1
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /output \
    && chown app:app /output

COPY solution.py benchmark.py metric.py ./
COPY tests/ ./tests/

USER app
RUN python -m unittest discover -s tests -v

CMD ["python", "solution.py", "--data-dir", "/data", "--output", "/output/submission.csv"]
