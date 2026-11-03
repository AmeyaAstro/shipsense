# ShipSense prediction API.
# Build:  make docker-build      Run:  make docker-run   (needs `make train` first for models/)
FROM python:3.11-slim

# No .pyc files; print logs immediately (important for `docker logs`).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first: this layer is cached and only rebuilds when the file changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Then the code and the production model (the API reads the threshold from metrics.json).
COPY src/ src/
COPY models/gradient_boosting.joblib models/
COPY reports/metrics.json reports/

# Run as an unprivileged user: if the app were compromised, it can't modify the image.
# COPY keeps the host's file modes (a file that is owner-only on a laptop would be
# unreadable here), so make the app files world-readable but still not writable.
RUN chmod -R a+rX,go-w /app/src /app/models /app/reports \
    && useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
# Docker marks the container unhealthy if /health stops answering (slim has no curl, so use Python).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# 0.0.0.0 = accept connections from outside the container (127.0.0.1 would only allow inside).
CMD ["uvicorn", "shipsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
