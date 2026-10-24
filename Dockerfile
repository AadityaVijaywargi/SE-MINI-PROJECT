# SM-73, SM-75 (OTH-2, SEC-8, PERF-6): multi-stage build on python:3.11-slim, non-root user.

# Stage 1: install the pinned dependencies into a virtualenv.
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: copy only the virtualenv and the application code.
FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PORT=5000 \
    UPLOAD_DIR=/app/uploads \
    TF_CPP_MIN_LOG_LEVEL=2
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app.py config.py gunicorn.conf.py ./
COPY --chown=app:app core ./core
COPY --chown=app:app static ./static
COPY --chown=app:app data/samples ./data/samples
RUN mkdir -p /app/uploads && chown app:app /app/uploads
USER 1000:1000
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '5000'), timeout=4)"
CMD ["gunicorn", "-c", "gunicorn.conf.py"]
