FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM node:22-bookworm-slim AS webcam-build
WORKDIR /build
COPY webcam/package*.json ./
RUN npm ci
COPY webcam/ ./
RUN mkdir -p public/vendor/mediapipe && \
    cp -r node_modules/@mediapipe/tasks-vision/wasm/. public/vendor/mediapipe/ && \
    npm run build

FROM python:3.11-slim-bookworm AS webcam
COPY --from=webcam-build /build/dist/ /web/
RUN mkdir -p /web/models
USER 1000:1000
EXPOSE 5180
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s \
  CMD ["python", "-c", "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:5180/', timeout=2).status == 200"]
CMD ["python", "-m", "http.server", "5180", "--bind", "0.0.0.0", "--directory", "/web"]

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SIGNACULTURE_STORAGE=/app/.runtime SIGNACULTURE_DB_PATH=/app/.runtime/plans.sqlite3 \
    SIGNACULTURE_MOTION_PYTHON=/opt/motion/bin/python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /tmp/api.txt
COPY experiments/motion_extraction/requirements-optional.txt /tmp/motion.txt
RUN pip install -r /tmp/api.txt \
    && python -m venv /opt/motion \
    && /opt/motion/bin/pip install -r /tmp/motion.txt
COPY backend/ ./backend/
COPY deployment/__init__.py deployment/server.py ./deployment/
COPY site/ ./site/
COPY docs/diagrams/ ./docs/diagrams/
COPY data/glossary.json data/motions_manifest.json ./data/
COPY data/approved_motion_assets/ ./data/approved_motion_assets/
COPY data/public_examples/ ./data/public_examples/
COPY --from=frontend /build/dist/ ./frontend/dist/
RUN mkdir -p /app/.runtime /app/.models /avatars
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD ["python", "-c", "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200"]
CMD ["python", "-m", "uvicorn", "deployment.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

FROM runtime AS full
ENV SIGNACULTURE_AUDIO_PYTHON=/opt/audio/bin/python
COPY deployment/requirements-docker-audio.txt /tmp/audio.txt
RUN python -m venv /opt/audio \
    && /opt/audio/bin/pip install torch==2.7.1+cpu torchaudio==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && /opt/audio/bin/pip install -r /tmp/audio.txt
