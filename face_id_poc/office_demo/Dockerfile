# Context layout: face_id_poc (NOT office_demo or its parent repository).
# Prefer scripts/build_image.py: it stages only allowlisted source files.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ca-certificates locales \
    ffmpeg libglib2.0-0 libgomp1 zlib1g \
    && locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3.10-dev && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY office_demo/requirements*.txt /tmp/requirements/
# Cython-bbox needs NumPy headers at build time. Do not use Ubuntu's pip wrapper.
RUN python -m pip install --index-url https://pypi.org/simple \
      pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 Cython==3.0.11 numpy==1.26.4 \
    && python -m pip install --index-url https://pypi.org/simple \
      --no-build-isolation -r /tmp/requirements/requirements-gpu.txt \
    && python -m pip check

FROM base AS runtime
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app:/app/research_demo:/app/office_demo \
    CAMEX_OFFICE_HOME=/runtime CAMEX_OFFICE_MODELS=/models \
    CAMEX_OFFICE_PROVIDER=CUDAExecutionProvider \
    CAMEX_OFFICE_QDRANT_URL=http://qdrant:6333 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
WORKDIR /app/office_demo
COPY research_demo/vision /app/research_demo/vision
COPY research_demo/core.py /app/research_demo/core.py
COPY office_demo/*.py ./
COPY office_demo/models_manifest.json ./models_manifest.json
COPY office_demo/static ./static
COPY office_demo/scripts ./scripts
RUN groupadd --gid 10001 office && useradd --uid 10001 --gid office --no-create-home office \
    && mkdir -p /runtime /models && chown office:office /runtime
USER 10001:10001
EXPOSE 9170
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9170", "--workers", "1", "--no-access-log"]
