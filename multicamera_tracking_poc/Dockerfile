FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CAMEX_MCT_ROOT=/app \
    CAMEX_MCT_THIRD_PARTY=/opt/camex/third_party \
    PYTHONPATH=/app/src:/opt/camex/third_party/ByteTrack:/opt/camex/third_party/BoT-SORT:/opt/camex/third_party/deep-person-reid

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt pyproject.toml third_party.lock ./
COPY patches ./patches
COPY scripts/setup_dependencies.sh /tmp/setup_dependencies.sh
RUN CAMEX_MCT_ROOT=/app CAMEX_MCT_THIRD_PARTY=/opt/camex/third_party \
    bash /tmp/setup_dependencies.sh \
    && python -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY src ./src
COPY configs ./configs
COPY examples ./examples
COPY scripts ./scripts
COPY tests ./tests
COPY README.md LICENSE_AND_USAGE.md MODEL_PROVENANCE.md THIRD_PARTY_NOTICES.md ./
RUN python -m pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "pytest", "-q", "tests"]
