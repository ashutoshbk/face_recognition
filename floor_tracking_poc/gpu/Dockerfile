FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime@sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/camex
COPY gpu/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --no-deps ultralytics==8.4.137
COPY engines /opt/camex/engines
COPY gpu/setup_vendor.py /tmp/setup_vendor.py
RUN python /tmp/setup_vendor.py
COPY src /opt/camex/src
COPY gpu /opt/camex/gpu
COPY tests /opt/camex/tests
COPY examples /opt/camex/examples
ENV CAMEX_MCT_ROOT=/opt/camex/engines/mct CAMEX_MCT_THIRD_PARTY=/opt/camex/third_party
ENV PYTHONPATH=/opt/camex/src:/opt/camex/engines/mct/src:/opt/camex/engines/calibration/vendor/GeoCalib:/opt/camex/engines/calibration/vendor/vggt:/opt/camex/third_party/ByteTrack:/opt/camex/third_party/BoT-SORT:/opt/camex/third_party/deep-person-reid
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ENV CAMEX_FFMPEG=/usr/bin/ffmpeg
ENV YOLO_CONFIG_DIR=/job/.cache/ultralytics MPLCONFIGDIR=/job/.cache/matplotlib TORCH_HOME=/job/.cache/torch
CMD ["python", "/opt/camex/gpu/worker.py", "doctor"]
