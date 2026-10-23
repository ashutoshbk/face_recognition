FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime@sha256:a355f16160f64219173261456bd5a62a8b99c3fb76ee405c7929a2c8df7dfeb3
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /opt/camex-faceid
RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc g++ ffmpeg
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir -r requirements-cuda.txt
COPY . .
RUN python -m pip install --no-deps .
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/torch/lib:/opt/conda/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
ENV FFMPEG_BIN=/usr/bin/ffmpeg
ENTRYPOINT ["camex-faceid"]
