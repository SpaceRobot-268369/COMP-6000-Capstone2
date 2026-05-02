FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies at build time so container starts instantly
COPY requirements.txt acoustic_ai/requirements.txt
RUN pip install --no-cache-dir -r acoustic_ai/requirements.txt

CMD ["python3", "acoustic_ai/server.py"]
