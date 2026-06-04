FROM python:3.11-slim

WORKDIR /app

# System-Abhängigkeiten für Audio (soundfile/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch separat installieren (cu128-Index)
RUN pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Rest
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    accelerate>=0.26.0 \
    soundfile>=0.12.1 \
    librosa>=0.10.0 \
    "numpy>=1.24,<2.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    httpx>=0.27.0

COPY app.py .

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
