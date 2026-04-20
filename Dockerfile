# ML SuperTrend v51 — GPU-Enabled Trading Bot
# Base: PyTorch CUDA 12.4 + Python 3.11
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

LABEL maintainer="ML SuperTrend Bot"
LABEL description="Algorithmic trading bot with 20 scientific ML strategies"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C library (required by ta-lib Python package)
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements_gpu.txt .
RUN pip install --no-cache-dir -r requirements_gpu.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data /app/experiments /app/models /app/logs

# Expose dashboard port
EXPOSE 5000

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=America/New_York

# Health check — dashboard responds
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Default: run bot in demo mode
CMD ["python", "main.py", "--demo"]
