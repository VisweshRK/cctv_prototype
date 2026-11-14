# Multi-stage Dockerfile for Presence B2C AI CCTV Platform
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and multimedia processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt asyncpg

# Copy application source
COPY ai/ /app/ai/
COPY apps/ /app/apps/
COPY packages/ /app/packages/
COPY docs/ /app/docs/

# Set Python path
ENV PYTHONPATH=/app
ENV USE_INT8_DETECTOR=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/admin/health || exit 1

CMD ["uvicorn", "apps.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
