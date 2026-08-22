# Production Dockerfile for GATE-CS Doubt Solver Backend
# Optimized for free-tier CPU containers (HuggingFace Spaces / Render / Railway)

FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install Python dependencies (with CPU-optimized llama-cpp-python)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn pydantic

# Copy project source, frontend, and artifacts
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY frontend/ ./frontend/

# Expose FastAPI backend port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
