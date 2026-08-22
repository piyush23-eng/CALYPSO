# Production Dockerfile for Calypso Reasoning Engine
# Optimized for free-tier cloud containers (Render / Hugging Face Spaces / Railway)

FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create models directory structure
RUN mkdir -p models/gguf

# Copy project source, data, and frontend assets
COPY src/ ./src/
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Expose standard port
EXPOSE 8000

# Start FastAPI server dynamically binding to Render's $PORT (default 8000)
CMD ["sh", "-c", "uvicorn src.backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
