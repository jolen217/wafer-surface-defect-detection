# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    curl \
    libfontconfig1-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for production
RUN pip install --no-cache-dir gunicorn

# Copy the application code
COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY train.py .
COPY inference.py .

# Create necessary directories (models and checkpoints will be mounted)
RUN mkdir -p logs data/processed models checkpoints

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use gunicorn for production deployment
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "api.main:app", "-k", "uvicorn.workers.UvicornWorker"]