# Use slim Python base
FROM python:3.10-slim

# Prevent Python from writing .pyc and force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set workdir inside container
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entrypoint script
COPY entrypoint.py /entrypoint.py
RUN chmod +x /entrypoint.py

# GitHub Actions entrypoint
ENTRYPOINT ["/entrypoint.py"]
