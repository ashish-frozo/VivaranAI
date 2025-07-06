# Multi-stage Dockerfile for MedBillGuard Agent System
# Production-ready container with security and optimization best practices

# Stage 1: Base dependencies and build environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r medbillguard && useradd -r -g medbillguard medbillguard

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Stage 2: Development dependencies (for testing and development)
FROM base as dev-deps

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Stage 3: Production dependencies only
FROM base as prod-deps

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Application code
FROM prod-deps as app

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R medbillguard:medbillguard /app

# Switch to non-root user
USER medbillguard

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "-m", "agents.server"]

# Stage 5: Development image with all tools
FROM dev-deps as development

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories  
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R medbillguard:medbillguard /app

# Switch to non-root user
USER medbillguard

# Expose ports for development (including debug ports)
EXPOSE 8000 8001 8002 5678

# Default command for development
CMD ["python", "-m", "agents.server", "--debug"]

# Stage 6: Testing image
FROM dev-deps as testing

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create test directories
RUN mkdir -p /app/test-results /app/coverage && \
    chown -R medbillguard:medbillguard /app

# Switch to non-root user  
USER medbillguard

# Default command for testing
CMD ["python", "-m", "pytest", "--cov=agents", "--cov-report=xml:/app/coverage/coverage.xml", "--junit-xml=/app/test-results/junit.xml"] 