# Use uv's ARM64 Python 3.13 base image
FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set the working directory inside the container
WORKDIR /app

# Copy uv files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Download the RDS CA certificate bundle for SSL connections to PostgreSQL
RUN apt-get update && apt-get install -y wget && \
    wget -O /app/global-bundle.pem https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem && \
    apt-get remove -y wget && rm -rf /var/lib/apt/lists/*

# Copy Python modules
COPY agent.py ./

# Expose port
EXPOSE 8080

# Run application
CMD ["uv", "run", "python", "agent.py"]