FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./
COPY src ./src

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 fastmcp>=0.1.0 && \
    pip install --no-cache-dir -e .

# Run the server
CMD ["python", "-m", "policy_mcp.server", "--host", "0.0.0.0", "--port", "8004"]

