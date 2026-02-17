# Production Deployment

This guide covers deploying your rollout server to production environments using Docker, Docker Compose, health checks, and logging best practices.

## Docker Configuration

Using the CLI (recommended):

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9000

# Using CLI (validates on startup)
CMD ["osmosis", "serve", "-m", "main:agent_loop", "-p", "9000", "--skip-register"]
```

Or with uvicorn directly:

```dockerfile
# Dockerfile (alternative)

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
```

```txt
# requirements.txt
osmosis-ai[server,config]>=0.1.0
```

## Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  rollout-server:
    build: .
    ports:
      - "9000:9000"
    environment:
      - OSMOSIS_ROLLOUT_LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Health Check Integration

```python
# For Kubernetes or load balancer health checks

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent_loop": agent_loop.name,
    }

@app.get("/ready")
async def ready():
    # Add any readiness checks here
    return {"ready": True}
```

## Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Reduce httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)
```

## See Also

- [Examples](./examples.md) -- agent implementations
- [Testing](./testing.md) -- unit tests and mock trainer
- [Architecture](./architecture.md) -- protocol design
