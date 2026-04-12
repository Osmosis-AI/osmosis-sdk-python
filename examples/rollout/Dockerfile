FROM python:3.12-slim

# System deps: git for pip installs, docker CLI for HarborBackend
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (not the daemon — the host daemon is mounted via socket)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar xz --strip-components=1 -C /usr/local/bin docker/docker

# Install docker-compose plugin
RUN mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL https://github.com/docker/compose/releases/download/v2.32.4/docker-compose-linux-x86_64 \
       -o /usr/local/lib/docker/cli-plugins/docker-compose \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

WORKDIR /app

# Copy SDK source and install with server extras
# Build context must be the repo root:
#   docker build -f examples/rollout_v2/Dockerfile .
COPY pyproject.toml uv.lock ./
COPY osmosis_ai ./osmosis_ai
RUN pip install --no-cache-dir ".[server]"

# Copy the rollout_v2 examples (workflow, grader, harbor task, entrypoint)
COPY examples/rollout_v2 ./examples/rollout_v2

WORKDIR /app/examples/rollout_v2

EXPOSE 8000

# Mount the Docker socket at runtime:
#   docker run -v /var/run/docker.sock:/var/run/docker.sock -p 8000:8000 <image>
CMD ["python", "harbor_rollout_server_example.py"]
