FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/Osmosis-AI/osmosis-sdk-python.git@feat/support-rollout-v2
RUN pip install strands-agents strands-agents-tools fastapi harbor

COPY workspace /workspace

CMD ["sleep", "infinity"]
