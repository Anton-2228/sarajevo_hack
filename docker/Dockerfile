# fasttext-community ships wheels for 3.9-3.13; 3.12 is what .python-version pins.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing src/ does not re-resolve the wheel set.
COPY pyproject.toml README.md ./
RUN pip install "fasttext-community>=0.11.7" "numpy>=1.26" "requests>=2.31" "psutil>=5.9"

COPY src ./src
RUN pip install --no-deps .

# The agent writes identity, models and the journal here; compose mounts a
# named volume per node so each container enrols as its own node and keeps it
# across restarts.
ENV NODE_AGENT_STATE_DIR=/var/lib/node-agent
RUN mkdir -p /var/lib/node-agent

ENTRYPOINT ["node-agent"]
CMD ["run"]
