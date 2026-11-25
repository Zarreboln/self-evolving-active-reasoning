# The proposer's sandbox: Claude Code and Python, and nothing of the host. The orchestrator mounts one
# workspace at /work (the only host path the container sees); HOME is fresh in every run, so no
# CLAUDE.md, memory, skills, plugins or earlier sessions can leak in.
FROM node:22-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-numpy ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*
ARG CLAUDE_CODE_VERSION=2.1.326
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} && npm cache clean --force
RUN useradd -m -u 1001 proposer
USER proposer
WORKDIR /work
