# Anthropic-format -> OpenAI-format bridge (LiteLLM proxy), so Claude Code can drive models that
# Prime Intellect serves only through its OpenAI-compatible API. Reaches the internet only through
# the egress proxy (allowlist), on the internal proposer network.
FROM python:3.12-slim
RUN pip install --no-cache-dir "litellm[proxy]==1.77.5"
COPY config.yaml /app/config.yaml
EXPOSE 4000
CMD ["litellm", "--config", "/app/config.yaml", "--port", "4000", "--host", "0.0.0.0"]
