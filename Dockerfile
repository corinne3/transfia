# Image de l'API (pour le cloud : Cloud Run, Container Apps... chapitres 13-14 du dossier).
# Ollama tourne à part : régler llm.base_url, ou utiliser config/gcp.yaml (LiteLLM).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY skills ./skills
COPY config ./config
COPY data/samples ./data/samples
RUN pip install --no-cache-dir -e ".[local,api,mcp]"
ENV TRANSFIA_CONFIG=config/local.yaml
EXPOSE 8000
CMD ["transfia", "serve", "--host", "0.0.0.0", "--port", "8000"]
