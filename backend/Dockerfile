FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . .
RUN pip install -e ".[dev]"

# Optional local reranker (RERANKER=local): CPU-only torch + sentence-transformers.
# Build with: INSTALL_LOCAL_RERANKER=true docker compose build backend
ARG INSTALL_LOCAL_RERANKER=false
RUN if [ "$INSTALL_LOCAL_RERANKER" = "true" ]; then       pip install torch --index-url https://download.pytorch.org/whl/cpu &&       pip install -e ".[rerank-local]";     fi

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
