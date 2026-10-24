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

# RAG evaluation (Module 5): RAGAS lives in its OWN virtualenv because ragas 0.4.3 pins an older
# LangChain stack that would downgrade the app's langchain-openai. langchain-community is pinned
# to 0.3.31 because ragas 0.4.3 imports a module that newer releases removed.
# Pins are mirrored in eval/rag/requirements-ragas.txt. Skip with INSTALL_EVAL=false (-770 MB).
ARG INSTALL_EVAL=true
RUN if [ "$INSTALL_EVAL" = "true" ]; then       python -m venv /opt/ragas-venv &&       /opt/ragas-venv/bin/pip install --no-cache-dir "ragas==0.4.3" &&       /opt/ragas-venv/bin/pip install --no-cache-dir --no-deps "langchain-community==0.3.31";     fi
