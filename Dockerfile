FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080
WORKDIR /app
COPY pyproject.toml ./
COPY copilot ./copilot
RUN pip install ".[gcp]"
COPY sample_data ./sample_data
# Run as a normal user; /app must be writable for the generated sample database.
RUN useradd --create-home app && chown -R app /app
USER app
# Cloud Run sends traffic to $PORT
CMD ["sh", "-c", "uvicorn copilot.main:build_app --factory --host 0.0.0.0 --port ${PORT}"]
