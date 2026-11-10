FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LOG_LEVEL=INFO

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (small, no GPU needed), then everything else from requirements.txt
# Deployment image: torch is not needed at serving time (autoencoder inference
# runs in NumPy via src/models/ae_serve.py; see README "Deployment").
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY . .

EXPOSE 8000 8501
# Default command runs the API; docker-compose overrides it for the dashboard.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
