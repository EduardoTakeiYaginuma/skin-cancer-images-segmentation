FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-api.txt

COPY skin_app/ skin_app/
COPY api/ api/
COPY config/ config/
COPY mlflow_config.py .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
