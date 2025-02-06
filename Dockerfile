FROM python:3.11.11-slim

WORKDIR /app
COPY requirements.docker.txt .
COPY src/models/ src/models/
COPY src/api/ src/api/
COPY src/inference/ src/inference/
COPY src/data/dataset.py src/data/dataset.py

RUN pip install -r requirements.docker.txt

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"] 