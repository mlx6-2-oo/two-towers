FROM python:3.11.11-slim

WORKDIR /app

# Install only CPU version of PyTorch to reduce size
RUN pip install --no-cache-dir torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Copy and install other requirements separately to leverage caching
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt \
    && rm -rf /root/.cache/pip

# Copy only necessary files
COPY src/models/ src/models/
COPY src/api/ src/api/
COPY src/inference/ src/inference/
COPY src/data/dataset.py src/data/dataset.py

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"] 