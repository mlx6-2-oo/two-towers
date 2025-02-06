FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the app-specific requirements file
COPY app-requirements.txt .

# Install torch first so it doesn't include CUDA
RUN pip install --no-cache-dir torch --index-url=https://download.pytorch.org/whl/cpu

# Install dependencies
RUN pip install --no-cache-dir -r app-requirements.txt

# Copy the weights to the weights directory
COPY weights/two_towers.pth weights/two_towers.pth
COPY weights/doc_embeddings.pth weights/doc_embeddings.pth

# Copy all python files
COPY *.py .

# Expose the port the app runs on
EXPOSE 60606

# Command to run the FastAPI server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "60606"]
