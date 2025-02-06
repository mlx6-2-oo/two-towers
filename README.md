# two-towers

# Project Setup

This repository includes a setup script to help you get started quickly.

## Prerequisites

- If you are on Mac OS, make sure you have brew installed (try executing `brew --version` to check)
- Docker for running the search server

## Usage Instructions

- To prepare your terminal for development on Linux or Mac OS, execute the following command:
   
   ```
   source setup.sh
   ```

    This script will:
   - Install pyenv to manage Python versions
   - Install the Python version specified in the setup script
   - Create a virtual environment and activate it
   - Install the dependencies specified in `requirements.txt`

- To share Python package dependency requirements with others:
  - For development: `pip freeze > requirements.txt`
  - For deployment: Update `requirements.docker.txt` with minimal dependencies

## Training Instructions

1. Generate embeddings only:
```bash
python -m src.data.generate_embeddings
```
2. Train with existing embeddings
```bash
python -m src.training.train
python -m src.training.train --wandb # save results to wandb
```
3. Run the full pipeline
```bash
python run_pipeline.py
python run_pipeline.py --wandb # save results to wandb
```

## Deployment Instructions

1. Build the Docker image:
```bash
docker build -t two-towers .
```

2. Run the search server:
```bash
docker run -p 8000:8000 two-towers
```

3. In a separate terminal, run the client:
```bash
# this will run it against a remote server
python -m src.client.cli --server <IP>:8000
# this will default to localhost
python -m src.client.cli
```

The server will be available at `http://localhost:8000` and the client will automatically connect to it.

Note: The Docker container uses a minimal set of dependencies defined in `requirements.docker.txt`, while the development environment uses the full set in `requirements.txt`.
