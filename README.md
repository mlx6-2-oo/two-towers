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

## Local Deployment Instructions

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

# Remote Deployment Instructions

```bash
docker build --platform linux/amd64 -t docker_hub_account_name/project_name .
docker push docker_hub_account_name/project_name
ssh root@ip
docker pull docker_hub_account_name/project_name
docker run -dp 8000:8000 docker_hub_account_name/project_name
# to kill
docker container ls
docker stop <id>
docker rm <ic>
```

# Frontend Deployment

The project includes a web-based terminal interface in the `frontend/` directory.

## Local Frontend Development

1. Build the frontend Docker image:
```bash
cd frontend
docker build -t two-towers-frontend .
```

2. Run the frontend container:
```bash
docker run -dp 80:80 two-towers-frontend
```

The terminal interface will be available at:
- Local development: http://localhost
- Remote server: http://your-server-ip

## Full Stack Deployment

To deploy both backend and frontend on a server:

1. Deploy backend:
```bash
# Build and push backend
docker build --platform linux/amd64 -t your-repo/two-towers-backend .
docker push your-repo/two-towers-backend

# On your server
docker pull your-repo/two-towers-backend
docker run -dp 8000:8000 your-repo/two-towers-backend
```

2. Deploy frontend:
```bash
# Build and push frontend
cd frontend
docker build --platform linux/amd64 -t your-repo/two-towers-frontend .
docker push your-repo/two-towers-frontend

# On your server
docker pull your-repo/two-towers-frontend
docker run -dp 80:80 your-repo/two-towers-frontend
```

Note: Before deploying the frontend, make sure to update `SERVER_URL` in `frontend/app.js` to point to your backend server's address.
