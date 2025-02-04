# two-towers

# Project Setup

This repository includes a setup script to help you get started quickly.

## Prerequisites

- If you are on Mac OS, make sure you have brew installed (try executing `brew --version` to check)

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

- To share Python package dependency requirements with others, run `pip freeze > requirements.txt` and commit any changes to `requirements.txt` to the repo

## Training Instructions

1. Generate embeddings only:
```bash
python -m src.data.generate_embeddings
```
2. Train with existing embeddings
```bash
python train.py
```
3. Run the full pipeline
```bash
python run_pipeline.py
```
