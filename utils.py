import os
import urllib.request

import torch


def download_weights(path, url):
    if not os.path.exists(path):
        print(f"Downloading weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
