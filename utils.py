import os
import urllib.request

def download_weights(path, url):
    if not os.path.exists(path):
        print(f"Downloading weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
