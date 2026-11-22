"""Download the YuNet face detector and verify it before use.

Run: python download_model.py
"""

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
# Pinning a commit means this URL always refers to the same file.
MODEL_COMMIT = "f12e12798e8314f7c074a6656816c048dcc95b7a"
MODEL_URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_COMMIT}"
    f"/models/face_detection_yunet/{MODEL_NAME}"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_SIZE = 232_589
MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_NAME


def sha256_of(path: Path) -> str:
    """A checksum is a fingerprint: any changed byte produces a different value."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == MODEL_SIZE and sha256_of(path) == MODEL_SHA256


def download(path: Path = MODEL_PATH) -> None:
    if is_valid(path):
        print(f"Model already verified: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name so a failed download never looks like a real model.
    temp_path = path.with_suffix(".download")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "model-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(MODEL_SIZE + 1)
        temp_path.write_bytes(data)
        if not is_valid(temp_path):
            raise ValueError("downloaded file failed size or SHA-256 verification")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Model downloaded and verified: {path}")


if __name__ == "__main__":
    try:
        download()
    except (OSError, urllib.error.URLError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
