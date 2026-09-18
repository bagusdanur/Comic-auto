#!/usr/bin/env python3
"""Download and verify the pinned dynamic LaMa ONNX artwork model."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"
SHA256 = "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("models/lama.onnx"))
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() and digest(destination) == SHA256:
        print(f"Verified: {destination}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "ryukomik-hybrid/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = digest(temporary)
    if actual != SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch: {actual}")
    temporary.replace(destination)
    print(f"Installed and verified: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

