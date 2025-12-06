#!/usr/bin/env python3

import torch
import torchaudio
import torchvision


def versions():
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"torchvision: {torchvision.__version__}")


if __name__ == "__main__":
    versions()
