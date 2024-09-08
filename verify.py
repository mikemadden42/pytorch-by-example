#!/usr/bin/env python3

# https://developer.apple.com/metal/pytorch/

import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
x = torch.ones(1, device=device)
print(x)
