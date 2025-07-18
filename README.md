# pytorch-by-example
PyTorch by Example

https://pytorch.org/get-started/locally/

macOS - https://developer.apple.com/metal/pytorch/
- python3 -m venv venv
- source venv/bin/activate
- pip3 install --pre --upgrade pip setuptools torch torchvision torchaudio transformers --extra-index-url https://download.pytorch.org/whl/nightly/cpu
- pip3 freeze > requirements-macos.txt

linux
- poetry install
- poetry shell
