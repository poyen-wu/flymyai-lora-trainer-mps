#!/bin/bash

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --pre --upgrade --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre --upgrade --no-cache-dir torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu128
