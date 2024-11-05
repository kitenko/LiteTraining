#!/bin/bash

echo "Installing primary dependencies..."

# Install primary dependencies
pip install \
    pytorch-lightning==2.4.0 \
    optuna==3.6.1 \
    jsonargparse==4.33.2 \
    datasets==3.1.0 \
    albumentations==1.4.21 \
    cylimiter==0.4.2

echo "Installing jsonargparse with signatures..."
pip install -U 'jsonargparse[signatures]>=4.27.7'

echo "Installing development tools..."

# Install development tools
pip install \
    pylint==3.3.1 \
    black==24.8.0 \
    pre-commit==3.8.0 \
    mypy==1.11.2 \
    types-pyyaml==6.0.12.20240917 \
    pytest-mock==3.14.0

# Set up pre-commit hooks
pre-commit install

echo "All dependencies have been successfully installed."
