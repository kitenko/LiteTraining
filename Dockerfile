FROM ubuntu:22.04

# Set the working directory
WORKDIR /app

COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl

# Install Pixi (specific version)
RUN curl -fsSL https://pixi.sh/install.sh | bash -s -- --version 0.41.3

# Make Pixi available in $PATH
ENV PATH="/root/.pixi/bin:$PATH"

# Defining an argument for selecting an environment with the default value "default"
ARG ENVIRONMENT=default

# We use the passed argument to set the environment via pixi
RUN pixi install -e $ENVIRONMENT
