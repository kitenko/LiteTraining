FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \

# Install Pixi (specific version)
RUN curl -fsSL https://pixi.sh/install.sh | bash -s -- --version 0.41.3

# Make Pixi available in $PATH
ENV PATH="/root/.pixi/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy code into the container (if needed)
COPY . /app

# Default command for the container (launches bash)
CMD ["/bin/bash"]
