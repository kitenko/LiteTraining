#!/bin/bash

# Variables
CONTAINER_NAME="LightningClassifier"

# Check if at least two arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: you need to specify both a mode (train, val, test) and a config file."
  echo "Usage: ./run_training.sh <mode> <config_file>"
  exit 1
fi

MODE=$1          # Mode (train, val, test)
CONFIG_PATH=$2   # Path to the configuration file

# Step 1: Check if Docker and docker-compose are installed
if ! command -v docker &> /dev/null; then
  echo "Error: Docker is not installed. Please install Docker and try again."
  exit 1
fi

if ! command -v docker-compose &> /dev/null; then
  echo "Error: docker-compose is not installed. Please install docker-compose and try again."
  exit 1
fi

# Step 2: Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
  echo "Container $CONTAINER_NAME is already running."
else
  echo "Building and starting Docker container..."
  docker-compose up --build -d
fi

# Step 3: Execute the appropriate command based on the mode
echo "Running main.py in $MODE mode with config file $CONFIG_PATH..."

case "$MODE" in
  train)
    docker exec -it $CONTAINER_NAME bash -c "python3 main.py --config $CONFIG_PATH"
    ;;
  val)
    docker exec -it $CONTAINER_NAME bash -c "python3 main.py val --config $CONFIG_PATH"
    ;;
  test)
    docker exec -it $CONTAINER_NAME bash -c "python3 main.py test --config $CONFIG_PATH"
    ;;
  *)
    echo "Error: unknown mode $MODE. Use train, val, or test."
    exit 1
    ;;
esac

# Completion message
echo "Script completed successfully."
