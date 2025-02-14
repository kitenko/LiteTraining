#!/bin/bash

# Variables
ENVIRONMENT="default"  # Specify the pixi environment to use

# Check if at least two arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: you need to specify both a mode (train, val, test) and a config file."
  echo "Usage: ./run_training.sh <mode> <config_file>"
  exit 1
fi

MODE=$1          # Mode (train, val, test)
CONFIG_PATH=$2   # Path to the configuration file

# Execute the appropriate command based on the mode using pixi run
echo "Running main.py in $MODE mode with config file $CONFIG_PATH using pixi run..."

case "$MODE" in
  train)
    pixi run --environment "$ENVIRONMENT" python3 main.py --config "$CONFIG_PATH"
    ;;
  val)
    pixi run --environment "$ENVIRONMENT" python3 main.py val --config "$CONFIG_PATH"
    ;;
  test)
    pixi run --environment "$ENVIRONMENT" python3 main.py test --config "$CONFIG_PATH"
    ;;
  *)
    echo "Error: unknown mode $MODE. Use train, val, or test."
    exit 1
    ;;
esac

# Completion message
echo "Script completed successfully."
