#!/bin/bash

# Check if path is specified as an argument
if [ -z "$1" ]; then
  echo "Please specify the path to the folder containing the data."
  exit 1
fi

# Check if the path exists
if [ ! -d "$1" ]; then
  echo "The specified path does not exist or is not a directory."
  exit 1
fi

# Check if gender is specified as an argument
if [ -z "$1" ]; then
  echo "Please specify the path to the folder containing the data."
  exit 1
fi

path=$(readlink -f "$1")

# Run SAM
python scripts/custom/run-sam.py --data_dir $path
python scripts/custom/extract-largest-connected-components.py --data_dir $path
