#!/bin/bash

# Check if the zip file argument is provided
if [ -z "$1" ]; then
  echo "Usage: bash $0 <zip_file_path>"
  exit 1
fi

# define the zip file and the destination directory
zip_file="$1"
destination_dir="data/raw"

# create the destination directory if it does not exist
mkdir -p "$destination_dir"

# Unzip the zip file into the destination directory without directories
unzip -j "$zip_file" -d "$destination_dir"
