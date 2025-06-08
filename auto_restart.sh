#!/bin/bash

while true; do
    echo "Starting script..."
    python3 experiment.py
    echo "Script crashed with exit code $?. Restarting in 5 seconds..."
    sleep 5
done