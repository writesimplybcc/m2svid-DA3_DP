#!/bin/bash
# entrypoint.sh - Standard entrypoint for Vast.ai containers
# Injects SSH keys if PUBLIC_KEY is provided as an environment variable

if [ -n "$PUBLIC_KEY" ]; then
    echo "Injecting PUBLIC_KEY into authorized_keys..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
fi

# Execute the CMD passed by the Dockerfile
exec "$@"
