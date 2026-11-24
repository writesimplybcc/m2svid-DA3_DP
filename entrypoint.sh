#!/bin/bash
# entrypoint.sh - Standard entrypoint for Vast.ai and RunPod containers
# Injects SSH keys if PUBLIC_KEY (Vast.ai / RunPod) or SSH_PUBLIC_KEY is provided

SSH_KEY="${PUBLIC_KEY:-$SSH_PUBLIC_KEY}"
if [ -n "$SSH_KEY" ]; then
    echo "Injecting SSH public key into authorized_keys..."
    mkdir -p ~/.ssh
    echo "$SSH_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
    # Start sshd if installed (for RunPod / Vast.ai direct SSH access)
    if command -v service &> /dev/null; then
        service ssh start 2>/dev/null || true
    fi
fi

# RunPod Compatibility: If RunPod mounted a fresh volume over /workspace,
# restore application files from /opt/stereofaster_app into /workspace/m2svid-DA3_DP
if [ ! -f "/workspace/m2svid-DA3_DP/webui.py" ] && [ -d "/opt/stereofaster_app" ]; then
    echo "📦 Initializing /workspace/m2svid-DA3_DP from image snapshot (RunPod volume detected)..."
    mkdir -p /workspace/m2svid-DA3_DP
    cp -rn /opt/stereofaster_app/. /workspace/m2svid-DA3_DP/
fi

# Execute the CMD passed by the Dockerfile
exec "$@"
