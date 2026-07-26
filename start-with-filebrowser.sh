#!/bin/bash

echo "Starting StereoFaster WebUI and FileBrowser..."

# Start FileBrowser in the background
# -r defines the root directory
# -p defines the port
# -a defines the address (0.0.0.0 to allow external access)
filebrowser -r /workspace/m2svid-DA3_DP -p 7879 -a 0.0.0.0 --noauth &

# Wait a moment to ensure it starts
sleep 2
echo "FileBrowser is running on port 8080"

# Start the Gradio WebUI in the foreground so the container stays alive
echo "Starting Gradio WebUI..."

# Override temporary directory to prevent Vast.ai's tiny 64MB /tmp partition from crashing video uploads
export GRADIO_TEMP_DIR="/workspace/m2svid-DA3_DP/tmp"
export TMPDIR="/workspace/m2svid-DA3_DP/tmp"
mkdir -p /workspace/m2svid-DA3_DP/tmp

python webui.py --server-name 0.0.0.0 --server-port 7878
