#!/bin/bash

echo "Starting StereoFaster WebUI and FileBrowser..."

# Start FileBrowser in the background
# -r defines the root directory
# -p defines the port
# -a defines the address (0.0.0.0 to allow external access)
filebrowser -r /workspace/m2svid-DA3_DP -p 8080 -a 0.0.0.0 --noauth &

# Wait a moment to ensure it starts
sleep 2
echo "FileBrowser is running on port 8080"

# Start the Gradio WebUI in the foreground so the container stays alive
echo "Starting Gradio WebUI..."
python webui.py --server-name 0.0.0.0 --server-port 7878
