#!/bin/bash
# Simple helper to build and run StereoFaster with Docker
set -e

echo "=== Building StereoFaster Docker image ==="
docker compose -f docker/docker-compose.yml build

echo ""
echo "=== Starting StereoFaster (WebUI on http://localhost:7860) ==="
docker compose -f docker/docker-compose.yml up
