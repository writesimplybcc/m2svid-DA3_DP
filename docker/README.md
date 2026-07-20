# StereoFaster Docker Setup

This folder contains a starter Docker configuration for running the modern **DA3 + M2SVid 2-step pipeline** and the **Gradio WebUI**.

## Quick Start

```bash
# 1. Build the image
docker compose build

# 2. Run the web UI (with GPU)
docker compose up

# Open http://localhost:7860
```

## Files

- `Dockerfile` — Multi-stage friendly CUDA + Python environment
- `docker-compose.yml` — Convenient GPU + volume mounting setup
- `.dockerignore` — Keeps image size reasonable

## Important Notes

- **Models are large**: Mount your Hugging Face cache and `m2svid/ckpts` folder.
- **First run** will download DA3 models (~6-15 GB depending on variant) and M2SVid weights.
- The container expects the project structure at `/app`.

## Volume Mounts (recommended)

Edit `docker-compose.yml` to point to your local folders:

```yaml
volumes:
  - /path/to/your/videos:/app/data
  - /path/to/your/outputs:/app/outputs
  - ~/.cache/huggingface:/root/.cache/huggingface
  - ./m2svid/ckpts:/app/m2svid/ckpts
```

## Running without docker-compose

```bash
docker build -t stereofaster .
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/m2svid/ckpts:/app/m2svid/ckpts \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  stereofaster
```

## CUDA / PyTorch Version

Current image uses:
- CUDA 12.1 + cuDNN 8
- PyTorch 2.3+ with CUDA 12.1

Adjust the base image and torch index in the Dockerfile if you need a different CUDA version.

## Next Steps / Customization

- Add a healthcheck for the Gradio server
- Multi-stage build to reduce final image size
- Support for DA3 backend server + webui separately
- Add nginx reverse proxy for production
