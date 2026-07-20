# =============================================================================
# StereoFaster - Docker Image
# Modern DA3 + M2SVid 2-step pipeline + Gradio WebUI
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Install PyTorch with CUDA 12.1
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    torchvision==0.18.0+cu121 \
    torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# =============================================================================
# Install project requirements
# =============================================================================
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Install DA3 with app extras (includes Gradio + addict, etc.)
# =============================================================================
COPY Depth-Anything-3 ./Depth-Anything-3
RUN cd Depth-Anything-3 && pip install -e ".[app]" --no-cache-dir

# Optional: Install gsplat for 3DGS features (can be heavy)
# RUN pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

# =============================================================================
# Copy the rest of the project
# =============================================================================
COPY . .

# =============================================================================
# Set PYTHONPATH for subprojects
# =============================================================================
ENV PYTHONPATH="/app:/app/Depth-Anything-3/src:/app/m2svid/third_party/Hi3D-Official:/app/m2svid/third_party/pytorch-msssim:${PYTHONPATH}"

# =============================================================================
# Prepare directories
# =============================================================================
RUN mkdir -p /app/outputs /app/data /app/m2svid/ckpts

# =============================================================================
# Expose Gradio port
# =============================================================================
EXPOSE 7860

# =============================================================================
# Default command: Launch the WebUI
# =============================================================================
CMD ["python", "webui.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
