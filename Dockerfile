FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.1 — falls back to CPU automatically if no GPU present
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY *.py ./
COPY bst_atlases/ bst_atlases/
COPY BrainSuite/bin/linux/ BrainSuite/bin/linux/

# Make BrainSuite binaries executable
RUN chmod +x BrainSuite/bin/linux/*

ENTRYPOINT ["python3", "auto_resection_mask.py"]