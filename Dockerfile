FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# TARGETARCH is set automatically by buildx: "amd64" or "arm64"
ARG TARGETARCH

# amd64: CUDA-enabled PyTorch | arm64: CPU-only PyTorch (no CUDA wheels exist for arm64)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --no-cache-dir \
            torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir \
            torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib -name "*.pyc" -delete \
    && find /usr/local/lib -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy source code
COPY *.py ./
COPY bst_atlases/ bst_atlases/
COPY BrainSuite/bin/linux/ BrainSuite/bin/linux/

# Make BrainSuite binaries executable
RUN chmod +x BrainSuite/bin/linux/*

ENTRYPOINT ["python3", "auto_resection_mask.py"]