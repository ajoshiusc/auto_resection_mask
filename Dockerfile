FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

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
            torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
            --index-url https://download.pytorch.org/whl/cu130; \
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

# Copy both BrainSuite binary sets and make them executable 
COPY BrainSuite/bin/linux_amd64/ BrainSuite/bin/linux_amd64/
COPY BrainSuite/bin/linux_arm64/ BrainSuite/bin/linux_arm64/
RUN chmod +x /app/BrainSuite/bin/linux_amd64/* /app/BrainSuite/bin/linux_arm64/*

ENTRYPOINT ["python3", "auto_resection_mask.py"]