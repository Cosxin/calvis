FROM python:3.11-slim

WORKDIR /app

# System deps (libgl1 + libglib2.0-0 required by opencv-python, pulled in by grad-cam)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller than CUDA wheel: ~200MB vs ~800MB)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SmolVLM-256M weights at build time
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'HuggingFaceTB/SmolVLM-256M-Instruct', \
    local_dir='/app/checkpoints/smolvlm-256m', \
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*','rust_model*'] \
)"

# Patch preprocessor_config.json — older snapshots lack image_processor_type,
# which causes AutoProcessor and Idefics3Processor to fail on newer transformers.
RUN python3 -c "\
import json, os; \
p = '/app/checkpoints/smolvlm-256m/preprocessor_config.json'; \
cfg = json.load(open(p)); \
cfg.setdefault('image_processor_type', 'Idefics3ImageProcessor'); \
json.dump(cfg, open(p,'w'), indent=2); \
print('preprocessor_config image_processor_type:', cfg['image_processor_type']) \
"

# Copy app
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860
CMD ["python3", "app.py"]
