# fakinflu — worker RunPod : tts-higgs (Higgs TTS 3, poids BAKES)
# Build par RunPod (integration GitHub) a chaque push sur main.
#
# Base = image vLLM officielle en CUDA 12.9 (la variante par defaut est en CUDA 13, que peu
# d'hotes RunPod acceptent), + vllm-omni a la meme version, comme le fait docker/Dockerfile.cuda
# de vllm-omni. Le fichier UV_OVERRIDE de la base garde torch/flashinfer coherents.
ARG VLLM_TAG=v0.30.0-cu129
FROM vllm/vllm-openai:${VLLM_TAG}

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0

WORKDIR /app

# L'image de base est incoherente : torch y est compile pour CUDA 13.0, torchaudio pour 12.9
# (transformers importe torchaudio -> "compiled with different CUDA versions" au demarrage).
# On realigne tout le trio sur cu129, aux versions qu'exige vllm 0.30, puis on les fige pendant
# l'installation de vllm-omni. Un hote CUDA >= 12.9 suffit alors (minCudaVersion de l'endpoint).
ARG TORCH_PINS="torch==2.13.0+cu129 torchvision==0.28.0+cu129 torchaudio==2.11.0+cu129"
RUN PY="$(python3 -c 'import sys; print(sys.executable)')" \
 && uv pip install --python "$PY" --no-cache-dir --reinstall-package torch --reinstall-package torchvision \
      --reinstall-package torchaudio --index-url https://download.pytorch.org/whl/cu129 \
      --extra-index-url https://pypi.org/simple ${TORCH_PINS} \
 && python3 -c "import torch, torchaudio, torchvision; assert torch.version.cuda.startswith('12.9'), torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)"

ARG VLLM_OMNI_VERSION=0.30.0
RUN PY="$(python3 -c 'import sys; print(sys.executable)')" \
 && printf '%s\n' ${TORCH_PINS} > /tmp/torch-pins.txt \
 && uv pip install --python "$PY" --no-cache-dir -c /tmp/torch-pins.txt \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      "vllm-omni==${VLLM_OMNI_VERSION}" "runpod>=1.7,<2" soundfile \
 && python3 -c "import torch, torchaudio, transformers; assert torch.version.cuda.startswith('12.9'), torch.version.cuda; print('ok', torch.__version__, torchaudio.__version__, transformers.__version__)"

# Poids BAKES (~10 Go) : talker + codec embarque (bosonai/higgs-tts-3-4b), et l'encodeur du
# codec pour le clonage (k2-fsa/OmniVoice, sous-dossier audio_tokenizer, ~0,8 Go).
ARG MODEL_ID=bosonai/higgs-tts-3-4b
ENV MODEL_ID=${MODEL_ID}
RUN python3 -c "from huggingface_hub import snapshot_download as d; \
d('${MODEL_ID}', allow_patterns=['*.json', '*.safetensors', '*.jinja', 'LICENSE']); \
d('k2-fsa/OmniVoice', allow_patterns=['audio_tokenizer/*'])"

COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# L'image vLLM a "vllm" pour ENTRYPOINT : sans cette remise a zero, CMD devient
# "vllm python3 -u handler.py" -> "unrecognized arguments" (v1.0.2).
ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
