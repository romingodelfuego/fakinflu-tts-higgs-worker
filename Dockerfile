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

# torch/torchaudio/torchvision de la base (+cu129) figes en contraintes : sans ca, l'installation
# de vllm-omni remplace torch par la roue PyPI CUDA 13.0 et laisse torchaudio en 12.9 -> crash a
# l'import ("PyTorch and TorchAudio were compiled with different CUDA versions", v1.0.0).
ARG VLLM_OMNI_VERSION=0.30.0
RUN PY="$(python3 -c 'import sys; print(sys.executable)')" \
 && python3 -c "import torch, torchaudio, torchvision as tv; print(f'torch=={torch.__version__}'); print(f'torchaudio=={torchaudio.__version__}'); print(f'torchvision=={tv.__version__}')" > /tmp/torch-pins.txt \
 && cat /tmp/torch-pins.txt \
 && uv pip install --python "$PY" --no-cache-dir -c /tmp/torch-pins.txt \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      "vllm-omni==${VLLM_OMNI_VERSION}" "runpod>=1.7,<2" soundfile \
 && python3 -c "import torch, torchaudio; assert torch.version.cuda.startswith('12.9'), torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)"

# Poids BAKES (~10 Go) : talker + codec embarque (bosonai/higgs-tts-3-4b), et l'encodeur du
# codec pour le clonage (k2-fsa/OmniVoice, sous-dossier audio_tokenizer, ~0,8 Go).
ARG MODEL_ID=bosonai/higgs-tts-3-4b
ENV MODEL_ID=${MODEL_ID}
RUN python3 -c "from huggingface_hub import snapshot_download as d; \
d('${MODEL_ID}', allow_patterns=['*.json', '*.safetensors', '*.jinja', 'LICENSE']); \
d('k2-fsa/OmniVoice', allow_patterns=['audio_tokenizer/*'])"

COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

CMD ["python3", "-u", "handler.py"]
