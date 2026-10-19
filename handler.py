#!/usr/bin/env python3
"""fakinflu — worker RunPod : tts-higgs (Higgs TTS 3, Boson AI, ~4B, 100+ langues, 24 kHz).

Meme contrat que le worker tts (tts/serverless), moteur "higgs" :
  {"voices": {"<voice_id>": {"ref_audio": "<b64 wav>", "ref_text": "..."}},   # optionnel
   "items": [{"id": "...", "text": "<|emotion:anger|>Tu m'as menti. <|prosody:pause|>...",
              "voice": {"mode": "clone", "ref": "<voice_id>"} | {"mode": "default"}}]}
Sortie : {"results": [{"id", "audio": "<b64 wav>", "sr", "duration_s", "seconds", "model"} | {"id", "error"}]}

Le jeu passe par des balises EN LIGNE dans "text" (le client les ecrit) :
  en tete  : <|emotion:...|> <|style:whispering|> <|prosody:speed_slow|> <|prosody:expressive_high|>
  en place : <|prosody:pause|> <|prosody:long_pause|> <|sfx:laughter|>Haha <|sfx:sigh|>Pff
Pas de parametre de langue : le modele la deduit du texte.

Inference hors ligne via vLLM-Omni (Omni : talker Qwen3 -> codec), chargee une fois au cold
start. Tous les items d'un job partent en un seul lot (batching continu de vLLM).
Licence du modele : recherche / non commerciale, + "Creator Use Grant" (contenus monetises
autorises en creditant "Boson AI's Higgs Audio"). Voir LICENSE du modele.
"""
import base64, glob, os, tempfile, time, traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # poids bakes : jamais de telechargement au runtime
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import numpy as np
import runpod
import soundfile as sf
import torch
from transformers import AutoTokenizer
from vllm_omni import Omni
from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
    HiggsAudioV3TokenizerAdapter, apply_delay_pattern, encode_reference_audio)

MODEL_ID = os.environ.get("MODEL_ID", "bosonai/higgs-tts-3-4b")
SR = 24_000


def _local(repo):
    """Dossier du snapshot bake (un chemin local evite tout appel a l'API HF)."""
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*")))
    return snaps[-1] if snaps else repo


# --- chargement unique au cold start ------------------------------------------
_t = time.time()
MODEL_PATH = _local(MODEL_ID)
engine = Omni(model=MODEL_PATH, deploy_config=os.environ.get("HIGGS_DEPLOY_CONFIG") or None,
              trust_remote_code=True)
adapter = HiggsAudioV3TokenizerAdapter(AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True))
print(f"[tts-higgs] modele pret en {time.time() - _t:.1f}s ({MODEL_PATH})", flush=True)

_refs = {}  # chemin du wav de reference -> codes du codec (delay pattern applique)


def _b64_to_file(b64, path):
    if b64.strip().startswith("data:") and "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def _ref_codes(path):
    if path not in _refs:
        wav, sr = sf.read(path, always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        _refs[path] = apply_delay_pattern(encode_reference_audio(wav, int(sr)))
    return _refs[path]


def _prompt(item, voices):
    text = (item.get("text") or "").strip()
    if not text:
        raise ValueError("champ 'text' vide")
    voice = item.get("voice") or {"mode": "default"}
    mode = voice.get("mode", "default")
    if mode == "default":
        return {"prompt_token_ids": adapter.build_prompt(text)}
    if mode != "clone":
        raise ValueError(f"higgs : mode de voix non gere : {mode} (clone | default)")
    ref = voice.get("ref")
    if ref not in voices:
        raise ValueError(f"voix de reference inconnue : {ref!r} (bloc 'voices' du job)")
    codes = _ref_codes(voices[ref]["path"])
    ids = adapter.build_prompt(text, num_ref_tokens=int(codes.shape[0]),
                               reference_text=voices[ref].get("ref_text") or None)
    ids, positions = adapter.prepare_prompt_for_engine(ids)
    return {"prompt_token_ids": ids, "additional_information": {
        "audio_input_ids": codes.to(torch.long),
        "audio_input_ids_mask": torch.ones(codes.shape[0], dtype=torch.bool),
        "audio_placeholder_positions": positions}}


def _pcm(output):
    mm = output.outputs[0].multimodal_output
    audio = mm.get("model_outputs")
    if audio is None:
        audio = mm.get("audio")
    if audio is None:
        raise ValueError(f"pas d'audio dans la sortie : {list(mm)}")
    if isinstance(audio, list):
        parts = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not parts:
            raise ValueError("sortie audio vide")
        audio = torch.cat(parts)
    arr = torch.as_tensor(audio).float().cpu().reshape(-1).numpy()
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)


def _encode(pcm, seconds):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, pcm, SR, format="WAV", subtype="PCM_16")
        audio = base64.b64encode(open(f.name, "rb").read()).decode()
    return {"audio": audio, "sr": SR, "duration_s": round(len(pcm) / SR, 2), "seconds": seconds,
            "model": os.path.basename(MODEL_ID), "engine": "higgs"}


def handler(job):
    inp = job.get("input", {}) or {}
    if inp.get("ping"):
        return {"pong": True, "engines": ["higgs"]}
    items = inp.get("items") or [inp]
    results = [{"id": it.get("id")} for it in items]
    with tempfile.TemporaryDirectory() as tmp:
        voices = {}
        for vid, v in (inp.get("voices") or {}).items():
            path = os.path.join(tmp, f"ref_{len(voices)}.wav")
            _b64_to_file(v["ref_audio"], path)
            voices[vid] = {"path": path, "ref_text": v.get("ref_text") or ""}
        todo = []  # (index, prompt)
        for i, it in enumerate(items):
            try:
                todo.append((i, _prompt(it, voices)))
            except Exception as e:
                traceback.print_exc()
                results[i]["error"] = f"{type(e).__name__}: {e}"
        if todo:
            t = time.time()
            try:
                outs = engine.generate([p for _, p in todo])
                per_item = round((time.time() - t) / len(todo), 2)  # lot : temps moyen par item
                for (i, _), out in zip(todo, outs):
                    try:
                        results[i].update(_encode(_pcm(out), per_item))
                    except Exception as e:
                        traceback.print_exc()
                        results[i]["error"] = f"{type(e).__name__}: {e}"
            except Exception as e:
                traceback.print_exc()
                for i, _ in todo:
                    results[i]["error"] = f"{type(e).__name__}: {e}"
        _refs.clear()  # les chemins temporaires meurent avec le job
    return {"results": results}


runpod.serverless.start({"handler": handler})
