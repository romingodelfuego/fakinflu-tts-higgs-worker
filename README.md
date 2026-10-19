# Worker RunPod : tts-higgs

**Higgs TTS 3** (Boson AI, ~4B, 100+ langues dont le français, 24 kHz) : synthèse expressive
pilotée par **balises en ligne**, avec clonage de voix zero-shot. Servi en inférence hors ligne par
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) 0.30.0.

Pile séparée du worker `tts` (Chatterbox / Qwen / VoxCPM2) : vLLM, CUDA 12.9, ~85 % de la VRAM.

## Contrat d'API

Identique au worker `tts`, moteur `higgs` :

```json
{"input": {
  "voices": {"fr_f": {"ref_audio": "<b64 wav>", "ref_text": "transcription exacte (améliore le clone)"}},
  "items": [
    {"id": "1", "text": "<|emotion:bitterness|><|prosody:speed_slow|>Tu m'as menti. <|prosody:pause|>Depuis le début.",
     "voice": {"mode": "clone", "ref": "fr_f"}},
    {"id": "2", "text": "<|style:whispering|>Ne fais pas de bruit.", "voice": {"mode": "default"}}
  ]}}
→ {"results": [{"id": "1", "audio": "<b64 wav>", "sr": 24000, "duration_s": 3.2, "seconds": 0.9, "model": "...", "engine": "higgs"}]}
```

## Balises

| Où | Balises |
|---|---|
| En tête | `<\|emotion:X\|>` (elation, amusement, enthusiasm, determination, pride, contentment, affection, relief, contemplation, confusion, surprise, awe, longing, arousal, anger, fear, disgust, bitterness, sadness, shame, helplessness) · `<\|style:singing\|shouting\|whispering\|>` · `<\|prosody:speed_very_slow\|speed_slow\|speed_fast\|speed_very_fast\|>` · `<\|prosody:pitch_low\|pitch_high\|>` · `<\|prosody:expressive_high\|expressive_low\|>` |
| À l'endroit voulu | `<\|prosody:pause\|>` (~0,5 s) · `<\|prosody:long_pause\|>` (~1 s) · `<\|sfx:X\|>` + onomatopée : cough, laughter (`Haha`), crying, screaming, burping, humming, sigh (`Pff`), sniff, sneeze |

Pas de paramètre de langue. Les paramètres d'échantillonnage sont ceux de la config de
déploiement vLLM-Omni (température 1.0, top_p 0.95, top_k 50, seed 42).

## Licence

Poids sous **Boson Higgs TTS 3 Research and Non-Commercial License**. La *Creator Use Grant*
autorise les contenus monétisés (vidéos, podcasts, réseaux sociaux) à condition de créditer
« Boson AI's Higgs Audio » de façon visible ; elle exclut l'exposition du modèle comme API ou
service à des tiers. Ce worker sert au bench interne.

## Cycle de vie

Voir `make worker-status W=tts-higgs` / `make worker-bump` / `make worker-release` (racine fakinflu).
Réglages de l'endpoint dans `worker.json` (dont `min_cuda` : 12.9).
