#!/usr/bin/env python3
"""
MVP wrapper for Layer C.

Usage:
    python scripts/layer_c/layer_c_generate.py --scene summer_rain --output output.wav

For smoke test, this wrapper calls AudioLDM2 baseline.
Later, backend can call this script or reuse its generate_layer_c_audio function.
"""

import argparse
from pathlib import Path
import scipy.io.wavfile
import torch
from diffusers import AudioLDM2Pipeline

PROMPTS = {
    "summer_rain": "summer rain afternoon, realistic clean outdoor nature ambience, only rain and leaves, no speech, no music, no traffic, no machines",
    "winter_snow": "winter snow night, realistic quiet cold wind and snowfall ambience, no footsteps, no speech, no music, no machines",
    "forest_bird": "forest birds morning, realistic clean forest birds ambience, leaves and light wind, no speech, no music, no traffic, no machines",
}

NEGATIVE_PROMPT = "speech, talking, human voice, music, melody, song, machine, engine, traffic, car, airplane, footsteps, gunshot, explosion, siren, distortion, noise"

def generate_layer_c_audio(scene: str, output: str, model_id: str = "cvssp/audioldm2", duration_sec: float = 30.0):
    if scene not in PROMPTS:
        raise ValueError(f"Unknown scene: {scene}. Choose from {list(PROMPTS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    result = pipe(
        PROMPTS[scene],
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=50,
        audio_length_in_s=duration_sec,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(out_path, rate=16000, data=result.audios[0])
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(PROMPTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_id", default="cvssp/audioldm2")
    args = parser.parse_args()

    path = generate_layer_c_audio(args.scene, args.output, args.model_id)
    print(f"Layer C generated audio: {path}")

if __name__ == "__main__":
    main()
