#!/usr/bin/env python3
"""
Generate baseline Layer C audio with AudioLDM2.

Install:
    pip install torch diffusers transformers accelerate scipy

Run:
    python scripts/layer_c/generate_audioldm2_baseline.py --output_dir smoke_test/layer_c/generated --num_per_scene 5
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="smoke_test/layer_c/generated")
    parser.add_argument("--num_per_scene", type=int, default=5)
    parser.add_argument("--model_id", default="cvssp/audioldm2")
    parser.add_argument("--duration_sec", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AudioLDM2Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene, prompt in PROMPTS.items():
        scene_dir = output_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, args.num_per_scene + 1):
            generator = torch.Generator(device=device).manual_seed(1000 + i)
            result = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=args.steps,
                audio_length_in_s=args.duration_sec,
                generator=generator,
            )
            audio = result.audios[0]
            out_path = scene_dir / f"gen_{scene}_{i:03d}.wav"
            scipy.io.wavfile.write(out_path, rate=16000, data=audio)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
