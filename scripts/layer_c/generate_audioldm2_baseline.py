import torch
from pathlib import Path
from scipy.io.wavfile import write
from diffusers import AudioLDM2Pipeline

OUT = Path("smoke_test/layer_c/generated")
OUT.mkdir(parents=True, exist_ok=True)

SCENES = {
    "forest_bird": "forest birds morning natural ambience, clean environmental sound, no speech, no music, no machine noise",
    "summer_rain": "summer rain natural ambience, clean environmental sound, rain, wind, thunder, no speech, no music, no machine noise"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

for scene, prompt in SCENES.items():
    scene_out = OUT / scene
    scene_out.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        print("generating:", scene, i)

        result = pipe(
            prompt,
            num_inference_steps=30,
            audio_length_in_s=30.0
        )

        audio = result.audios[0]
        out_path = scene_out / f"{scene}_generated_{i:03d}.wav"

        write(out_path, rate=16000, data=audio)
        print("saved:", out_path)