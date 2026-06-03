import subprocess
from pathlib import Path

PYTHONPATH = ".;acoustic_ai"

JOBS = [
    {
        "label": "boobook",
        "prompt": "Southern Boobook owl call at night",
        "lora": "model/candidates/baoyuchen/layer-c-audiogen-boobook",
        "out": "eval/generated/boobook",
        "n": 20,
    },
    {
        "label": "raven",
        "prompt": "Australian Raven calling loudly, harsh croaking caw",
        "lora": "model/candidates/baoyuchen/layer-c-audiogen-raven",
        "out": "eval/generated/raven",
        "n": 20,
    },
]

for job in JOBS:
    out_dir = Path(job["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, job["n"] + 1):
        out_path = out_dir / f"{job['label']}_{i:03d}.wav"
        seed = 1000 + i

        cmd = [
            "python",
            "-m",
            "acoustic_ai.layers.layer_c.attempts.lucas__smoke_1__audiogen_boobook.code.sample_audiogen_lora",
            "--prompt",
            job["prompt"],
            "--lora_dir",
            job["lora"],
            "--output_path",
            str(out_path),
            "--device",
            "cpu",
            "--seed",
            str(seed),
        ]

        print("Generating:", out_path)
        subprocess.run(cmd, check=True)