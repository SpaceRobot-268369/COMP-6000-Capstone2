"""Run the Layer C retrieval baseline and write a debug bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from event_index import DEFAULT_OUTPUT
from retriever import EventRetriever
from scheduler import EventScheduler


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_OUT_DIR = REPO_ROOT / "debug" / "layer_c" / "retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Layer C event retrieval baseline.")
    parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--species", default="Horsfield's Bronze-cuckoo")
    parser.add_argument("--diel-bin", default="morning")
    parser.add_argument("--season", default="summer")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--enable-variation",
        action="store_true",
        help="Apply small pitch/time-stretch variation. Off by default for smoke reliability.",
    )
    parser.add_argument(
        "--ecological",
        action="store_true",
        help="Use species-specific bout spacing instead of simple uniform gaps.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR / f"{slugify(args.species)}_seed{args.seed}"

    retriever = EventRetriever(args.index)
    selected = retriever.retrieve(
        species=args.species,
        diel_bin=args.diel_bin,
        season=args.season,
        count=args.count,
        seed=args.seed,
    )

    scheduler = EventScheduler(
        target_duration_s=args.duration,
        seed=args.seed,
        enable_variation=args.enable_variation,
        ecological_mode=args.ecological,
    )
    events = scheduler.schedule(selected)
    result = scheduler.render(events)
    scheduler.write_debug_bundle(
        result,
        out_dir=out_dir,
        request={
            "species": args.species,
            "diel_bin": args.diel_bin,
            "season": args.season,
            "seed": args.seed,
            "requested_count": args.count,
            "scheduled_count": len(events),
            "enable_variation": args.enable_variation,
            "ecological_mode": args.ecological,
        },
    )

    print(f"Written Layer C retrieval bundle to {out_dir}")
    print(f"Scheduled {len(events)} events")
    for event in events:
        snippet = event.retrieved.snippet
        print(
            f"- {event.onset_s:.1f}s {snippet.audio_event_id} "
            f"{snippet.diel_bin}/{snippet.season} score={snippet.score:.4f}"
        )


if __name__ == "__main__":
    main()
