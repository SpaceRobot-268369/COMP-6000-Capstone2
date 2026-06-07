"""Registry-style handler for the Layer E analysis aggregator.

This handler does not run audio inference. It combines already-computed E-A,
E-B, and E-C reports into the fused Analysis Aggregator v1 schema.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs.
    yaml = None

try:
    from .aggregator import aggregate_reports
except ImportError:  # pragma: no cover - direct script execution fallback.
    from aggregator import aggregate_reports


PARAMS_PATH = Path(__file__).resolve().parents[1] / "params.yaml"


@dataclass(frozen=True)
class AggregatorState:
    params: dict[str, Any]


def load(
    checkpoint_dir: Path | None = None,
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> AggregatorState:
    """Load static fusion params.

    ``checkpoint_dir`` and ``extra`` are accepted for registry compatibility;
    the aggregator has no model checkpoint.
    """
    del checkpoint_dir, extra
    merged = load_params()
    if params:
        merged = _merge_dicts(merged, params)
    return AggregatorState(params=merged)


def aggregate(
    state: AggregatorState,
    *,
    ambient_report: dict[str, Any] | None = None,
    weather_report: dict[str, Any] | None = None,
    events_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse three head reports into the final analysis report."""
    return aggregate_reports(
        ambient_report=ambient_report,
        weather_report=weather_report,
        events_report=events_report,
        params=state.params,
    )


def generate(state: AggregatorState, seed: int | None = None, **_ignored: Any) -> dict[str, Any]:
    del state, seed
    raise NotImplementedError(
        "Layer E Aggregator is report-based, not seed/audio generation based. "
        "Use aggregate(state, ambient_report=..., weather_report=..., events_report=...)."
    )


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse Layer E head reports.")
    parser.add_argument(
        "input_json",
        type=Path,
        help="JSON file with ambient_report, weather_report, and events_report keys.",
    )
    parser.add_argument("--out", type=Path, help="Output JSON path. Prints to stdout if omitted.")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input_json must contain a JSON object")

    state = load()
    result = aggregate(
        state,
        ambient_report=payload.get("ambient_report"),
        weather_report=payload.get("weather_report"),
        events_report=payload.get("events_report"),
    )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
