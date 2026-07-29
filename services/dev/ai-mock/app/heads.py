"""Per-head report slices for POST /layers/{layer}/attempts/{attempt}/analyze.

Each Layer E head returns a different shape on /dev/analysis, so the canned
bundle is sliced rather than returned whole. The aggregator head has no
per-head card in the UI — it is only used by "Run Full Analysis" — but it is
handled here so a direct call still answers sensibly.
"""

from __future__ import annotations


def report_for(head: str | None, bundle: dict) -> dict:
    reports = bundle.get("head_reports", {})
    if head == "ambient":
        return reports.get("ambient", {})
    if head == "weather":
        return reports.get("weather", {})
    if head == "events":
        return reports.get("events", {})
    if head == "aggregator":
        return bundle.get("report", {})
    # Attempts with no declared head (e.g. layer_e/lucas__smoke_1__detectors)
    # are stubs in the real registry too; answer with the ambient view so the
    # route never 500s.
    return reports.get("ambient", {})
