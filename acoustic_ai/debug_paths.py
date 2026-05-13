"""Shared debug artifact locations.

Runtime APIs should keep audio in memory. Development and smoke-test scripts
write inspection artifacts under the project-level ``debug/`` tree.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_ROOT = PROJECT_ROOT / "debug"
