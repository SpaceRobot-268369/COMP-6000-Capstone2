"""Skill loader.

Skills are the per-job instruction sets, authored as **separate markdown
files** in this directory (the static system message for each LLM job). The
dynamic job payload (raw prompt / fused JSON / gate findings) is passed as the
user message at call time — never baked into the file. See
.claude/context/ai/llm_layer_implementation_plan.md §2.1.

The current `*.md` files are PLACEHOLDER stubs (owner: Lucas authors the real
skills later). Wiring runs against these stubs and swaps in the authored
versions without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent
_VALID_REGISTERS = {"analytical", "immersive"}


class SkillNotFound(FileNotFoundError):
    """Raised when a requested skill file does not exist."""


@lru_cache(maxsize=None)
def load_skill(name: str) -> str:
    """Read + cache a skill file (`<name>.md`) and return it as the system
    message. Cached for the process lifetime; restart to pick up edits."""
    path = _SKILLS_DIR / f"{name}.md"
    if not path.is_file():
        raise SkillNotFound(f"skill not found: {name!r} (looked in {path})")
    return path.read_text(encoding="utf-8").strip()


def report_skill_name(register: str) -> str:
    """Map an analysis register to its skill file name."""
    reg = (register or "analytical").lower()
    if reg not in _VALID_REGISTERS:
        reg = "analytical"
    return f"report_{reg}"


def available_skills() -> list[str]:
    """List skill names present on disk (filenames without `.md`)."""
    return sorted(p.stem for p in _SKILLS_DIR.glob("*.md"))


__all__ = ["load_skill", "report_skill_name", "available_skills", "SkillNotFound"]
