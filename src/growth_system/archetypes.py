from __future__ import annotations

import re
from typing import Literal

ArmId = Literal["contrarian", "data", "question", "statement"]
ALL_ARMS: list[ArmId] = ["contrarian", "data", "question", "statement"]

_HOOK_CONTRARIAN_RE = re.compile(
    r'^\s*(Le mythe|L.illusion|Et si|Le march|Pourquoi|Personne|On a |L.inflation|Je ne crois|Arr[êe]tez|En \d{4})',
    re.IGNORECASE,
)


def archetype_of(text: str, format: str = "") -> ArmId:
    """Classify a post text into one of 4 archetypes based on its hook."""
    first_line = text[:200].split('\n')[0].strip() if text else ""
    if _HOOK_CONTRARIAN_RE.match(first_line):
        return "contrarian"
    if first_line.endswith('?'):
        return "question"
    if re.search(r'\d', first_line):
        return "data"
    return "statement"
