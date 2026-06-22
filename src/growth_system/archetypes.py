from __future__ import annotations

from typing import Literal

from .features import hook_type_label

ArmId = Literal["contrarian", "data", "question", "statement"]
ALL_ARMS: list[ArmId] = ["contrarian", "data", "question", "statement"]


def archetype_of(text: str, format: str = "") -> ArmId:
    """Classify a post text into one of 4 archetypes based on its hook."""
    return hook_type_label(text).lower()  # type: ignore[return-value]
