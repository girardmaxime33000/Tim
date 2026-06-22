from __future__ import annotations

import re
import math

# Hook patterns for first line (≤200 chars), case-insensitive
_HOOK_CONTRARIAN_RE = re.compile(
    r'^\s*(Le mythe|L.illusion|Et si|Le march|Pourquoi|Personne|On a |L.inflation|Je ne crois|Arr[êe]tez|En \d{4})',
    re.IGNORECASE,
)
_EMOJI_RE = re.compile(r'[\U0001F300-\U0001FAFF☀-➿]')
_TAGS_RE = re.compile(r'[A-ZÀ-Ý][a-zà-ÿ]+\s+[A-ZÀ-Ý]')

# CTA patterns — defined once, shared by table display and future scoring features
_CTA_RE = re.compile(
    r'(👉|Découvrez|Dites-moi|Partagez|[Cc]ommentez|DM|contactez|abonnez)',
    re.IGNORECASE,
)
_LINK_RE = re.compile(r'https?://')


def extract_features(text: str) -> dict[str, float]:
    """Extract the 8 features used by the tail scorer."""
    if not isinstance(text, str):
        text = ""

    log_len = math.log(1 + len(text))
    tags = len(_TAGS_RE.findall(text))
    emoji = len(_EMOJI_RE.findall(text))
    hashtag = 1.0 if '#' in text else 0.0
    question_body = 1.0 if '?' in text else 0.0

    first_line = text[:200].split('\n')[0] if text else ""
    hook_contrarian = 1.0 if _HOOK_CONTRARIAN_RE.match(first_line) else 0.0
    hook_question = 1.0 if first_line.strip().endswith('?') else 0.0
    hook_data = 1.0 if re.search(r'\d', first_line) else 0.0

    return {
        "log_len": log_len,
        "tags": float(tags),
        "emoji": float(emoji),
        "hashtag": hashtag,
        "question_body": question_body,
        "hook_contrarian": hook_contrarian,
        "hook_question": hook_question,
        "hook_data": hook_data,
    }


def hook_type_label(text: str) -> str:
    """Return the archetype label for the hook of a post.

    Priority: Contrarian > Question > Data > Statement.
    Mirrors the priority order used by archetypes.py (archetype_of).
    """
    if not isinstance(text, str):
        return "Statement"
    feats = extract_features(text)
    if feats["hook_contrarian"]:
        return "Contrarian"
    if feats["hook_question"]:
        return "Question"
    if feats["hook_data"]:
        return "Data"
    return "Statement"


def has_link(text: str) -> bool:
    """Return True if the text contains a URL (http/https)."""
    return bool(_LINK_RE.search(text)) if isinstance(text, str) else False


def has_cta(text: str) -> bool:
    """Return True if the text contains a call-to-action pattern."""
    return bool(_CTA_RE.search(text)) if isinstance(text, str) else False

