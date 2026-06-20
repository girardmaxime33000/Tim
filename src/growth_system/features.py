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
