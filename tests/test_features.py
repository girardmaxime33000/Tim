from __future__ import annotations

import math

import pytest

from growth_system.features import extract_features


def test_empty_string() -> None:
    feat = extract_features("")
    assert feat["log_len"] == math.log(1)
    assert feat["hashtag"] == 0.0
    assert feat["question_body"] == 0.0
    assert feat["hook_contrarian"] == 0.0
    assert feat["hook_question"] == 0.0
    assert feat["hook_data"] == 0.0


def test_hashtag_detected() -> None:
    feat = extract_features("Bonjour #LinkedIn")
    assert feat["hashtag"] == 1.0


def test_question_body() -> None:
    feat = extract_features("Qu'est-ce que vous pensez?")
    assert feat["question_body"] == 1.0


def test_hook_contrarian() -> None:
    feat = extract_features("Le mythe de la croissance rapide\nSuite...")
    assert feat["hook_contrarian"] == 1.0


def test_hook_question_first_line() -> None:
    feat = extract_features("Et si on arrêtait de tout mesurer?\nSuite du texte.")
    # First line ends with '?' → hook_question
    assert feat["hook_question"] == 1.0


def test_hook_data_in_first_line() -> None:
    feat = extract_features("En 2024, tout a changé\nblah blah")
    assert feat["hook_data"] == 1.0
    # "En 2024" also matches contrarian pattern
    assert feat["hook_contrarian"] == 1.0


def test_log_len_increases() -> None:
    short = extract_features("abc")
    long = extract_features("abc" * 100)
    assert long["log_len"] > short["log_len"]


def test_non_string_input() -> None:
    feat = extract_features(None)  # type: ignore[arg-type]
    assert feat["log_len"] == math.log(1)


def test_returns_all_features() -> None:
    expected_keys = {
        "log_len", "tags", "emoji", "hashtag", "question_body",
        "hook_contrarian", "hook_question", "hook_data",
    }
    feat = extract_features("Test text")
    assert set(feat.keys()) == expected_keys
