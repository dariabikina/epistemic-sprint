# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Is a response correct? Wu: "correct if it contains any normalized gold answer".

Normalization is SQuAD's (lowercase, drop punctuation and articles, squeeze spaces). We require the
answer to sit on word boundaries: PopQA aliases include strings like "pol" for "politician", which
a plain substring test would find inside "politics" or "Napoleon". `loose` (plain substring, as in
PopQA's own evaluation script) is kept alongside, to see how much the choice matters.
"""

import re
import string
import unicodedata

_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def match(response: str, answers: list[str]) -> bool:
    padded = f" {normalize(response)} "
    return any((g := normalize(a)) and f" {g} " in padded for a in answers)


def loose_match(response: str, answers: list[str]) -> bool:
    low = response.lower()
    return any(a and a.lower() in low for a in answers)
