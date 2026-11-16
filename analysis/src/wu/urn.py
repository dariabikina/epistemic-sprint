# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Marbles in words: text problems whose next-colour probability is known exactly.

The ground-truth worlds (`wu.gt`) score small models against an exact Bayes predictive. This is the same
idea for a language model: a short text ends where a colour word comes next, and the probability of each
colour is exact.

Two kinds of item, K colours each (K = 2, 3 or 4, drawn from COLOURS):

    bag     "A bag contains 7 red marbles, 2 blue marbles and 1 green marble. Marbles are drawn at random,
            one at a time, and each is put back before the next draw." P(colour) = its count / 10 at every
            draw. Nothing is inferred: the numbers are in the text. Three earlier draws (sampled from the
            bag) are listed, only so that the text continues with a colour word; they do not change P.
    draws   A Polya urn, stated in the text: the bag starts with one marble of each colour; each drawn marble
            is put back with one more of the same colour. After n draws (n = 4, 8, 16, 32) with count_c of
            colour c,  P(next = c) = (1 + count_c) / (K + n).  The draws are sampled from that urn. This is
            the `zipf` world's urn (Blackwell & MacQueen 1973) with a uniform base and alpha = K; the model
            must count what the text has shown.

Each item is read in two ways (see `wu.confidence`):
    predict  the text continued as plain text (no chat template): it ends "Colours drawn so far: red, blue,"
             so the next token is a colour word; the model's probabilities of the K colour words are its
             prediction.
    state    a chat question, once per colour: how likely is this colour, as a digit 0-9 (d = d0%-(d+1)0%)?
             The reply is started with STATE_PREFILL, or the model reasons aloud before it answers.

Prompt wording was fixed on 15 Nov after a smoke run on qwen3-1.7b and qwen3-8b (base): with the first
wording the next token was a colour in 57% of bag items, and a digit in under 1% of state questions.

`python -m wu.urn` writes data/marbles-900.jsonl and its sha256 to data/MANIFEST.json.
"""

import json
import random

from .data import DATA, sha256

COLOURS = ["red", "blue", "green", "yellow", "black", "white", "pink", "brown"]
BAG_TOTAL = 10
BAG_SHOWN = 3  # earlier draws listed in a bag item
DRAWS = (4, 8, 16, 32)
PER_CELL = {"bag": 100, "draws": 50}  # bag: per K; draws: per (K, n)
FILE = "marbles-900.jsonl"

RULE = (
    "A bag starts with one marble of each of these colours: {colours}. "
    "A marble is drawn at random and its colour is written down. Then it is put back in the bag "
    "together with one more marble of the same colour. This is repeated many times."
)
BAG_RULE = "Marbles are drawn at random, one at a time, and each is put back before the next draw."
STATE_PREFILL = "The digit is "
STATE = (
    "How likely is it that the next marble drawn is {colour}? Answer with one digit from 0 to 9: "
    "0 means a chance between 0% and 10%, 1 between 10% and 20%, and so on up to 9, between 90% and 100%."
)


def _listing(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def bag_item(rng: random.Random, k: int) -> dict:
    colours = rng.sample(COLOURS, k)
    cuts = sorted(rng.sample(range(1, BAG_TOTAL), k - 1))  # a uniform composition of 10 into k positive parts
    counts = [b - a for a, b in zip([0, *cuts], [*cuts, BAG_TOTAL])]
    said = _listing([f"{n} {c} marble{'s' if n > 1 else ''}" for c, n in zip(colours, counts)])
    seen = rng.choices(colours, weights=counts, k=BAG_SHOWN)
    text = f"A bag contains {said}. {BAG_RULE}\nColours drawn so far: " + ", ".join(seen)
    return {
        "kind": "bag",
        "k": k,
        "n": 0,
        "colours": colours,
        "p": [n / BAG_TOTAL for n in counts],
        "text": text + ".",
        "predict": text + ",",
    }


def draws_item(rng: random.Random, k: int, n: int) -> dict:
    colours = rng.sample(COLOURS, k)
    counts, seen = [1] * k, []
    for _ in range(n):  # the urn itself: draw in proportion to the current contents, add one of that colour
        j = rng.choices(range(k), weights=counts)[0]
        counts[j] += 1
        seen.append(colours[j])
    text = RULE.format(colours=_listing(colours)) + "\nColours drawn so far: " + ", ".join(seen)
    return {
        "kind": "draws",
        "k": k,
        "n": n,
        "colours": colours,
        "p": [c / (k + n) for c in counts],  # counts already include the starting marble
        "text": text + ".",
        "predict": text + ",",
    }


def state_message(item: dict, colour: str) -> list[dict]:
    """The chat question about one colour."""
    return [{"role": "user", "content": item["text"] + "\n" + STATE.format(colour=colour)}]


def stated_probability(digit_probs) -> float:
    """Expected probability from a distribution over the digits 0-9: digit d stands for (d + 0.5) / 10."""
    total = sum(digit_probs)
    return sum(p * (d + 0.5) / 10 for d, p in enumerate(digit_probs)) / total


def build(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items = [bag_item(rng, k) for k in (2, 3, 4) for _ in range(PER_CELL["bag"])]
    items += [draws_item(rng, k, n) for k in (2, 3, 4) for n in DRAWS for _ in range(PER_CELL["draws"])]
    for i, item in enumerate(items):
        item["id"] = i
    return items


def freeze(seed: int = 0):
    items = build(seed)
    path = DATA / FILE
    path.write_text("".join(json.dumps(it) + "\n" for it in items))
    manifest_path = DATA / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[FILE] = {"sha256": sha256(path), "source": "wu.urn.build", "seed": seed, "items": len(items)}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def marbles() -> list[dict]:
    """The frozen items, verified against the manifest."""
    path = DATA / FILE
    if sha256(path) != json.loads((DATA / "MANIFEST.json").read_text())[FILE]["sha256"]:
        raise ValueError(f"{path} does not match its manifest; refreeze deliberately or restore it")
    return [json.loads(line) for line in path.read_text().split("\n") if line]


if __name__ == "__main__":
    print(f"wrote {freeze()}")
