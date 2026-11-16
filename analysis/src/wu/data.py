# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Datasets, frozen: PopQA (questions, popularity buckets), WikiText-2 (perplexity), C4 (calibration).

PopQA buckets follow Wu A.1.3: sort all 14,267 questions by subject popularity (s_pop) and split
into equal thirds. The buckets are assigned on the full set, *then* we draw a stratified
subsample, so a bucket means the same thing whatever the subsample size.
"""

import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

POPQA = ("akariasai/PopQA", "098765c79ea10a2cb19c828324e33281b8336ec0")
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")
C4 = ("allenai/c4", "en/c4-validation.00000-of-00008.json.gz")

DATA = Path(__file__).resolve().parents[2] / "data"  # model/data: small frozen files, kept with the code
BUCKETS = ("head", "middle", "tail")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def popqa_all() -> list[dict]:
    path = hf_hub_download(POPQA[0], "test.tsv", repo_type="dataset", revision=POPQA[1])
    rows = [
        {
            "id": r["id"],
            "question": r["question"],
            "answers": json.loads(r["possible_answers"]),
            "s_pop": int(r["s_pop"]),
            "prop": r["prop"],
            "subj": r["subj"],
        }
        for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t")
    ]
    rows.sort(key=lambda r: (-r["s_pop"], r["id"]))  # most popular first; id breaks ties deterministically
    third = len(rows) / 3
    for i, r in enumerate(rows):
        r["bucket"] = BUCKETS[min(int(i // third), 2)]
    return rows


def freeze_popqa(per_bucket: int = 1000, seed: int = 0) -> Path:
    """Write data/popqa-<n>.jsonl (stratified, sorted by id) and its sha256 to data/MANIFEST.json."""
    rows = popqa_all()
    rng = random.Random(seed)
    chosen = []
    for b in BUCKETS:
        pool = [r for r in rows if r["bucket"] == b]
        chosen += pool if per_bucket <= 0 else rng.sample(pool, per_bucket)
    chosen.sort(key=lambda r: int(r["id"]))
    DATA.mkdir(exist_ok=True)
    out = DATA / f"popqa-{len(chosen)}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen))
    manifest_path = DATA / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[out.name] = {
        "sha256": sha256(out),
        "source": f"{POPQA[0]}@{POPQA[1]}",
        "per_bucket": per_bucket,
        "seed": seed,
        "bucket_rule": "sort by s_pop desc (id asc), equal thirds of all 14,267",
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return out


def popqa(n: int = 3000) -> list[dict]:
    """The frozen subsample, verified against the manifest."""
    path = DATA / f"popqa-{n}.jsonl"
    expected = json.loads((DATA / "MANIFEST.json").read_text())[path.name]["sha256"]
    if sha256(path) != expected:
        raise ValueError(f"{path} does not match its manifest; refreeze deliberately or restore it")
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def wikitext2_test() -> str:
    """Joined the usual way (as in GPTQ/WANDA evaluation code): rows joined by blank lines."""
    import pyarrow.parquet as pq

    path = hf_hub_download(WIKITEXT[0], WIKITEXT[1], repo_type="dataset")
    return "\n\n".join(pq.read_table(path)["text"].to_pylist())


def c4_calibration(tokenizer, samples: int, length: int, seed: int = 42) -> list[list[int]]:
    """`samples` random windows of `length` tokens from documents of the C4 validation shard that are long enough
    (the GPTQ recipe). Wu: 128 x 512 for quantization, 32 x 512 for pruning, seed 42."""
    path = hf_hub_download(C4[0], C4[1], repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        docs = [json.loads(line)["text"] for line in f]
    rng, out = random.Random(seed), []
    while len(out) < samples:
        ids = tokenizer.encode(docs[rng.randrange(len(docs))], add_special_tokens=False)
        if len(ids) > length:
            start = rng.randrange(len(ids) - length)
            out.append(ids[start : start + length])
    return out
