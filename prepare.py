"""Prepare prompt-disjoint English HS3 pairs and prefetch the small encoder."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random

from huggingface_hub import hf_hub_download, snapshot_download

from common import digest, load_config, snapshot, write_json


def source_rows(name, revision, split):
    path = hf_hub_download(
        name, f"preference/{split}.jsonl.gz", repo_type="dataset", revision=revision
    )
    with gzip.open(path, "rt") as source:
        for line in source:
            yield json.loads(line)


def normalize(rows, sealed_prompts=()):
    """Canonicalize response order, collapse duplicates, and omit disagreements."""
    pairs, conflicts, counts = {}, set(), Counter()
    for row in rows:
        counts["source_rows"] += 1
        if digest(row["context"]) in sealed_prompts:
            counts["official_test_prompt_overlap_rows"] += 1
            continue
        # Code's language field names the programming language, not prompt language.
        if row["domain"].lower() == "multilingual":
            counts["multilingual_rows"] += 1
            continue
        context = row["context"]
        prompt_id = digest(context)
        left, right = row["response1"], row["response2"]
        preference = row["overall_preference"]
        # HS3 negative -> Response 1 wins; our Y=+1 -> left wins.
        label = -1 if preference > 0 else (1 if preference < 0 else 0)
        if left == right:
            counts["identical_responses"] += 1
            continue
        if right < left:
            left, right, label = right, left, -label
        pair_id = digest([prompt_id, left, right])
        if pair_id in pairs:
            counts["duplicate_rows"] += 1
            if pairs[pair_id]["label"] != label:
                conflicts.add(pair_id)
        pairs[pair_id] = dict(
            pair_id=pair_id,
            prompt_id=prompt_id,
            context=context,
            left=left,
            right=right,
            label=label,
        )
    counts["conflicting_pairs"] = len(conflicts)
    counts["tie_pairs"] = sum(pair["label"] == 0 for pair in pairs.values())
    return [
        pair
        for key, pair in pairs.items()
        if key not in conflicts and pair["label"] != 0
    ], dict(counts)


def split_pairs(development, test, config):
    """Official validation owns its prompts; split remaining prompts before capping."""
    test_prompts = {pair["prompt_id"] for pair in test}
    development = [
        pair for pair in development if pair["prompt_id"] not in test_prompts
    ]
    prompts = sorted({pair["prompt_id"] for pair in development})
    random.Random(config["seed"]).shuffle(prompts)
    boundary = max(1, int(len(prompts) * config["data"]["validation_fraction"]))
    validation_prompts = set(prompts[:boundary])
    partitions = {
        "train": [
            pair for pair in development if pair["prompt_id"] not in validation_prompts
        ],
        "validation": [
            pair for pair in development if pair["prompt_id"] in validation_prompts
        ],
        "test": test,
    }
    for split, pairs in partitions.items():
        prompts = sorted({pair["prompt_id"] for pair in pairs})
        random.Random(config["seed"]).shuffle(prompts)
        selected = set(prompts[: config["data"][split + "_prompts"]])
        partitions[split] = [pair for pair in pairs if pair["prompt_id"] in selected]
    return partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new --output")
    data, model = config["data"], config["model"]
    # Seal every official held-out context, including rows later omitted as ties.
    official_test = list(source_rows(data["name"], data["revision"], "validation"))
    sealed = {digest(row["context"]) for row in official_test}
    test, test_audit = normalize(official_test)
    development, train_audit = normalize(
        source_rows(data["name"], data["revision"], "train"), sealed
    )
    partitions = split_pairs(development, test, config)
    snapshot_download(
        model["name"],
        revision=model["revision"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    manifest = snapshot(args.output, config)
    manifest.update(
        dataset=data,
        model=model,
        tokenizer={"name": model["name"], "revision": model["revision"]},
        audit={"train_source": train_audit, "test_source": test_audit},
        splits={},
    )
    for split, pairs in partitions.items():
        if not pairs:
            raise ValueError(f"No {split} pairs; increase the prompt limits")
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs)
        )
        manifest["splits"][split] = {
            "pairs": len(pairs),
            "prompts": len({pair["prompt_id"] for pair in pairs}),
            "sha256": digest(pairs),
        }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
