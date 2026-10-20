"""Prepare HS3 English preference pairs with the released NVIDIA reward template."""

import argparse
from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from common import digest, file_hash, load_config, write_json


def source_rows(config, split):
    spec = config["data"]
    path = hf_hub_download(
        spec["name"],
        f"preference/{split}.jsonl.gz",
        repo_type="dataset",
        revision=spec["revision"],
    )
    with gzip.open(path, "rt") as source:
        for line in source:
            yield json.loads(line)


def encode_pair(row, tokenizer, max_length):
    preference = row["overall_preference"]
    if preference == 0:
        return None  # The paper excludes preference ties from training.
    responses = [row["response1"], row["response2"]]
    if preference > 0:
        responses.reverse()  # HS3 positive means Response 2 wins.
    tokens = [
        tokenizer.apply_chat_template(
            row["context"] + [{"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
        )
        for response in responses
    ]
    if max(map(len, tokens)) > max_length:
        return None  # Omit the whole pair; never silently truncate a response.
    return dict(
        prompt_id=digest(row["context"]),
        chosen=tokens[0],
        rejected=tokens[1],
        strength=abs(preference),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", type=Path, default=Path("data/llama33-hs3"))
    parser.add_argument("--skip-model-download", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new --output")
    config = load_config(args.config)
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["name"], revision=model["revision"])
    template_path = hf_hub_download(
        model["template_model"],
        "tokenizer_config.json",
        revision=model["template_revision"],
    )
    tokenizer.chat_template = json.loads(Path(template_path).read_text())[
        "chat_template"
    ]
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"
    official_validation = list(source_rows(config, "validation"))
    held_out = {digest(row["context"]) for row in official_validation}
    args.output.mkdir(parents=True)
    tokenizer.save_pretrained(args.output / "tokenizer")
    audit = {}
    for split in ("train", "validation"):
        counts = Counter()
        rows = source_rows(config, "train") if split == "train" else official_validation
        with (args.output / f"{split}.jsonl").open("w") as target:
            for i, row in enumerate(rows):
                counts["source_rows"] += 1
                if row["domain"].lower() not in config["data"]["domains"]:
                    counts["other_domains"] += 1
                    continue
                if split == "train" and digest(row["context"]) in held_out:
                    counts["validation_prompt_overlap"] += 1
                    continue
                pair = encode_pair(row, tokenizer, model["max_length"])
                if pair is None:
                    counts["ties_or_overlength"] += 1
                    continue
                target.write(json.dumps(dict(pair, pair_id=f"{split}-{i}")) + "\n")
                counts["pairs"] += 1
        audit[split] = dict(counts)
    if not args.skip_model_download:
        snapshot_download(
            model["name"],
            revision=model["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        )
    write_json(
        args.output / "manifest.json",
        dict(
            config=config,
            tokenizer=dict(name=model["name"], revision=model["revision"]),
            template_sha256=sha256(tokenizer.chat_template.encode()).hexdigest(),
            pad_token_id=tokenizer.pad_token_id,
            model_prefetched=not args.skip_model_download,
            audit=audit,
            files_sha256={
                split: file_hash(args.output / f"{split}.jsonl")
                for split in ("train", "validation")
            },
        ),
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
