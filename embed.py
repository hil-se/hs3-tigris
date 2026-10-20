"""Cache last-token embeddings once; every treatment uses these exact pairs."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from common import digest, read_jsonl, snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new --output")
    prepared = json.loads((args.data / "manifest.json").read_text())
    config = prepared["config"]
    spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        spec["name"],
        revision=spec["revision"],
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.to(args.device).eval()
    manifest = snapshot(args.output, config)
    manifest.update(
        data_manifest=prepared,
        device=args.device,
        pooling="last non-padding chat token",
        splits={},
    )
    batch_size = spec["batch_size"]
    for split in ("train", "validation", "test"):
        rows = read_jsonl(args.data / f"{split}.jsonl")
        kept, encoded, directions, excluded = [], [], [], []
        for pair in rows:
            tokens = [
                tokenizer.apply_chat_template(
                    pair["context"] + [{"role": "assistant", "content": pair[side]}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                for side in ("left", "right")
            ]
            if max(map(len, tokens)) > spec["max_length"]:
                excluded.append(pair["pair_id"])
                continue  # Explicit pair filtering, never token truncation.
            lengths = [
                len(tokenizer.encode(pair[side], add_special_tokens=False))
                for side in ("left", "right")
            ]
            directions.append((lengths[0] > lengths[1]) - (lengths[0] < lengths[1]))
            kept.append(pair)
            encoded.extend({"input_ids": ids} for ids in tokens)
        if not kept:
            raise ValueError(f"No {split} pairs fit max_length={spec['max_length']}")
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(encoded), batch_size):
                batch = tokenizer.pad(
                    encoded[start : start + batch_size],
                    padding=True,
                    return_tensors="pt",
                ).to(args.device)
                hidden = model(**batch, use_cache=False).last_hidden_state
                last = batch["attention_mask"].sum(dim=1) - 1
                features = hidden[torch.arange(len(last), device=args.device), last]
                embeddings.append(features.float().cpu())
        features = torch.cat(embeddings).reshape(len(kept), 2, -1)
        artifact = {
            "left": features[:, 0],
            "right": features[:, 1],
            "labels": torch.tensor([pair["label"] for pair in kept]),
            "directions": torch.tensor(directions),
            "pair_ids": [pair["pair_id"] for pair in kept],
            "prompt_ids": [pair["prompt_id"] for pair in kept],
        }
        torch.save(artifact, args.output / f"{split}.pt")
        manifest["splits"][split] = {
            "pairs": len(kept),
            "pair_ids_sha256": digest(artifact["pair_ids"]),
            "excluded_overlength_pair_ids": excluded,
        }
        print(f"{split}: {len(kept)} embedded, {len(excluded)} overlength", flush=True)
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
