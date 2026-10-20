"""Fit matched unweighted/CFR linear reward heads on frozen embeddings."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import cfr_weights, pair_metrics, snapshot, write_json


def evaluate(head, pairs, scale):
    with torch.no_grad():
        margins = head((pairs["left"] - pairs["right"]) / scale).flatten().tolist()
    metrics = pair_metrics(
        margins, pairs["labels"].tolist(), pairs["directions"].tolist()
    )
    metrics["prompts"] = len(set(pairs["prompt_ids"]))
    scores = [
        dict(
            pair_id=pair_id,
            prompt_id=prompt_id,
            margin=margin,
            label=label,
            length_direction=direction,
        )
        for pair_id, prompt_id, margin, label, direction in zip(
            pairs["pair_ids"],
            pairs["prompt_ids"],
            margins,
            pairs["labels"].tolist(),
            pairs["directions"].tolist(),
        )
    ]
    return metrics, scores


def fit(train, validation, config, method, output):
    """Only train fits the scale and weights; only validation selects the head."""
    torch.manual_seed(config["seed"])
    spec = config["training"]
    # Common scaling fitted on training responses only, shared across treatments.
    scale = torch.cat([train["left"], train["right"]]).std(dim=0).clamp_min(1e-6)
    differences = (train["left"] - train["right"]) / scale
    labels = train["labels"].float()
    weights = (
        torch.tensor(cfr_weights(train["directions"].tolist(), labels.tolist()))
        if method == "cfr"
        else torch.ones(len(labels))
    )
    head = nn.Linear(differences.shape[1], 1, bias=False)
    nn.init.zeros_(head.weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=spec["learning_rate"])
    best_loss, steps, best_step = float("inf"), 0, None
    history = []
    for epoch in range(spec["epochs"]):
        order = torch.randperm(len(labels))
        for batch in order.split(spec["batch_size"]):
            margin = head(differences[batch]).flatten()
            # Binary BT: L = mean_i w_i log(1 + exp(-Y_i * (r_left-r_right))).
            loss = (weights[batch] * F.softplus(-labels[batch] * margin)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
        metrics, _ = evaluate(head, validation, scale)
        history.append(dict(epoch=epoch + 1, step=steps, validation=metrics))
        if metrics["bt_loss"] < best_loss:
            best_loss, best_step = metrics["bt_loss"], steps
            torch.save(
                {
                    "weight": head.weight.detach().clone(),
                    "scale": scale,
                    "selected_step": steps,
                },
                output / "head.pt",
            )
        print(
            f"{method} epoch={epoch + 1} step={steps} val_loss={metrics['bt_loss']:.4f}",
            flush=True,
        )
    saved = torch.load(output / "head.pt", weights_only=True)
    head.load_state_dict({"weight": saved["weight"]})
    return (
        head,
        scale,
        dict(
            actual_final_step=steps,
            selected_step=best_step,
            selection="lowest unweighted validation BT loss",
            history=history,
            train_weights=weights.tolist(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--method", choices=("unweighted", "cfr", "both"), default="both"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new --output")
    manifest = json.loads((args.features / "manifest.json").read_text())
    config = manifest["config"]
    pairs = {
        split: torch.load(args.features / f"{split}.pt", weights_only=True)
        for split in ("train", "validation", "test")
    }
    methods = ("unweighted", "cfr") if args.method == "both" else (args.method,)
    summary = {}
    for method in methods:
        output = args.output / method
        run = snapshot(output, config)
        head, scale, training = fit(
            pairs["train"], pairs["validation"], config, method, output
        )
        metrics, scores = evaluate(head, pairs["test"], scale)
        write_json(output / "metrics.json", metrics)
        (output / "pair_scores.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in scores)
        )
        run.update(
            method=method,
            seed=config["seed"],
            seed_count=1,
            max_steps=-1,
            features_manifest=manifest,
            training=training,
            checkpoint="head.pt",
            complete=True,
        )
        write_json(output / "run.json", run)
        summary[method] = metrics
    write_json(args.output / "comparison.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
