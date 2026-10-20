"""Shared I/O and the two equations used in this prototype."""

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent


def load_config(path):
    return tomllib.loads(Path(path).read_text())


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return sha256(text.encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def snapshot(output, config):
    """Save the actual source, config and package versions beside an artifact."""
    output = Path(output)
    paths = [*ROOT.glob("*.py"), ROOT / "requirements.txt"]
    paths += list((ROOT / "cluster/tigris").glob("*"))
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(path.read_bytes()).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "source_sha256": hashes,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "huggingface-hub", "numpy")
        },
    }


def cfr_weights(directions, labels):
    """Fit on training rows only; count virtual reverse pairs to remove position."""
    pairs = list(zip(directions, labels))
    table = Counter(pairs + [(-d, -y) for d, y in pairs])
    direction_count = Counter()
    label_count = Counter()
    for (d, y), count in table.items():
        direction_count[d] += count
        label_count[y] += count
    # Comparative FR: w(d,y) = P(D=d) P(Y=y) / P(D=d,Y=y).
    weights = [
        direction_count[d] * label_count[y] / (2 * len(pairs) * table[d, y])
        for d, y in pairs
    ]
    mean = sum(weights) / len(weights)
    return [weight / mean for weight in weights]


def pair_metrics(margins, labels, directions):
    """Descriptive metrics. Undefined group gaps stay null."""
    correct = [y * margin > 0 for y, margin in zip(labels, margins)]
    groups = {}
    for name, sign in [("preferred_longer", 1), ("preferred_shorter", -1)]:
        values = [ok for ok, y, d in zip(correct, labels, directions) if y * d == sign]
        groups[name] = sum(values) / len(values) if values else None
        groups["n_" + name] = len(values)
    gap = (
        groups["preferred_longer"] - groups["preferred_shorter"]
        if all(
            groups[name] is not None
            for name in ("preferred_longer", "preferred_shorter")
        )
        else None
    )
    # Binary BT: loss = log(1 + exp(-Y * (r_left - r_right))).
    losses = [
        max(-y * m, 0) + math.log1p(math.exp(-abs(y * m)))
        for y, m in zip(labels, margins)
    ]
    return {
        "pairs": len(labels),
        "accuracy": sum(correct) / len(correct),
        "bt_loss": sum(losses) / len(losses),
        **groups,
        "n_equal_length": directions.count(0),
        "signed_gap": gap,
        "absolute_gap": abs(gap) if gap is not None else None,
    }
