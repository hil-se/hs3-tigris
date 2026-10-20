"""Small helpers for JSON and reproducible run records."""

from datetime import datetime, timezone
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import tomllib

ROOT = Path(__file__).resolve().parent


def load_config():
    return tomllib.loads((ROOT / "config.toml").read_text())


def digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def read_jsonl(path):
    return [dict(json.loads(line), index=i) for i, line in enumerate(Path(path).open())]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def snapshot(output, config):
    files = [
        *ROOT.glob("*.py"),
        ROOT / "requirements.txt",
        *ROOT.glob("cluster/tigris/*"),
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = Path(output) / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(path.read_bytes()).hexdigest()
    return dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        source_sha256=hashes,
        python=platform.python_version(),
        packages={
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "numpy",
                "sentencepiece",
            )
        },
    )
