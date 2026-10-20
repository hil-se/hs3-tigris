"""Run the trainer with a tiny random Llama and toy preference pairs."""

import json
from pathlib import Path
import sys
import tempfile

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    LlamaConfig,
    LlamaForSequenceClassification,
    PreTrainedTokenizerFast,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config
from train import train


root = Path(tempfile.mkdtemp(prefix="hs3-smoke-"))
model_path = root / "model"
model = LlamaForSequenceClassification(
    LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_labels=1,
        pad_token_id=0,
    )
)
model.save_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(
        WordLevel({f"t{i}": i for i in range(64)}, unk_token="t1")
    ),
    pad_token="t0",
    unk_token="t1",
)
data = root / "data"
tokenizer.save_pretrained(data / "tokenizer")
for split, count in [("train", 16), ("validation", 3)]:
    rows = [
        dict(
            pair_id=f"{split}-{i}",
            prompt_id=f"{split}-{i}",
            chosen=[1, 3 + i % 4, 2],
            rejected=[1, 12 + i % 4, 2],
            strength=1 + i % 3,
        )
        for i in range(count)
    ]
    (data / f"{split}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
(data / "manifest.json").write_text(
    json.dumps({"pad_token_id": 0, "fixture": "random tiny Llama"})
)
config = load_config()
config["model"] = dict(name=str(model_path), revision="local")
config["training"].update(
    global_batch_pairs=4, learning_rate=0.001, warmup_steps=1, eval_steps=2
)
train(config, data, root / "checkpoints", root / "results")
print(f"Smoke-test files: {root}")
