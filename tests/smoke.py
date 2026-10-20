"""Create a tiny random Llama and tokenized toy pairs for testing train.py."""

import json
from pathlib import Path
import sys

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    LlamaConfig,
    LlamaForSequenceClassification,
    PreTrainedTokenizerFast,
)


root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True)
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
(root / "config.toml").write_text(f'''seed = 1234
[model]
name = "{model_path}"
revision = "local"
[training]
epochs = 1
max_steps = -1
global_batch_pairs = 4
micro_batch_pairs = 1
learning_rate = 0.001
warmup_steps = 1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.98
max_grad_norm = 1.0
eval_steps = 2
''')
print(root)
