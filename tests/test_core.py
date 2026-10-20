"""Only the preference mapping and the paper's loss equation."""

import math
import torch

from prepare import encode_pair
from train import scaled_bt


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [ord(char) for char in messages[-1]["content"]]


def test_preference_direction_strength_and_ties():
    row = dict(
        context=[{"role": "user", "content": "prompt"}],
        response1="a",
        response2="b",
        overall_preference=3,
    )
    pair = encode_pair(row, Tokenizer(), 4096)
    assert pair["chosen"] == [ord("b")]
    assert pair["rejected"] == [ord("a")]
    assert pair["strength"] == 3
    assert encode_pair(dict(row, overall_preference=0), Tokenizer(), 4096) is None


def test_scaled_bt_value_and_gradient():
    chosen = torch.zeros(2, requires_grad=True)
    losses = scaled_bt(chosen, torch.zeros(2), torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(losses, torch.tensor([math.log(2), 3 * math.log(2)]))
    losses.mean().backward()
    torch.testing.assert_close(chosen.grad, torch.tensor([-0.25, -0.75]))
