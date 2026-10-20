"""Full-model Llama reward training with the HS3 paper's scaled BT loss."""

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from torch.distributed import checkpoint as dcp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from common import load_config, read_jsonl, snapshot, write_json


def scaled_bt(chosen, rejected, strength):
    # HS3 Section 3.2: L_i = |preference_i| * log(1 + exp(-(r_chosen-r_rejected))).
    return strength * F.softplus(-(chosen.float() - rejected.float()))


def collate(rows, pad_token_id):
    # Put all chosen responses first, then all rejected responses.
    tokens = [
        torch.tensor(row[side]) for side in ("chosen", "rejected") for row in rows
    ]
    input_ids = pad_sequence(tokens, batch_first=True, padding_value=pad_token_id)
    return dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(pad_token_id),
        strength=torch.tensor([row["strength"] for row in rows], dtype=torch.float32),
        indices=[row["index"] for row in rows],
    )


def forward_pair(model, batch, device):
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        rewards = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits.flatten()
    chosen, rejected = rewards.chunk(2)
    return scaled_bt(chosen, rejected, batch["strength"].to(device)), chosen > rejected


@torch.no_grad()
def evaluate(model, loader, device, world_size):
    model.eval()
    rows = []
    for batch in loader:
        losses, correct = forward_pair(model, batch, device)
        rows.extend(
            zip(batch["indices"], losses.cpu().tolist(), correct.cpu().tolist())
        )
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, rows)
        rows = [row for group in gathered for row in group]
    # DistributedSampler pads validation. Count each original example once.
    unique = {index: (loss, correct) for index, loss, correct in rows}
    model.train()
    return dict(
        loss=sum(row[0] for row in unique.values()) / len(unique),
        accuracy=sum(row[1] for row in unique.values()) / len(unique),
        pairs=len(unique),
    )


def build_model(spec, pad_token_id, device, rank, world_size):
    config = AutoConfig.from_pretrained(spec["name"], revision=spec["revision"])
    config.num_labels, config.pad_token_id, config.use_cache = 1, pad_token_id, False
    # Keep FP32 master parameters; FSDP casts computation to BF16 on the GPU.
    # Only rank zero loads real weights. Other GPU ranks start with empty tensors.
    if rank == 0 or device.type == "cpu":
        model = AutoModelForSequenceClassification.from_pretrained(
            spec["name"],
            revision=spec["revision"],
            config=config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    else:
        with torch.device("meta"):
            model = AutoModelForSequenceClassification.from_config(
                config, torch_dtype=torch.float32, attn_implementation="sdpa"
            )
    if world_size == 1:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        return model.to(device)
    model = FSDP(
        model,
        device_id=device,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
        ),
        param_init_fn=lambda module: module.to_empty(device=device, recurse=False),
        sync_module_states=device.type == "cuda",
        use_orig_params=False,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        if device.type == "cuda"
        else None,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        limit_all_gathers=True,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda module: isinstance(module, LlamaDecoderLayer),
    )
    return model


def save_checkpoint(model, path, tokenizer_path, rank, world_size):
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            dcp.save_state_dict(
                {"model": model.state_dict()}, dcp.FileSystemWriter(path / "model")
            )
        if rank == 0:
            model.module.config.save_pretrained(path)
    else:
        model.save_pretrained(path)
    if rank == 0:
        shutil.copytree(tokenizer_path, path / "tokenizer")
    if world_size > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--data", type=Path, default=Path("data/llama33-hs3"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Checkpoint directory on scratch/shared storage",
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Small run records, usually in home"
    )
    args = parser.parse_args()
    config, started = load_config(args.config), time.monotonic()
    rank, world_size = (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    if args.output.exists() or args.results.exists():
        raise FileExistsError("Choose new --output and --results directories")
    if world_size > 1:
        dist.barrier()
    set_seed(config["seed"])
    manifest = json.loads((args.data / "manifest.json").read_text())
    train, validation = (
        read_jsonl(args.data / "train.jsonl"),
        read_jsonl(args.data / "validation.jsonl"),
    )
    spec = config["training"]
    batch, global_batch = spec["micro_batch_pairs"], spec["global_batch_pairs"]
    if global_batch % (batch * world_size):
        raise ValueError(
            "global_batch_pairs must divide evenly across ranks and microbatches"
        )
    accumulation = global_batch // (batch * world_size)
    train_sampler = DistributedSampler(
        train, world_size, rank, seed=config["seed"], drop_last=True
    )
    val_sampler = DistributedSampler(validation, world_size, rank, shuffle=False)
    collator = partial(collate, pad_token_id=manifest["pad_token_id"])
    train_loader = DataLoader(
        train, batch_size=batch, sampler=train_sampler, collate_fn=collator
    )
    val_loader = DataLoader(
        validation, batch_size=batch, sampler=val_sampler, collate_fn=collator
    )
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * spec["epochs"]
    if spec["max_steps"] > 0:
        total_steps = min(total_steps, spec["max_steps"])
    run = None
    if rank == 0:
        run = snapshot(args.results, config)
        run.update(
            data_manifest=manifest,
            world_size=world_size,
            seed_count=1,
            complete=False,
            checkpoint_format="FSDP model shards" if world_size > 1 else "Hugging Face",
        )
        write_json(args.results / "run.json", run)
    if world_size > 1:
        dist.barrier()
    model = build_model(
        config["model"], manifest["pad_token_id"], device, rank, world_size
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec["learning_rate"],
        betas=(spec["beta1"], spec["beta2"]),
        weight_decay=spec["weight_decay"],
        fused=device.type == "cuda",
        foreach=False,
    )
    optimizer.zero_grad()
    step, best_loss, selected, history = 0, float("inf"), None, []
    for epoch in range(spec["epochs"]):
        train_sampler.set_epoch(epoch)
        for microstep, batch in enumerate(train_loader):
            # Last accumulation group may be smaller. Normalize by its actual size.
            group_start = (microstep // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            loss, _ = forward_pair(model, batch, device)
            (loss.mean() / group_size).backward()  # FSDP synchronizes each microbatch.
            if (microstep + 1) % accumulation and microstep + 1 != len(train_loader):
                continue
            warmup = min((step + 1) / max(spec["warmup_steps"], 1), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = spec["learning_rate"] * warmup
            if isinstance(model, FSDP):
                model.clip_grad_norm_(spec["max_grad_norm"])
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), spec["max_grad_norm"]
                )
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if rank == 0:
                print(
                    f"step={step}/{total_steps} lr={optimizer.param_groups[0]['lr']:.2g}",
                    flush=True,
                )
            if step % spec["eval_steps"] == 0 or step == total_steps:
                metrics = evaluate(model, val_loader, device, world_size)
                path = args.output / f"checkpoint-{step}"
                save_checkpoint(model, path, args.data / "tokenizer", rank, world_size)
                history.append(
                    dict(step=step, validation=metrics, checkpoint=str(path.resolve()))
                )
                if metrics["loss"] < best_loss:
                    best_loss, selected = metrics["loss"], history[-1]
                if rank == 0:
                    write_json(args.results / "history.json", history)
                    write_json(args.results / "selected.json", selected)
                    print(f"validation={metrics}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    if rank == 0:
        run.update(
            complete=True,
            actual_final_step=step,
            epochs_completed=step / steps_per_epoch,
            selected_checkpoint=selected,
            seconds=time.monotonic() - started,
        )
        write_json(args.results / "run.json", run)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
