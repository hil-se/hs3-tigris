# HS3 on TIGRIS

A small example for learning to run HelpSteer3 preference experiments on TIGRIS.
It freezes **Qwen2.5-0.5B-Instruct**, embeds each chat/response, and fits two linear
reward heads: unweighted Bradley–Terry and Comparative FairReweighing (CFR).
One GH200 is enough. The encoder is never fine-tuned.

Read the code in execution order:

1. `prepare.py` downloads pinned data/model revisions and writes fixed splits.
2. `embed.py` caches the last non-padding chat-token embedding for each response.
3. `train.py` fits matched heads, selects by validation loss, and evaluates test.
4. `common.py` contains the weighting equation and descriptive metrics.

## First TIGRIS run

Use your own RIT login and Slurm project account. Your PI must have granted you
TIGRIS access. Start from a clean clone:

```bash
ssh YOUR_RIT_USERNAME@tigris.rc.rit.edu
git clone https://github.com/hil-se/hs3-tigris.git
cd hs3-tigris

source /.autofs/tools/spack/share/spack/setup-env.sh
spack env activate default-ml-aarch64-25050701
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt

source cluster/tigris/env.sh
HF_HUB_OFFLINE=0 python prepare.py
sbatch --account=YOUR_SLURM_ACCOUNT cluster/tigris/run.slurm
```

The virtual environment reuses Spack's GPU-enabled PyTorch; do not install a
replacement CPU PyTorch wheel on TIGRIS. Preparation downloads data and the
encoder before requesting a GPU. The job runs offline from that cache.

```bash
squeue -M tigris -u "$USER"
sacct -M tigris -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -f slurm-hs3-prototype-JOB_ID.out
cat results/job-JOB_ID/comparison.json
```

The launch script requests one GH200, four CPU cores, 16 GB RAM, and 30 minutes.
This is a resource request, not a runtime guarantee. It contains no node
exclusions or placement constraints. Paths use your username:

- `data/`: prepared pairs, split audit and pinned configuration, in the clone.
- `/scratch/$USER/hs3-tigris/`: model cache and per-job frozen features.
- `results/job-JOB_ID/`: small head checkpoints, metrics, pair scores and source snapshots.

Set `ARTIFACT_ROOT` before submission to use another writable feature directory.
Set `HF_HOME` before **both preparation and submission** to move the model cache.
To change the cohort, edit `config.toml`, prepare into a new directory with
`python prepare.py --output data-new`, then submit with `DATA_DIR=data-new`.
Existing outputs are never overwritten or pruned automatically.

## What the comparison means

HS3 negative preferences favor Response 1; positive preferences favor Response 2.
Here `Y=+1` means the left response wins. Preference ties and identical responses
are omitted; preference strength is ignored for this binary BT example.
Response order is canonicalized and conflicting duplicate labels are omitted.
The non-multilingual domains include English code prompts.

The official HS3 validation contexts are reserved for test, including contexts
whose test pairs are later filtered. The remaining training contexts are split
by prompt into train/development-validation before the prompt limits are applied.
Full multi-turn context is preserved. Pairs exceeding `model.max_length` are
excluded as whole pairs and recorded; responses are never silently truncated.

Let `D = sign(tokens_left - tokens_right)`. CFR estimates
`w(D,Y) = P(D) P(Y) / P(D,Y)` on **training pairs only**, counting their virtual
reverses `(-D,-Y)` to remove arbitrary display position. Weights have mean one.
Both heads use the same frozen features, train-only feature scaling, zero
initialization, minibatch order, one full training epoch and no step cap.
The selected head minimizes **unweighted validation BT loss**.

`comparison.json` reports test preference accuracy and the accuracy difference
between preferred-longer and preferred-shorter pairs. Lower absolute gap is
better; higher accuracy is better. A gap is `null` if either group is absent.
`run.json` records the actual final step, selected step, seed, revisions, config,
versions and source hashes. `head.pt` stores the selected weights and scale;
the training script reloads it before test evaluation.

This small, single-seed demonstration does not reproduce the 70B HS3 paper or
establish a mitigation benefit. It reports no significance tests. Final research
comparisons need matched seeds and prompt-cluster inference. Pointwise CSEP is
not computed because this example does not provide pointwise quality targets.

## Local check

Python 3.11 or 3.12:

```bash
uv venv --python 3.12
uv pip install torch==2.6.0 -r requirements.txt pytest==8.3.5
.venv/bin/python -m pytest -q
```

The three tests cover preference direction/duplicate disagreement, prompt
separation, and the CFR equation. For an actual local run, use the same three
scripts (`embed.py --device cpu`); reduce the prompt limits first.

## Next student task: warmth

Start by finding a published definition and measurement method for warmth.
Validate a rubric with independent human ratings on a small pilot. Then examine
whether reward preferences vary with warmth when quality and length are
comparable. This repo computes no warmth labels or warmth-bias results.

Sources: [HS3 dataset](https://huggingface.co/datasets/nvidia/HelpSteer3#preference),
[Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[RIT cluster documentation](https://research-computing.git-pages.rit.edu/docs/).
