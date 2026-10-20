# Validation of the Llama BT trainer

## Current check: 2026-10-20 02:47 UTC

- The training launcher requires RDMA via `NCCL_NET=IB` and
  `NCCL_IB_DISABLE=0`. INFO logs expose NCCL network initialization.
- `rctrust.571` is retained for socket bootstrap, not the collective data transport.
- Bash syntax and TIGRIS `sbatch --test-only` accepted the updated 16-GPU launcher.
- No training job was submitted. Runtime RDMA operation remains unverified.

## Earlier validation

### Script simplification: 2026-10-20 UTC

- Removed command-line options and custom path/batch error checks. Both launchers
  use `config.toml` and fixed data paths; training outputs use the Slurm job ID.
- Both focused tests pass. `python tests/smoke.py` directly completed four CPU
  updates, validation on three pairs, and model saving.
- Bash syntax checks and TIGRIS `sbatch --test-only` accepted both launchers.
  These were scheduler dry runs; no training job was submitted.

### Initial Llama trainer

- Two focused tests pass: HS3 preference direction/strength, and the scaled-BT
  loss value and gradient.
- A tiny random Llama completed four updates in one epoch, validation and model
  saving on local CPU (PyTorch 2.6.0).
- The same trainer completed four updates using **two FSDP processes** on CPU.
  Validation correctly counted three original pairs despite sampler padding.
- Both ordinary and distributed checkpoints were reloaded. Backbone attention
  weights changed, confirming full-model training rather than head-only fitting.
- TIGRIS CPU execution passed the same four-update training exercise with Python
  3.11.7, PyTorch 2.1.1, Transformers 4.46.3 and Accelerate 1.2.1.
- The pinned Llama 3.3 tokenizer is accessible on TIGRIS (128,256 vocabulary
  entries). SentencePiece 0.2.1 fixes the inherited Spack binary's static-TLS
  import conflict.
- The full 80-layer architecture initializes on the meta device: 69,503,041,536
  reward-model parameters. No large weight allocation was needed for this check.
- Slurm dry runs accepted the production launcher: 16 GH200s, four CPU cores and
  384 GiB host RAM per node, 24-hour limit. No full 70B training was submitted.
- Real HS3 preparation passed CPU-only TIGRIS job **364545** (exit 0, 3m30s):
  28,431 training pairs and 1,582 validation pairs. Their prompt IDs are disjoint;
  preference strengths, sequence lengths, file hashes and NVIDIA template hash
  were checked. Model weight downloading was skipped for this data check.

The tiny-model tests verify code paths, not 70B memory use, GPU throughput or
the paper's benchmark results. Full-size GPU execution remains unverified.
