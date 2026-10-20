# Initial validation

Pipeline source: `20a48192b0bc7991e26e37cb0ff780919a5954bd`.

- Three focused tests passed locally and again from a fresh GitHub clone.
- A real HS3 CPU smoke run completed preparation, embedding and both treatments
  with Python 3.12.12, PyTorch 2.6.0 and Transformers 4.46.3.
- The smoke config selected 8/4/4 train/validation/test prompts and a 1,024-token
  limit. After whole-pair length filtering, 7/2/3 pairs remained. Both heads
  finished one epoch (one optimizer step), saved and reloaded their selected
  checkpoint, and produced finite held-out loss on the same three test pairs.
- Actual artifact prompt IDs were disjoint across train, validation and test.
- Training from the fresh clone reproduced the local smoke outputs exactly.
- TIGRIS setup/imports passed with Spack `default-ml-aarch64-25050701`, Python
  3.11.7, PyTorch 2.1.1 and Transformers 4.46.3.
- The default config prepared 264/65/66 pairs from 256/64/64 prompts on TIGRIS.
- Source, configuration, requirements and launcher hashes match the local release.
- Slurm accepted GPU job **364331** on cluster **tigris**, account **loop**:
  one GH200, four CPU cores, 16 GB RAM, 30-minute limit. It was pending for
  priority at this validation snapshot. GPU execution is not yet verified.
- A 10-minute scheduler dry run did not improve the forecast; the submitted job
  and its queue age were preserved.

The CPU smoke metrics verify execution only. This single-seed, tiny-cohort run
does not establish a fairness or accuracy benefit.

Remote source copy: `/home/xx4455/hs3-tigris`. Once the GPU job finishes, inspect
`results/job-364331/`, its completion manifests, Slurm accounting and error log.
