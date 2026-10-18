# GPU Training Readiness Report — 2026-08-15

## Material Passport

- Artifact type: implementation readiness validation
- Project: `nldm-ann`
- Scope: ASAP7 feature-conditioned generalization training
- Verification status: CPU pipeline VERIFIED; CUDA execution NOT RUN on this host
- Upload status: not committed or pushed

## Outcome

The repository-side preparation needed before moving to a GPU host is complete. Formal
training has not started, and no smoke result is eligible as thesis evidence.

## Passed gates

- Dataset: 12,073,208 rows; SHA-256
  `79ae211cec386a170455f6fdf0d38c1043d835786408d5aa93c94a1bec855ba5`.
- Split contracts: 28 files plus index; all indexed hashes verified.
- Formal run expansion: 40 runs, because 3 unseen-corner folds each use 5 model seeds.
- Unseen-cell holdout: 40 of 200 `cell_base` topologies for every split seed.
- Model input: explicit cell/PVT features; no cell/corner ID embedding.
- Data loading: streaming Parquet batches; full Liberty tables are not split at batch boundaries.
- Random-point split: exact rounded count per table with deterministic full table+point identity.
- Joint split: cross-quadrants excluded from primary training and testing.
- Normalization: fit only on optimization rows; zero-variance features use scale 1.
- Validation: deterministic subset of the contract training partition.
- Checkpoint: best model, last model, optimizer, RNG, normalization, manifest/dataset provenance,
  package versions, device and training configuration.
- Resume: interruption after epoch 1 was successfully resumed at epoch 2.
- Evaluation: immutable manifest/hash verification, micro and equal-cell macro metrics,
  per-cell CSV, optional point predictions, negative-target count.
- Aggregation: per-run records plus mean, sample SD and Student-t 95% CI by mode/scope and corner.
- Orchestration: restartable runner reuses completed models and resumes `.last.pt` files.
- Automated tests: 24 passed.
- Static Python compilation: passed for `src/` and `scripts/`.
- Local preflight: passed with `--allow-cpu`; row count, dataset hash and all manifest hashes match.
- Real ASAP7 smoke: train → checkpoint → evaluate → aggregate completed on CPU with capped rows.

## Mandatory GPU-host gate

This machine has CPU-only PyTorch, so GTX 1070/RTX 3090 execution cannot be claimed as
verified. On the GPU host, follow `docs/GPU_TRAINING_RUNBOOK.md` and require:

1. `scripts/gpu_preflight.py` passes without `--allow-cpu`.
2. `torch.cuda.is_available()` is true and the expected GPU name is printed.
3. One capped 2-epoch timed smoke completes and records wall time/VRAM.
4. Only then run the 40 formal jobs without `--max-rows`.

## Files used on the GPU host

- `environment-gpu.yml`
- `docs/GPU_TRAINING_RUNBOOK.md`
- `scripts/gpu_preflight.py`
- `src/global_engine.py`
- `src/train_global.py`
- `src/evaluate_global.py`
- `src/run_generalization.py`
- `src/aggregate_global.py`
- `experiments/asap7_generalization_v1/manifests/`

The Parquet itself is intentionally excluded from Git and must be copied or rebuilt from
the pinned OpenROAD-flow-scripts commit before preflight.

