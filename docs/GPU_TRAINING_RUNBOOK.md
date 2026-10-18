# ASAP7 GPU Training Runbook

## Material Passport

- Artifact type: reproducible experiment runbook
- Scope: ASAP7 global feature-conditioned ANN
- Status: GPU-ready procedure; formal GPU runtime not yet measured
- Split contracts: 28
- Formal model runs: 40 (unseen-corner expands 3 folds × 5 model seeds)

## 1. Clone and create the GPU environment

```bash
git clone <YOUR_REPOSITORY_URL> nldm-ann
cd nldm-ann
conda env create -f environment-gpu.yml
conda activate nldm-gpu
```

Do not use the original `environment.yml` for formal GPU runs: it intentionally contains
`cpuonly`. The GPU environment requests CUDA 12.1. Confirm that the installed NVIDIA driver
supports it before training.

## 2. Supply the dataset

`data/processed/asap7_nldm.parquet` is intentionally excluded from Git. Use either method:

1. Copy the existing 67 MB Parquet and its metadata/audit files to
   `data/processed/`; or
2. Rebuild it from OpenROAD-flow-scripts commit
   `da37dce1cef69bb75efbc2dc18ed95a29651295b` using
   `scripts/build_asap7_manifest.py`, `scripts/extract_asap7_dataset.py`, and
   `scripts/audit_asap7_dataset.py`.

The required dataset SHA-256 is recorded in
`data/processed/asap7_nldm.metadata.json`; a different hash must not be used with the
committed split contracts.

## 3. Mandatory preflight

```bash
python scripts/gpu_preflight.py \
  --parquet data/processed/asap7_nldm.parquet \
  --metadata data/processed/asap7_nldm.metadata.json \
  --manifest-dir experiments/asap7_generalization_v1/manifests
```

Proceed only when it prints `"status": "PASS"`, `"cuda_available": true`, the expected
GPU name, 12,073,208 rows, and 28 split contracts.

## 4. Timed smoke run

```bash
python src/run_generalization.py \
  --parquet data/processed/asap7_nldm.parquet \
  --manifest-dir experiments/asap7_generalization_v1/manifests \
  --out-dir experiments/asap7_gpu_smoke \
  --device cuda --epochs 2 --patience 2 \
  --batch-rows 250000 --micro-batch 8192 \
  --max-rows 200000 --limit 1
```

This is pipeline/runtime evidence only, not a paper result. Record GPU name, peak VRAM,
wall-clock time, and rows/second before choosing the final micro-batch size. GTX 1070 may
need 2,048–4,096; RTX 3090 should start at 8,192 and can be increased after observing VRAM.

## 5. Formal run

```bash
python src/run_generalization.py \
  --parquet data/processed/asap7_nldm.parquet \
  --manifest-dir experiments/asap7_generalization_v1/manifests \
  --out-dir experiments/asap7_generalization_v1/runs \
  --device cuda --epochs 50 --patience 8 \
  --batch-rows 250000 --micro-batch 8192
```

Never pass `--max-rows` for formal results. The runner reuses completed `model.pt` files and
automatically resumes interrupted training from `model.last.pt`. It does not upload results.

## 6. Readiness and interpretation boundaries

- Normalization is fit only on optimization rows, never on validation/test rows.
- Validation is deterministically drawn from the contract's training partition.
- Cell/corner IDs are absent from model inputs.
- Joint cross-quadrants are excluded from the primary train/test comparison.
- ASAP7 P/V/T values are coupled; results do not identify isolated causal P, V, or T effects.
- Smoke metrics and capped runs must never be copied into thesis result tables.

