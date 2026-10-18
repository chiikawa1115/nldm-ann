"""Train one global feature-conditioned NLDM model from a compact split contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from global_engine import (
    NormalizationStats, file_sha256, fit_normalization, iter_tensors, load_contract,
    seed_everything,
)
from models.mlp_nldm import FeatureConditionedNldm


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    names = ("torch", "numpy", "pandas", "pyarrow", "scipy")
    versions = {}
    for name in names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return versions


def loss_on_partition(model, parquet, contract, stats, partition, device, batch_rows, max_rows):
    model.eval(); total = count = 0
    with torch.no_grad():
        for x, y, _ in iter_tensors(
            parquet, contract, stats, partition, batch_rows, max_rows=max_rows
        ):
            x = x.to(device); y = y.to(device)
            total += torch.nn.functional.mse_loss(model(x), y, reduction="sum").item()
            count += len(y)
    return total / count if count else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-rows", type=int, default=250_000)
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128, 64])
    parser.add_argument("--max-rows", type=int, default=None,
                        help="smoke-only cap per partition; omit for the real experiment")
    parser.add_argument("--skip-dataset-hash", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="model/validation seed; defaults to the split seed")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume from a .last.pt checkpoint created by this CLI")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    manifest_contract = load_contract(args.manifest, args.parquet, not args.skip_dataset_hash)
    seed = int(args.seed if args.seed is not None else
               manifest_contract.get("split_seed", manifest_contract.get("fold", 0)))
    contract = {**manifest_contract, "_run_seed": seed}
    seed_everything(seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto": device_name = "cpu"
    device = torch.device(device_name)
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume is not None and resume["contract"] != manifest_contract:
        raise ValueError("resume checkpoint and requested split contracts differ")
    if resume is not None and int(resume["seed"]) != seed:
        raise ValueError("resume checkpoint model seed differs from --seed")
    if resume is not None and list(resume["hidden_dims"]) != list(args.hidden_dims):
        raise ValueError("resume checkpoint architecture differs from --hidden-dims")
    stats = (
        fit_normalization(args.parquet, contract, args.batch_rows, args.max_rows)
        if resume is None else NormalizationStats.from_dict(resume["normalization"])
    )
    model = FeatureConditionedNldm(
        len(stats.feature_names), tuple(args.hidden_dims), out_features=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf"); best_state = None; stale = 0; history = []; start_epoch = 1
    if resume is not None:
        model.load_state_dict(resume["last_model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        best_state = resume["best_model_state"]
        best_loss = float(resume["best_validation_mse_norm"])
        stale = int(resume["stale_epochs"])
        history = list(resume["history"])
        start_epoch = int(resume["last_epoch"]) + 1
        torch.set_rng_state(resume["torch_rng_state"])
        if torch.cuda.is_available() and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
    started = time.time()
    last_path = args.out.with_suffix(".last.pt")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total = count = 0
        for x_chunk, y_chunk, _ in iter_tensors(
            args.parquet, contract, stats, "fit", args.batch_rows, args.max_rows
        ):
            order = torch.randperm(len(x_chunk))
            for start in range(0, len(order), args.micro_batch):
                index = order[start:start + args.micro_batch]
                x = x_chunk[index].to(device); y = y_chunk[index].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(x), y)
                loss.backward(); optimizer.step()
                total += loss.item() * len(y); count += len(y)
        validation = loss_on_partition(
            model, args.parquet, contract, stats, "validation", device,
            args.batch_rows, args.max_rows,
        )
        row = {"epoch": epoch, "fit_mse_norm": total / count, "validation_mse_norm": validation}
        history.append(row); print(json.dumps(row), flush=True)
        if np.isfinite(validation) and validation < best_loss:
            best_loss = validation; stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1, "contract": manifest_contract, "seed": seed,
            "normalization": stats.to_dict(), "hidden_dims": args.hidden_dims,
            "last_model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "best_model_state": best_state, "optimizer_state": optimizer.state_dict(),
            "best_validation_mse_norm": best_loss, "stale_epochs": stale,
            "history": history, "last_epoch": epoch,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, last_path)
        if stale >= args.patience: break
    if best_state is None:
        raise RuntimeError("validation partition produced no finite loss")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1, "model_class": "FeatureConditionedNldm",
        "model_state": best_state, "hidden_dims": args.hidden_dims,
        "normalization": stats.to_dict(), "contract": manifest_contract,
        "manifest_path": str(args.manifest), "history": history,
        "best_validation_mse_norm": best_loss, "seed": seed,
        "runtime_seconds": time.time() - started, "device": str(device),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": git_commit(root), "max_rows": args.max_rows,
        "manifest_sha256": file_sha256(args.manifest),
        "training_config": {
            "epochs": args.epochs, "patience": args.patience,
            "batch_rows": args.batch_rows, "micro_batch": args.micro_batch,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "hidden_dims": args.hidden_dims,
        },
    }
    torch.save(checkpoint, args.out)
    args.out.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={args.out} best_validation_mse_norm={best_loss:.8g}")


if __name__ == "__main__":
    main()
