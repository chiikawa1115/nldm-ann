"""Restartable orchestrator for all ASAP7 generalization contracts and model seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SEEDS = (11, 23, 37, 53, 71)


def runs(manifest_dir: Path):
    index = json.loads((manifest_dir / "manifest_index.json").read_text(encoding="utf-8"))
    for item in index["manifests"]:
        manifest = manifest_dir / item["path"]
        contract = json.loads(manifest.read_text(encoding="utf-8"))
        if contract["mode"] == "unseen_corner":
            for seed in SEEDS:
                yield manifest, contract, seed
        else:
            yield manifest, contract, int(contract["split_seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-rows", type=int, default=250_000)
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="smoke-only; omit for formal runs")
    parser.add_argument("--mode", choices=["random_point", "unseen_cell", "unseen_corner", "joint_unseen"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reevaluate", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    selected = list(runs(args.manifest_dir))
    if args.mode:
        selected = [item for item in selected if item[1]["mode"] == args.mode]
    if args.limit is not None:
        selected = selected[:args.limit]
    for manifest, contract, seed in selected:
        run_name = f"{manifest.stem}_modelseed{seed}"
        run_dir = args.out_dir / run_name
        checkpoint = run_dir / "model.pt"
        metrics = run_dir / "evaluation" / "metrics.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            command = [
                sys.executable, str(root / "src" / "train_global.py"),
                "--parquet", str(args.parquet), "--manifest", str(manifest),
                "--out", str(checkpoint), "--device", args.device,
                "--epochs", str(args.epochs), "--patience", str(args.patience),
                "--batch-rows", str(args.batch_rows), "--micro-batch", str(args.micro_batch),
                "--seed", str(seed),
            ]
            last = checkpoint.with_suffix(".last.pt")
            if last.exists(): command.extend(["--resume", str(last)])
            if args.max_rows is not None: command.extend(["--max-rows", str(args.max_rows)])
            subprocess.run(command, cwd=root, check=True)
        if not metrics.exists() or args.reevaluate:
            command = [
                sys.executable, str(root / "src" / "evaluate_global.py"),
                "--parquet", str(args.parquet), "--manifest", str(manifest),
                "--checkpoint", str(checkpoint), "--out", str(run_dir / "evaluation"),
                "--device", args.device, "--batch-rows", str(args.batch_rows),
            ]
            if args.max_rows is not None: command.extend(["--max-rows", str(args.max_rows)])
            subprocess.run(command, cwd=root, check=True)
    if selected:
        subprocess.run([
            sys.executable, str(root / "src" / "aggregate_global.py"),
            "--runs-dir", str(args.out_dir), "--out-dir", str(args.out_dir / "aggregate"),
        ], cwd=root, check=True)
    print(f"completed_or_reused={len(selected)} out={args.out_dir}")


if __name__ == "__main__":
    main()
