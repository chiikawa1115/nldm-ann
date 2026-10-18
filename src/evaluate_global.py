"""Evaluate a global feature-conditioned checkpoint on its immutable test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from global_engine import NormalizationStats, file_sha256, iter_tensors, load_contract
from metrics import micro_macro_metrics
from models.mlp_nldm import FeatureConditionedNldm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-rows", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true",
                        help="persist point predictions; off by default to reduce RAM/disk")
    parser.add_argument("--skip-dataset-hash", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.manifest, args.parquet, not args.skip_dataset_hash)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["contract"] != contract:
        raise ValueError("checkpoint and requested split contracts differ")
    if checkpoint.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("checkpoint manifest SHA-256 differs from requested manifest")
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    model = FeatureConditionedNldm(
        len(stats.feature_names), tuple(checkpoint["hidden_dims"]), out_features=1
    )
    model.load_state_dict(checkpoint["model_state"])
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto": device_name = "cpu"
    device = torch.device(device_name); model.to(device).eval()
    pieces = []
    with torch.no_grad():
        for x, _, source in iter_tensors(
            args.parquet, contract, stats, "test", args.batch_rows, args.max_rows
        ):
            prediction = model(x.to(device)).cpu().numpy().ravel()
            prediction = prediction * stats.y_std + stats.y_mean
            payload = {
                "cell_base": source["cell_base"].to_numpy(),
                "y_true": source["value"].to_numpy(dtype=float),
                "y_pred": prediction.astype(float),
            }
            if args.save_predictions:
                payload.update({
                    "cell": source["cell"].to_numpy(),
                    "corner": source["corner"].to_numpy(),
                    "table_type": source["table_type"].to_numpy(),
                })
            pieces.append(pd.DataFrame(payload))
    if not pieces:
        raise RuntimeError("test partition is empty")
    result = pd.concat(pieces, ignore_index=True)
    micro, macro, per_cell = micro_macro_metrics(result, group_col="cell_base")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        result.to_parquet(args.out / "predictions.parquet", index=False)
    per_cell.to_csv(args.out / "per_cell_metrics.csv", index=False)
    summary = {
        "schema_version": 1, "mode": contract["mode"],
        "seed": checkpoint["seed"], "micro": micro, "macro": macro,
        "heldout_pvt": contract.get("heldout_pvt"), "fold": contract.get("fold"),
        "n_negative_targets": int((result.y_true < 0).sum()),
        "checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
