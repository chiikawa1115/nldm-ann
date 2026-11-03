"""Export small metadata JSONs from this project's own ANN checkpoints.

Run on the original training host. Only load model.pt files you produced and
trust: PyTorch checkpoints may contain Python pickle objects. No training runs.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ridge_audit import read_json, repo_path, write_json
from global_engine import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", default="experiments/configs/global_ann_ridge_v1.json")
    parser.add_argument("--out-dir", default="experiments/global_ann_ridge_metadata")
    args = parser.parse_args()
    import torch
    root = args.root.resolve()
    config = read_json(repo_path(root, args.config))
    out = repo_path(root, args.out_dir)
    for pair in config["pairs"]:
        ann = read_json(repo_path(root, pair["ann_metrics"]))
        path = repo_path(root, ann["checkpoint"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        keys = ("seed", "contract", "manifest_sha256", "normalization", "max_rows", "git_commit",
                "training_config", "best_validation_mse_norm", "package_versions", "python_version")
        meta = {key: checkpoint[key] for key in keys if key in checkpoint}
        hashes = {}
        commit = checkpoint.get("git_commit")
        for source in config["reference_source_git_blob_sha256"]:
            result = subprocess.run(["git", "show", f"{commit}:{source}"], cwd=root,
                                    capture_output=True) if commit else None
            hashes[source] = hashlib.sha256(result.stdout).hexdigest() if result and result.returncode == 0 else None
        meta.update({"schema_version": 1, "run_id": pair["run_id"],
                     "checkpoint_sha256": file_sha256(path), "training_source_sha256": hashes,
                     "completed_epochs": len(checkpoint.get("history", [])),
                     "source_caveat": "Git commit sources do not independently attest absence of uncommitted edits during ANN training."})
        write_json(out / (pair["run_id"] + ".json"), meta)
        print("exported=" + pair["run_id"], flush=True)


if __name__ == "__main__":
    main()
