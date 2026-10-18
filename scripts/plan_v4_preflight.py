"""PLAN v4 preflight: run on the GPU host before any formal training.

Checks, in order, and stops at the first failure:
  1. dataset matches the committed reference (file hash, else content hash)
  2. PLAN v4 manifests match their recorded hashes
  3. feature cache exists for this dataset + feature code, and its matrix hash
     equals the cache JSON (optionally also another machine's JSON)
  4. CUDA really computes: a matmul and a small MLP train step on the GPU are
     compared against the same computation on CPU. `is_available()` alone is
     not trusted, because a PyTorch build without kernels for the card's
     architecture (e.g. Pascal / sm_61 on GTX 1070) can report True and fail
     only when a kernel launches.
Writes the result (PASS/FAIL + versions) to experiments/plan_v4/preflight_<host>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from plan_v4_data import verify_parquet, write_json  # noqa: E402

# Cache JSON from the dev machine, committed so other hosts can compare matrix hashes.
DEV_CACHE_REFERENCE = REPO / "experiments/plan_v4/feature_cache_reference.json"


def check_manifests(manifest_dir: Path) -> dict:
    index = json.loads((manifest_dir / "manifest_index.json").read_text(encoding="utf-8"))
    for item in index["manifests"] + [index["L0_table_sample"]]:
        path = manifest_dir / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {path} (CRLF checkout? see runbook 1)")
    return {"count": index["count"], "by_experiment": index["by_experiment"]}


def check_cache(cache_dir: Path, dataset_sha: str, compare_with: Path | None) -> dict:
    from plan_v4_w2 import FEATURE_CODE, code_hash
    stem = f"features_{dataset_sha[:12]}_{code_hash(FEATURE_CODE)[:12]}"
    meta_path = cache_dir / f"{stem}.json"
    if not meta_path.exists():
        raise RuntimeError(f"feature cache {stem} missing; run scripts/plan_v4_w2.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    matrix = np.load(cache_dir / f"{stem}.npy", mmap_mode="r")
    digest = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    if digest != meta["matrix_sha256"]:
        raise RuntimeError("feature cache .npy does not match its JSON (partial write?)")
    out = {"stem": stem, "shape": list(matrix.shape), "matrix_sha256": digest}
    if compare_with is None and DEV_CACHE_REFERENCE.exists():
        dev = json.loads(DEV_CACHE_REFERENCE.read_text(encoding="utf-8"))
        if dev["feature_code_sha256"] == meta["feature_code_sha256"]:
            compare_with = DEV_CACHE_REFERENCE
    if compare_with is not None:
        other = json.loads(compare_with.read_text(encoding="utf-8"))
        out["matches_other_machine"] = other["matrix_sha256"] == digest
        if not out["matches_other_machine"]:
            raise RuntimeError(f"feature matrix differs from {compare_with}")
    return out


def check_cuda(allow_cpu: bool) -> dict:
    import torch
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        if allow_cpu:
            return {**info, "compute_test": "skipped (--allow-cpu)"}
        raise RuntimeError("torch.cuda.is_available() is False")
    props = torch.cuda.get_device_properties(0)
    info.update({"gpu": props.name, "capability": f"sm_{props.major}{props.minor}",
                 "vram_gb": round(props.total_memory / 2**30, 1),
                 "arch_list": torch.cuda.get_arch_list(),
                 "driver_cuda": getattr(torch._C, "_cuda_getDriverVersion", lambda: None)()})
    arch = f"sm_{props.major}{props.minor}"
    if info["arch_list"] and arch not in info["arch_list"] and not any(
            a.startswith("compute_") and int(a.split("_")[1]) <= props.major * 10 + props.minor
            for a in info["arch_list"]):
        raise RuntimeError(f"this PyTorch build has no kernels for {arch}: {info['arch_list']}; "
                           "install an older CUDA wheel (see runbook 2)")
    torch.manual_seed(0)
    a, b = torch.randn(512, 512), torch.randn(512, 512)
    gpu = (a.cuda() @ b.cuda()).cpu()
    if not torch.allclose(gpu, a @ b, rtol=1e-3, atol=1e-3):
        raise RuntimeError("GPU matmul result differs from CPU")

    def train_step(device: str) -> float:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(69, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, y = torch.randn(4096, 69, device=device), torch.randn(4096, 1, device=device)
        for _ in range(5):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x.to(device)), y)
            loss.backward()
            opt.step()
        return float(loss)

    gpu_loss, cpu_loss = train_step("cuda"), train_step("cpu")
    if not np.isfinite(gpu_loss) or abs(gpu_loss - cpu_loss) > 1e-3 * max(1.0, abs(cpu_loss)):
        raise RuntimeError(f"GPU train step loss {gpu_loss} differs from CPU {cpu_loss}")
    torch.cuda.synchronize()
    start = time.time()
    big = torch.randn(4096, 4096, device="cuda")
    for _ in range(10):
        big = big @ big / 64
    torch.cuda.synchronize()
    info.update({"compute_test": "PASS", "train_step_loss_gpu": gpu_loss,
                 "train_step_loss_cpu": cpu_loss,
                 "matmul_4096_x10_seconds": round(time.time() - start, 3)})
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=REPO / "data/processed/asap7_nldm.parquet")
    parser.add_argument("--reference", type=Path, default=REPO / "data/asap7_nldm.reference.json")
    parser.add_argument("--manifest-dir", type=Path, default=REPO / "experiments/plan_v4/manifests")
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data/cache/plan_v4")
    parser.add_argument("--compare-cache-json", type=Path,
                        help="cache JSON from another machine to compare the matrix hash with")
    parser.add_argument("--skip-cache", action="store_true", help="before plan_v4_w2.py has run")
    parser.add_argument("--allow-cpu", action="store_true", help="dev machine dry run only")
    args = parser.parse_args()
    report = {"host": socket.gethostname(), "platform": platform.platform(),
              "python": platform.python_version(), "numpy": np.__version__}
    steps = [
        ("dataset", lambda: {k: v for k, v in verify_parquet(args.parquet, args.reference).items()
                             if k in ("verified_by", "rows", "parquet_sha256",
                                      "local_parquet_sha256")}),
        ("manifests", lambda: check_manifests(args.manifest_dir)),
        ("cache", lambda: {"skipped": True} if args.skip_cache else check_cache(
            args.cache_dir, json.loads(args.reference.read_text(encoding="utf-8"))
            ["parquet_sha256"], args.compare_cache_json)),
        ("cuda", lambda: check_cuda(args.allow_cpu)),
    ]
    status = "PASS"
    for name, step in steps:
        try:
            report[name] = step()
            print(f"[PASS] {name}")
        except Exception as exc:  # report every failure kind the same way
            report[name] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"[FAIL] {name}: {exc}")
            status = "FAIL"
            break
    report["status"] = status
    out = REPO / "experiments/plan_v4" / f"preflight_{report['host']}.json"
    write_json(out, report)
    print(f"{status} -> {out}")
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
