"""Aggregate global generalization metrics across model seeds and PVT folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

METRICS = ("MAE", "RMSE", "sMAPE_pct", "P90RE_pct", "P99RE_pct", "R2")


def load_records(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*/evaluation/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        heldout = payload.get("heldout_pvt") or {}
        for scope in ("micro", "macro"):
            row = {
                "run": path.parents[1].name, "mode": payload["mode"],
                "model_seed": payload["seed"], "scope": scope,
                "heldout_corner": heldout.get("corner"),
                "fold": payload.get("fold"),
            }
            row.update({name: payload[scope][name] for name in METRICS})
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no metrics.json files below {runs_dir}")
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in records.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values); mean = float(values.mean()); sd = float(values.std(ddof=1)) if n > 1 else 0.0
            half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
            rows.append({**base, "metric": metric, "n_runs": n, "mean": mean,
                         "standard_deviation": sd, "ci95_low": mean - half,
                         "ci95_high": mean + half})
    return pd.DataFrame(rows)


def aggregate(runs_dir: Path, out_dir: Path) -> None:
    records = load_records(runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_dir / "run_metrics.csv", index=False)
    summarize(records, ["mode", "scope"]).to_csv(
        out_dir / "summary_by_mode.csv", index=False
    )
    summarize(records, ["mode", "heldout_corner", "scope"]).to_csv(
        out_dir / "summary_by_mode_corner.csv", index=False
    )
    print(f"runs={records['run'].nunique()} records={len(records)} out={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.runs_dir, args.out_dir)


if __name__ == "__main__":
    main()
