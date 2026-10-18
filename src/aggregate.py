"""Aggregate repeated NLDM experiments into paper-ready statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t

METRICS = ["MAE", "RMSE", "sMAPE_pct", "P50RE_pct", "P90RE_pct", "P99RE_pct", "R2"]


def load_run_records(result_path: Path, seed: int) -> list[dict]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows: list[dict] = []

    def add_scope(scope: str, methods: dict) -> None:
        for method in ("MLP", "LUT"):
            values = methods[method]
            for metric in METRICS:
                rows.append({
                    "seed": seed, "holdout_ratio": result["holdout_ratio"],
                    "requested_holdout_ratio": result.get(
                        "requested_holdout_ratio", result["holdout_ratio"]
                    ),
                    "scope": scope, "method": method, "metric": metric,
                    "value": values[metric], "result_file": str(result_path),
                })

    add_scope("overall", result["overall"])
    for table_type, methods in result["per_table_type"].items():
        add_scope(table_type, methods)
    return rows


def summarize_records(records: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    required = {"seed", "holdout_ratio", "scope", "method", "metric", "value"}
    if missing := required - set(records.columns):
        raise ValueError(f"missing record columns: {sorted(missing)}")
    rows = []
    keys = ["holdout_ratio", "scope", "method", "metric"]
    for key, group in records.groupby(keys, sort=True):
        values = group["value"].astype(float)
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else float("nan")
        half = float(t.ppf((1 + confidence) / 2, n - 1) * std / math.sqrt(n)) if n > 1 else float("nan")
        rows.append(dict(zip(keys, key), n_seeds=n, mean=mean, std=std,
                         ci_low=mean - half, ci_high=mean + half,
                         confidence=confidence, ci_method="student_t_across_seeds"))
    return pd.DataFrame(rows)


def paired_differences(records: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Summarize paired MLP-LUT differences; negative error deltas favor MLP."""
    index = ["seed", "holdout_ratio", "scope", "metric"]
    wide = records.pivot(index=index, columns="method", values="value").reset_index()
    if not {"MLP", "LUT"} <= set(wide.columns):
        raise ValueError("paired comparison requires both MLP and LUT for every group")
    wide["value"] = wide["MLP"] - wide["LUT"]
    wide["method"] = "MLP_minus_LUT"
    return summarize_records(wide, confidence)


def write_aggregate(records: pd.DataFrame, output_dir: Path, confidence: float = 0.95) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_records(records, confidence)
    paired = paired_differences(records, confidence)
    records.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_ci.csv", index=False)
    paired.to_csv(output_dir / "paired_difference_ci.csv", index=False)
    metadata = {
        "schema_version": 1, "confidence": confidence,
        "ci_method": "two-sided Student t interval across fixed seeds",
        "paired_difference": "MLP minus LUT; negative error-metric values favor MLP",
        "seeds": sorted(int(x) for x in records["seed"].unique()),
    }
    (output_dir / "aggregate_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def plot_overall(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ["RMSE", "sMAPE_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    overall = summary[(summary["scope"] == "overall") & summary["metric"].isin(metrics)]
    for axis, metric in zip(axes, metrics):
        for method, group in overall[overall["metric"] == metric].groupby("method"):
            group = group.sort_values("holdout_ratio")
            axis.plot(group["holdout_ratio"], group["mean"], marker="o", label=method)
            axis.fill_between(group["holdout_ratio"], group["ci_low"], group["ci_high"], alpha=0.2)
        axis.set(xlabel="Holdout ratio", ylabel=metric, title=f"{metric} (95% CI)")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
