"""Paired, per-mode ANN/Ridge summaries; incomplete/pending evidence stays visible."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from ridge_audit import write_json

METRICS = ("MAE", "RMSE", "P50RE_pct", "P90RE_pct", "P99RE_pct", "R2", "sMAPE_pct")
PAIR_COLUMNS = ("run_id", "mode", "seed", "scope", "metric", "ann", "ridge", "delta",
                "improvement_pct", "ann_provenance_status", "delta_direction")


def paired_rows(run: str, mode: str, seed: int, ann: dict, ridge: dict, status: str) -> list[dict]:
    rows = []
    for scope in ("micro", "macro"):
        for metric in METRICS:
            a, r = ann[scope].get(metric), ridge[scope].get(metric)
            a = float(a) if a is not None else float("nan")
            r = float(r) if r is not None else float("nan")
            valid = math.isfinite(a) and math.isfinite(r)
            delta = (a - r if metric == "R2" else r - a) if valid else float("nan")
            improvement = (100 * delta / r if valid and metric != "R2" and r != 0
                           else float("nan"))
            rows.append({"run_id": run, "mode": mode, "seed": seed, "scope": scope,
                         "metric": metric, "ann": a, "ridge": r, "delta": delta,
                         "improvement_pct": improvement, "ann_provenance_status": status,
                         "delta_direction": "ANN-Ridge" if metric == "R2" else "Ridge-ANN"})
    return rows


def holm_adjust(pvalues: list[float]) -> list[float]:
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("Holm correction requires finite probabilities")
    order = np.argsort(pvalues, kind="stable")
    result = np.empty(len(pvalues))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(pvalues) - rank) * pvalues[index])
        result[index] = min(1.0, previous)
    return result.tolist()


def summarize_pairs(records: pd.DataFrame, config: dict) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["mode", "scope", "metric", "n_pairs", "expected_pairs",
                                     "delta_mean", "delta_sd", "delta_ci95_low", "delta_ci95_high"])
    keys = ["mode", "seed", "scope", "metric"]
    if records.duplicated(keys).any():
        raise ValueError("duplicate paired observations")
    if not records.ann_provenance_status.isin(["matched", "ann_provenance_pending"]).all():
        raise ValueError("mismatched or unknown provenance must be excluded before summarizing")
    rows = []
    for (mode, scope, metric), g in records.groupby(["mode", "scope", "metric"], sort=True):
        if mode not in config["modes"] or not set(g.seed).issubset(config["seeds"]):
            raise ValueError("unplanned mode or seed in comparison")
        valid = g.loc[np.isfinite(g.delta)]
        x = valid.delta.to_numpy(dtype=float)
        n = len(x)
        mean = float(x.mean()) if n else float("nan")
        sd = float(x.std(ddof=1)) if n > 1 else float("nan")
        half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
        row = {"mode": mode, "scope": scope, "metric": metric, "n_pairs": n,
               "expected_pairs": len(config["seeds"]),
               "n_matched": int((valid.ann_provenance_status == "matched").sum()),
               "n_provenance_pending": int((valid.ann_provenance_status == "ann_provenance_pending").sum()),
               "n_undefined": len(g) - n, "delta_mean": mean, "delta_sd": sd,
               "delta_ci95_low": mean - half, "delta_ci95_high": mean + half,
               "improvement_pct_mean": float(valid.improvement_pct.mean()),
               "improvement_pct_n": int(valid.improvement_pct.notna().sum()),
               "inference_status": "descriptive_only",
               "p_two_sided": float("nan"), "p_holm": float("nan"),
               "formal_inference_eligible": False}
        for model in ("ann", "ridge"):
            row[model + "_mean"] = float(valid[model].mean())
            row[model + "_sd"] = float(valid[model].std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    primary_indices = result.index[(result.scope == "macro") & (result.metric == "MAE")].tolist()
    eligible = (len(primary_indices) == len(config["modes"]) == 2
                and len(config["seeds"]) == 5
                and all(result.loc[i, "n_pairs"] == len(config["seeds"])
                        and result.loc[i, "n_matched"] == len(config["seeds"])
                        and np.isfinite(result.loc[i, "delta_sd"])
                        and result.loc[i, "delta_sd"] > 0
                        for i in primary_indices))
    if eligible:
        pvalues = []
        for i in primary_indices:
            r = result.loc[i]
            statistic = abs(r.delta_mean) / (r.delta_sd / np.sqrt(r.n_pairs))
            p = float(2 * t.sf(statistic, r.n_pairs - 1))
            pvalues.append(p)
            result.loc[i, "p_two_sided"] = p
            result.loc[i, "formal_inference_eligible"] = True
            result.loc[i, "inference_status"] = "paired_t_holm"
        for i, p in zip(primary_indices, holm_adjust(pvalues)):
            result.loc[i, "p_holm"] = p
    return result


def write_comparison(output: Path, config: dict, rows: list[dict], run_states: list[dict],
                     make_plots: bool = True) -> None:
    aggregate = output / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    summary = summarize_pairs(records, config)
    records.to_csv(aggregate / "paired_metrics.csv", index=False)
    summary.to_csv(aggregate / "summary_by_mode.csv", index=False)
    complete = [r for r in run_states if r["status"] == "complete"]
    status = {"expected_runs": config["expected_runs"], "completed_runs": len(complete),
              "all_runs_complete": len(complete) == config["expected_runs"],
              "all_ann_provenance_matched": len(complete) == config["expected_runs"] and all(
                  r.get("ann_provenance_status") == "matched" for r in complete),
              "figures_updated": make_plots,
              "runs": run_states}
    write_json(aggregate / "comparison_status.json", status)
    lines = ["# Global ANN / Ridge paired accuracy comparison", "",
             config["research_question"], "",
             f"Completed planned runs: {len(complete)}/{config['expected_runs']}.", "",
             "Target scope: mixed delay + transition, with the frozen 42 input features.",
             "Ridge uses fit-only scaling, an unpenalized intercept and the frozen six-lambda grid; validation MSE selects lambda without refitting on validation.",
             "Positive error delta (Ridge - ANN) means lower ANN error. R2 uses ANN - Ridge.",
             "", "| Mode | Pairs | ANN macro MAE | Ridge macro MAE | Delta | 95% CI |",
             "|---|---:|---:|---:|---:|---|"]
    if not summary.empty:
        for r in summary.loc[(summary.scope == "macro") & (summary.metric == "MAE")].itertuples():
            lines.append(f"| {r.mode} | {r.n_pairs}/{r.expected_pairs} | {r.ann_mean:.6g} | "
                         f"{r.ridge_mean:.6g} | {r.delta_mean:.6g} | "
                         f"[{r.delta_ci95_low:.6g}, {r.delta_ci95_high:.6g}] |")
    pending = sum(r.get("ann_provenance_status") == "ann_provenance_pending" for r in complete)
    lines += ["", "## Coverage and limitations", "",
              f"- Completed pairs with ANN provenance pending: {pending}. Pending comparisons are provisional, not fully verified reproduction.",
              "- Missing, failed, or mismatched runs remain listed in comparison_status.json.",
              "- Modes are summarized separately; repeated seeds have overlapping source data.",
              "- Five-seed t intervals are descriptive and rely on assumptions; they do not establish cross-library generalization.",
              "- Formal two-sided primary p-values and Holm correction are emitted only when both primary mode groups are complete and provenance-matched.",
              "- Both modes require five pairs with positive finite paired variance; degenerate or incomplete families remain descriptive only.",
              "- See paired_metrics.csv for each seed and summary_by_mode.csv for relative gains, SD, confidence intervals and eligible p-values. Relative-error deltas use percentage points.",
              "- Review P90/P99 alongside mean error; a lower MAE alone does not establish a tail-error benefit.",
              "- Checkpoint metadata and committed source hashes cannot independently attest absence of uncommitted ANN training edits.",
              f"- Figures updated by this aggregation: {make_plots}; if false, existing images may be stale.",
              "- Hardware differs; runtime is not a primary comparison or a model speedup claim.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    if make_plots:
        plot_comparison(output, records, config["modes"])


def plot_comparison(output: Path, records: pd.DataFrame, modes: list[str]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for filename, metrics in (("macro_mae.png", ["MAE"]),
                              ("macro_tail_error.png", ["P90RE_pct", "P99RE_pct"])):
        fig, axes = plt.subplots(len(metrics), len(modes), squeeze=False,
                                 figsize=(5 * len(modes), 3.6 * len(metrics)))
        for i, metric in enumerate(metrics):
            for j, mode in enumerate(modes):
                ax = axes[i, j]
                selected = records[(records.scope == "macro") & (records.metric == metric)
                                   & (records["mode"] == mode)].sort_values("seed")
                for row in selected.itertuples():
                    ax.plot([0, 1], [row.ann, row.ridge], marker="o", alpha=.7,
                            label=f"seed {row.seed}")
                ax.set_xticks([0, 1], ["ANN", "Ridge"])
                ax.set_ylabel("MAE (ps)" if metric == "MAE" else metric + " (%)")
                ax.set_title(mode + " / per-cell macro")
                ax.grid(axis="y", alpha=.2)
                if len(selected):
                    ax.legend(fontsize="small")
        fig.suptitle("Paired observations; check report for provenance and coverage")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
