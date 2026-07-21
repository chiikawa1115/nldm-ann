"""實驗 A 收尾：把多個 holdout ratio 的結果畫成論文級圖表。

執行前需先跑完各 ratio 的 evaluate.py，產出 ratio{X}.json。

CLI：
    python experiments/expA/plot_expA.py --results-dir experiments/expA/

輸出：
    - experiments/expA/summary.csv  各 ratio × method 的指標
    - experiments/expA/fig_accuracy_vs_ratio.png  論文主圖
    - experiments/expA/fig_by_table_type.png     按表格類型分開
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS_TO_PLOT = [
    ("MAPE_pct", "MAPE (%)"),
    ("MaxRE_pct", "Max Relative Error (%)"),
    ("MAE", "MAE"),
    ("R2", "R²"),
]


def load_results(results_dir: Path) -> pd.DataFrame:
    """讀取 results_dir/ratio*.json，彙整成長表。"""
    rows = []
    for jf in sorted(results_dir.glob("ratio*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        ratio = data["holdout_ratio"]
        for method in ("MLP", "LUT"):
            m = data["overall"][method]
            rows.append({
                "holdout_ratio": ratio,
                "method": method,
                "scope": "overall",
                **m,
            })
            for tt, g in data["per_table_type"].items():
                rows.append({
                    "holdout_ratio": ratio,
                    "method": method,
                    "scope": tt,
                    **g[method],
                })
    return pd.DataFrame(rows)


def plot_accuracy_vs_ratio(df: pd.DataFrame, out_path: Path) -> None:
    """論文主圖：X = holdout 比例, Y = 誤差指標, 兩條線（MLP vs LUT）。"""
    overall = df[df["scope"] == "overall"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, label) in zip(axes.flatten(), METRICS_TO_PLOT):
        for method, sub in overall.groupby("method"):
            sub = sub.sort_values("holdout_ratio")
            ax.plot(
                sub["holdout_ratio"] * 100, sub[col],
                marker="o", label=method, linewidth=2, markersize=8,
            )
        ax.set_xlabel("Holdout ratio (%)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if col.endswith("_pct") or col == "MAE":
            ax.set_yscale("log")
    fig.suptitle("Experiment A: ANN vs Traditional LUT under varying grid holdout",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"存 {out_path}")
    plt.close(fig)


def plot_by_table_type(df: pd.DataFrame, out_path: Path) -> None:
    """按 table_type 拆開的比較（cell_rise / cell_fall / rise_trans / fall_trans）。"""
    tts = sorted([s for s in df["scope"].unique() if s != "overall"])
    if not tts:
        return
    fig, axes = plt.subplots(1, len(tts), figsize=(4 * len(tts), 4), sharey=True)
    if len(tts) == 1:
        axes = [axes]
    for ax, tt in zip(axes, tts):
        sub = df[df["scope"] == tt].sort_values(["method", "holdout_ratio"])
        for method, g in sub.groupby("method"):
            ax.plot(
                g["holdout_ratio"] * 100, g["MAPE_pct"],
                marker="o", label=method, linewidth=2,
            )
        ax.set_title(tt)
        ax.set_xlabel("Holdout ratio (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("MAPE (%)")
    fig.suptitle("Experiment A: MAPE by table_type", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"存 {out_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="experiments/expA")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    df = load_results(results_dir)
    if df.empty:
        print(f"在 {results_dir} 找不到 ratio*.json，請先跑 evaluate.py")
        return

    csv_path = results_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"存 {csv_path}  ({len(df)} 列)")

    plot_accuracy_vs_ratio(df, results_dir / "fig_accuracy_vs_ratio.png")
    plot_by_table_type(df, results_dir / "fig_by_table_type.png")

    print("\n可用的 holdout ratio：")
    for r in sorted(df["holdout_ratio"].unique()):
        print(f"  {r*100:.0f}%")


if __name__ == "__main__":
    main()
