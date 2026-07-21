"""Phase 4：實驗 A 精度評估。

回答論文核心問題：
    「在 grid holdout 上，ANN 與傳統雙線性插值誰更準？」

流程：
    1. 讀 parquet + train.py 產出的 checkpoint
    2. 用 checkpoint 中的 seed + holdout 重現 train/test 切分（重要：不能重新亂數）
    3. 對每張表的 holdout 格點：
       - 用 MLP 預測（載 per-table state_dict）
       - 用 scipy LinearNDInterpolator 對訓練點做插值
    4. 計算 MAPE、Max Error、MAE、R²
    5. 分不同 table_type、cell、arc 統計
    6. 存 experiments/expA/ratio{X}.parquet 與 .json 摘要

CLI 用法：
    python src/evaluate.py --parquet data/processed/nangate45.parquet `
        --ckpt checkpoints/nangate45/ratio0.3.pt `
        --out experiments/expA/ratio0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from models.mlp_nldm import MLPNldm
from train import ARC_COLS, split_by_grid_holdout


# ============================================================
# 1. 載入 checkpoint + 重現切分
# ============================================================
def reconstruct_split(
    df: pd.DataFrame, ckpt: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用 checkpoint 中的 seed 與 holdout 比例重現訓練切分。"""
    cfg = ckpt["config"]
    return split_by_grid_holdout(df, cfg["holdout"], seed=cfg["seed"])


# ============================================================
# 2. MLP 預測
# ============================================================
def mlp_predict_arc(
    arc_key: tuple,
    test_arc: pd.DataFrame,
    ckpt_entry: dict,
    hidden_dims: tuple[int, ...],
) -> np.ndarray:
    """對單一 arc 的 holdout 點做 MLP 預測（含反正規化）。"""
    model = MLPNldm(in_features=2, hidden_dims=hidden_dims, out_features=1)
    model.load_state_dict(ckpt_entry["state_dict"])
    model.eval()

    stats = ckpt_entry["stats"]
    X = test_arc[["log_slew", "log_load"]].to_numpy(dtype=np.float32)
    Xn = ((X - stats["x_mean"]) / stats["x_std"]).astype(np.float32)

    with torch.no_grad():
        yn = model(torch.from_numpy(Xn)).numpy().flatten()
    return yn * stats["y_std"] + stats["y_mean"]


# ============================================================
# 3. LUT 雙線性 baseline 預測
# ============================================================
def lut_predict_arc(train_arc: pd.DataFrame, test_arc: pd.DataFrame) -> np.ndarray:
    """對單一 arc 的 holdout 點做傳統雙線性插值預測。

    重要：LUT baseline 只能用「train 點」建插值器，不能偷看 holdout 點。
    這才是實驗 A 的公平比較。
    """
    if len(train_arc) < 3:
        return np.full(len(test_arc), np.nan)

    pts = train_arc[["slew", "load"]].to_numpy()
    vals = train_arc["value"].to_numpy()
    try:
        f_lin = LinearNDInterpolator(pts, vals)
        f_near = NearestNDInterpolator(pts, vals)
    except Exception:
        return np.full(len(test_arc), np.nan)

    q = test_arc[["slew", "load"]].to_numpy()
    p = f_lin(q)
    mask = np.isnan(p)
    if mask.any():
        p[mask] = f_near(q[mask])
    return p


# ============================================================
# 4. 指標
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """計算 MAE / MaxE / MAPE / MaxRE / R²。

    MAPE 用 |y_true| 為分母時，若 y_true 極小會爆炸；此處加 epsilon。
    """
    err = y_pred - y_true
    abs_err = np.abs(err)
    denom = np.maximum(np.abs(y_true), 1e-6)
    rel_err = abs_err / denom
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1e-12
    return {
        "n": int(len(y_true)),
        "MAE": float(abs_err.mean()),
        "MaxE": float(abs_err.max()),
        "MAPE_pct": float(rel_err.mean() * 100),
        "P50RE_pct": float(np.median(rel_err) * 100),
        "P90RE_pct": float(np.percentile(rel_err, 90) * 100),
        "P99RE_pct": float(np.percentile(rel_err, 99) * 100),
        "MaxRE_pct": float(rel_err.max() * 100),
        "R2": float(1 - ss_res / ss_tot),
    }


# ============================================================
# 5. 主流程
# ============================================================
def evaluate(parquet_path: Path, ckpt_path: Path, out_prefix: Path) -> dict:
    print("=" * 60)
    print(" 實驗 A：MLP vs LUT 精度比對")
    print("=" * 60)
    print(f"parquet    : {parquet_path}")
    print(f"checkpoint : {ckpt_path}")

    df = pd.read_parquet(parquet_path)
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["config"]
    hidden_dims = tuple(cfg["hidden_dims"])
    print(f"holdout    : {cfg['holdout']*100:.0f}%")
    print(f"trained    : {len(ckpt['checkpoints']):,} tables")

    train_df, test_df = reconstruct_split(df, ckpt)
    print(f"train/test : {len(train_df):,} / {len(test_df):,}")

    # 分組
    train_groups = {k: g for k, g in train_df.groupby(ARC_COLS, sort=False)}
    test_groups = {k: g for k, g in test_df.groupby(ARC_COLS, sort=False)}

    per_row_rows: list[dict] = []
    n_mlp_missing = 0
    n_lut_nan = 0

    for arc_key in ckpt["arcs"]:
        if arc_key not in test_groups:
            continue
        test_arc = test_groups[arc_key]
        train_arc = train_groups.get(arc_key)
        if train_arc is None:
            continue

        # MLP 預測
        try:
            mlp_pred = mlp_predict_arc(
                arc_key, test_arc, ckpt["checkpoints"][arc_key], hidden_dims
            )
        except Exception:
            n_mlp_missing += len(test_arc)
            continue

        # LUT 預測
        lut_pred = lut_predict_arc(train_arc, test_arc)
        lut_ok = ~np.isnan(lut_pred)
        n_lut_nan += int((~lut_ok).sum())

        y_true = test_arc["value"].to_numpy(dtype=np.float64)
        # 逐點記錄以便進一步分析
        for i, (idx, row) in enumerate(test_arc.iterrows()):
            per_row_rows.append({
                "cell": row["cell"],
                "output_pin": row["output_pin"],
                "related_pin": row["related_pin"],
                "when": row["when"],
                "table_type": row["table_type"],
                "slew": row["slew"],
                "load": row["load"],
                "y_true": float(y_true[i]),
                "mlp_pred": float(mlp_pred[i]),
                "lut_pred": float(lut_pred[i]) if lut_ok[i] else float("nan"),
            })

    result_df = pd.DataFrame(per_row_rows)
    valid = ~result_df["lut_pred"].isna()
    print(f"\n有效比對點：{valid.sum():,} / {len(result_df):,}")
    if n_mlp_missing:
        print(f"MLP 缺失   : {n_mlp_missing}")
    if n_lut_nan:
        print(f"LUT NaN    : {n_lut_nan}  (已從評估中排除)")

    r = result_df[valid]
    y_true = r["y_true"].to_numpy()
    mlp_metrics = compute_metrics(y_true, r["mlp_pred"].to_numpy())
    lut_metrics = compute_metrics(y_true, r["lut_pred"].to_numpy())

    # 分 table_type 統計
    per_table_type: dict[str, dict] = {}
    for tt, g in r.groupby("table_type"):
        yt = g["y_true"].to_numpy()
        per_table_type[tt] = {
            "n": len(g),
            "MLP": compute_metrics(yt, g["mlp_pred"].to_numpy()),
            "LUT": compute_metrics(yt, g["lut_pred"].to_numpy()),
        }

    summary = {
        "holdout_ratio": cfg["holdout"],
        "n_train": int(len(train_df)),
        "n_test_evaluated": int(valid.sum()),
        "overall": {"MLP": mlp_metrics, "LUT": lut_metrics},
        "per_table_type": per_table_type,
    }

    # 印摘要
    _print_summary(summary)

    # 存檔（直接拼接副檔名，避免 with_suffix 把 "ratio0.3" 的 ".3" 誤認為 ext）
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    parquet_out = out_prefix.with_name(out_prefix.name + ".parquet")
    json_out = out_prefix.with_name(out_prefix.name + ".json")
    result_df.to_parquet(parquet_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n每筆結果 : {parquet_out}")
    print(f"摘要 JSON: {json_out}")
    return summary


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 76)
    print(f" 實驗 A 結果  (holdout = {s['holdout_ratio']*100:.0f}%)")
    print("=" * 76)

    def _row(name: str, m: dict) -> str:
        return (
            f"{name:<6} {m['MAE']:>10.4f} {m['MaxE']:>10.4f} "
            f"{m['MAPE_pct']:>9.3f} {m['P90RE_pct']:>10.3f} "
            f"{m['MaxRE_pct']:>10.3f} {m['R2']:>10.6f}"
        )

    print(f"{'Method':<6} {'MAE':>10} {'MaxE':>10} {'MAPE%':>9} "
          f"{'P90RE%':>10} {'MaxRE%':>10} {'R²':>10}")
    print("-" * 76)
    print(_row("LUT", s["overall"]["LUT"]))
    print(_row("MLP", s["overall"]["MLP"]))
    print()
    print("按 table_type 分開：")
    for tt, g in s["per_table_type"].items():
        print(f"  [{tt}]  n={g['n']:,}")
        print("    " + _row("LUT", g["LUT"]))
        print("    " + _row("MLP", g["MLP"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="輸出前綴（會加 .parquet / .json）")
    args = p.parse_args()
    evaluate(Path(args.parquet), Path(args.ckpt), Path(args.out))


if __name__ == "__main__":
    main()
