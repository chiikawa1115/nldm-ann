"""Phase 3：Per-(cell, arc, table) MLP 訓練，含 grid-holdout。

實驗 A 的核心訓練腳本。對每個獨立 LUT 表（3,540 張）訓一個小 MLP，
用 log-scale 輸入 + per-table 標準化，Adam + early stopping。

CLI 用法（不同 holdout 比例）：
    conda activate nldm
    cd C:\\Users\\ruru8\\claude_project\\nldm_ann

    # 20% holdout（保留 80% 用來訓，很多點）
    python src/train.py --parquet data/processed/nangate45.parquet `
        --holdout 0.2 --out checkpoints/nangate45/ratio0.2.pt

    # 70% holdout（只保留 30% —— 論文重點：ANN 能否在少量點下贏 LUT）
    python src/train.py --parquet data/processed/nangate45.parquet `
        --holdout 0.7 --out checkpoints/nangate45/ratio0.7.pt

輸出 checkpoint 內容：
    {
      "config": {holdout, seed, hidden_dims, ...},
      "arcs": [                       # 順序固定
          (cell, output_pin, related_pin, when, table_type),
          ...
      ],
      "checkpoints": {                # arc_key (tuple) → 訓練結果
          arc_key: {
              "state_dict": ...,
              "stats": {x_mean, x_std, y_mean, y_std},
              "n_train": int,
              "train_loss": float,
          }
      }
    }
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from models.mlp_nldm import MLPNldm

# 對於 3,540 個微型 MLP，PyTorch 預設多執行緒反而拖慢（同步 overhead 大於計算）。
# 鎖到 2 執行緒，讓系統其他行程有 CPU 可用，總體時間反而縮短。
torch.set_num_threads(2)


ARC_COLS = ["cell", "output_pin", "related_pin", "when", "table_type"]


# ============================================================
# 1. Grid holdout 切分
# ============================================================
def split_by_grid_holdout(
    df: pd.DataFrame, holdout_ratio: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """對每張 LUT 表獨立隨機遮蓋 `holdout_ratio` 比例的格點。

    這是實驗 A 的核心：ANN 與 LUT 都只能用「非 holdout 的格點」訓練/建表，
    然後在 holdout 格點上比誰更準——回答「用更少 SPICE 點能達到多少精度」。

    每張表獨立切分（不是全域切分）以保證：
        - 每張表都有 train 與 test
        - 各表 holdout 比例一致
    """
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(df), dtype=bool)

    for _, g in df.groupby(ARC_COLS, sort=False):
        n = len(g)
        n_test = int(round(n * holdout_ratio))
        if n_test == 0 or n_test == n:
            # 太極端則不切（否則會沒 train 或沒 test）
            continue
        idx = g.index.to_numpy(copy=True)
        rng.shuffle(idx)
        is_test[idx[:n_test]] = True

    train_df = df[~is_test].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)
    return train_df, test_df


# ============================================================
# 2. 單張表的訓練
# ============================================================
def train_one_table(
    df_arc: pd.DataFrame,
    hidden_dims: tuple[int, ...] = (64, 64),
    n_epochs: int = 400,
    patience: int = 40,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
) -> tuple[MLPNldm, dict, float]:
    """對單張 LUT 表訓練一個小 MLP。

    因為每張表只有 ~30-49 個格點，用 full-batch（一整批一起前向）
    而不用 DataLoader —— 大幅省下 Python 層 overhead。

    Args:
        df_arc: 單張表的 DataFrame（同一個 cell, arc, table_type）

    Returns:
        model, stats(用於反正規化), final_train_loss
    """
    X = df_arc[["log_slew", "log_load"]].to_numpy(dtype=np.float32)
    y = df_arc["value"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # per-table 標準化：每張表的 delay 範圍差很大，共用會破壞收斂
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    y_mean = float(y.mean())
    y_std = float(y.std()) + 1e-8

    Xn = ((X - x_mean) / x_std).astype(np.float32)
    yn = ((y - y_mean) / y_std).astype(np.float32)

    X_t = torch.from_numpy(Xn)
    y_t = torch.from_numpy(yn)

    model = MLPNldm(
        in_features=2, hidden_dims=hidden_dims, out_features=1, dropout=0.0
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    model.train()
    for epoch in range(n_epochs):
        optim.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optim.step()

        cur = loss.item()
        if cur < best_loss - 1e-6:
            best_loss = cur
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    stats = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": np.float32(y_mean),
        "y_std": np.float32(y_std),
    }
    return model, stats, best_loss


# ============================================================
# 3. 訓所有表
# ============================================================
def train_all_tables(
    train_df: pd.DataFrame,
    hidden_dims: tuple[int, ...],
    n_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    min_points: int = 6,
    limit: int | None = None,
) -> tuple[list[tuple], dict[tuple, dict]]:
    """對所有 arc 訓練 per-table MLP。

    Args:
        min_points: 少於此點數的 arc 略過（沒訓練意義）
        limit: 若非 None，只訓前 N 個 arc（開發時用）

    Returns:
        arcs 順序 list, {arc_key: {state_dict, stats, n_train, train_loss}}
    """
    groups = list(train_df.groupby(ARC_COLS, sort=False))
    if limit is not None:
        groups = groups[:limit]

    arcs: list[tuple] = []
    ckpts: dict[tuple, dict] = {}
    skipped = 0

    for keys, g in tqdm(groups, desc="training", ncols=80):
        if len(g) < min_points:
            skipped += 1
            continue
        model, stats, final_loss = train_one_table(
            g,
            hidden_dims=hidden_dims,
            n_epochs=n_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
        )
        arc_key = keys  # pandas groupby 回傳 tuple
        arcs.append(arc_key)
        ckpts[arc_key] = {
            "state_dict": model.state_dict(),
            "stats": stats,
            "n_train": len(g),
            "train_loss": final_loss,
        }

    print(f"訓練 {len(ckpts):,} 張表；略過 {skipped} 張（樣本 < {min_points}）")
    return arcs, ckpts


# ============================================================
# 4. 主流程 + CLI
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Per-table MLP training with grid holdout.")
    p.add_argument("--parquet", required=True, help="輸入 parquet 路徑")
    p.add_argument("--holdout", type=float, required=True, help="holdout 比例（0-1）")
    p.add_argument("--seed", type=int, default=42, help="切分隨機種子")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64],
                   help="MLP 隱藏層大小，例：--hidden-dims 64 64")
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--limit", type=int, default=None, help="只訓前 N 張表（開發用）")
    p.add_argument("--out", required=True, help="輸出 .pt 檔路徑")
    args = p.parse_args()

    print("=" * 60)
    print(" Per-table MLP 訓練")
    print("=" * 60)
    print(f"parquet     : {args.parquet}")
    print(f"holdout     : {args.holdout*100:.0f}%")
    print(f"seed        : {args.seed}")
    print(f"hidden_dims : {args.hidden_dims}")
    print(f"n_epochs    : {args.n_epochs} (early stop, patience={args.patience})")
    print(f"lr          : {args.lr}, weight_decay={args.weight_decay}")

    df = pd.read_parquet(args.parquet)
    print(f"\n總樣本數    : {len(df):,}")

    train_df, test_df = split_by_grid_holdout(df, args.holdout, seed=args.seed)
    print(f"訓練集      : {len(train_df):,}")
    print(f"測試集      : {len(test_df):,}")

    t0 = time.time()
    arcs, ckpts = train_all_tables(
        train_df,
        hidden_dims=tuple(args.hidden_dims),
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        limit=args.limit,
    )
    elapsed = time.time() - t0
    print(f"\n訓練耗時    : {elapsed:.1f} 秒")

    # 存 checkpoint
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "parquet": args.parquet,
            "holdout": args.holdout,
            "seed": args.seed,
            "hidden_dims": list(args.hidden_dims),
            "n_epochs": args.n_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "arcs": arcs,
        "checkpoints": ckpts,
    }, out_path)
    print(f"\n已存 {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
