"""用真實 .lib 資料訓練 ANN 並對比傳統雙線性 LUT 插值。

這是論文的第一個實驗腳本——回答核心問題：
    「用 ANN 逼近 NLDM 的精度，能否勝過傳統雙線性插值？」

實驗設計：
    1. 資料：ASAP7 INVBUF，三個 process corner (FF/SS/TT)，只用 cell_rise
       總樣本數：37 cells × 3 corners × 49 grid = 5,439
    2. 切分：每個 (cell, corner) 群組內 80/20 隨機切分（分層抽樣）
    3. MLP baseline：一個模型跨所有 cell（用 cell embedding）
    4. LUT baseline：對每個 (cell, corner) 用 scipy 的 LinearNDInterpolator
       在訓練點上做內插，預測測試點
    5. 對比：MAE、MaxE、相對誤差、R²

執行：
    conda activate nldm
    cd C:\\Users\\ruru8\\claude_project\\nldm_ann
    python src/train_real.py

輸出：
    - 訓練 loss 曲線（列印）
    - checkpoints/mlp_multicell_asap7.pt
    - 比較表格（MLP vs LUT）
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_nldm import MLPNldmMultiCell


# ============================================================
# 設定（改這裡）
# ============================================================
PARQUET_PATH = Path("data/processed/asap7_invbuf.parquet")
TABLE_TYPE = "cell_rise"        # 一次先訓一種表格
CORNERS = ["TT"]                # 先從單一 corner 開始（跨 corner 需另加 embedding）
TEST_RATIO = 0.20
BATCH_SIZE = 64
N_EPOCHS = 200
LEARNING_RATE = 3e-3
EMBEDDING_DIM = 16
HIDDEN_DIMS = (128, 128, 128)
LOG_TRANSFORM_Y = True          # 對 delay 取 log —— 處理動態範圍極大的問題
SEED = 42

CKPT_PATH = Path("checkpoints/mlp_multicell_asap7.pt")


# ============================================================
# Step 1: 載入 + 過濾 + 切分
# ============================================================
def load_and_split(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """讀 parquet，過濾到 cell_rise，做分層 80/20 切分。

    分層切分：在每個 (cell, corner) 群組**內**隨機切。這樣訓練與測試集
    都會覆蓋到所有 cell，避免「某個 cell 完全沒出現在訓練集」的問題。
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["table_type"] == TABLE_TYPE]
    df = df[df["corner"].isin(CORNERS)].reset_index(drop=True)
    print(f"過濾後總樣本數 : {len(df):,}")
    print(f"cell 數量      : {df['cell'].nunique()}")
    print(f"corner 分佈    : {dict(df['corner'].value_counts())}")

    rng = np.random.default_rng(SEED)
    is_test = np.zeros(len(df), dtype=bool)
    for _, group in df.groupby(["cell", "corner"], sort=False):
        n = len(group)
        n_test = int(round(n * TEST_RATIO))
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        is_test[idx[:n_test]] = True

    train_df = df[~is_test].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)
    print(f"訓練集 {len(train_df):,} / 測試集 {len(test_df):,}")
    return train_df, test_df


# ============================================================
# Step 2: 轉成 tensor + DataLoader
# ============================================================
@dataclass
class NormStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def build_loaders(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[DataLoader, DataLoader, dict[str, int], NormStats]:
    """把 DataFrame 包成 PyTorch DataLoader（含正規化與 cell 索引）。

    重要：用「訓練集」的統計量做正規化。測試集共用同一組 stats，
    避免 data leakage。
    """
    # 建立 cell → int 映射（用訓練集裡出現的 cell）
    cells_sorted = sorted(train_df["cell"].unique())
    cell_vocab = {c: i for i, c in enumerate(cells_sorted)}

    def to_arrays(df):
        X = df[["slew", "load"]].to_numpy(dtype=np.float32)
        y_raw = df["value"].to_numpy(dtype=np.float32).reshape(-1, 1)
        # 對 y 取 log 讓不同尺度的 delay 值都好學（例如 5ps 與 500ps）
        y = np.log(y_raw + 1e-3).astype(np.float32) if LOG_TRANSFORM_Y else y_raw
        cid = np.array([cell_vocab[c] for c in df["cell"]], dtype=np.int64)
        return X, y, cid

    X_tr, y_tr, cid_tr = to_arrays(train_df)
    X_te, y_te, cid_te = to_arrays(test_df)

    stats = NormStats(
        x_mean=X_tr.mean(axis=0),
        x_std=X_tr.std(axis=0) + 1e-8,
        y_mean=float(y_tr.mean()),
        y_std=float(y_tr.std()) + 1e-8,
    )

    Xn_tr = ((X_tr - stats.x_mean) / stats.x_std).astype(np.float32)
    Xn_te = ((X_te - stats.x_mean) / stats.x_std).astype(np.float32)
    yn_tr = ((y_tr - stats.y_mean) / stats.y_std).astype(np.float32)
    yn_te = ((y_te - stats.y_mean) / stats.y_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(Xn_tr), torch.from_numpy(cid_tr), torch.from_numpy(yn_tr)
    )
    test_ds = TensorDataset(
        torch.from_numpy(Xn_te), torch.from_numpy(cid_te), torch.from_numpy(yn_te)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, cell_vocab, stats


# ============================================================
# Step 3: 訓練迴圈
# ============================================================
def train_mlp(
    model: MLPNldmMultiCell,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
) -> None:
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n開始訓練 MLP")
    print(f"參數量         : {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Epoch':>6} {'Train Loss':>14} {'Test Loss':>14} {'Time (s)':>10}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # 訓練
        model.train()
        tot, n = 0.0, 0
        for x, cid, y in train_loader:
            optim.zero_grad()
            pred = model(x, cid)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            tot += loss.item() * len(x)
            n += len(x)
        train_loss = tot / n

        # 評估
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, cid, y in test_loader:
                pred = model(x, cid)
                loss = loss_fn(pred, y)
                tot += loss.item() * len(x)
                n += len(x)
        test_loss = tot / n

        if epoch == 1 or epoch % 5 == 0 or epoch == n_epochs:
            print(
                f"{epoch:>6d} {train_loss:>14.6f} {test_loss:>14.6f} "
                f"{time.time() - t0:>10.2f}"
            )


# ============================================================
# Step 4: LUT baseline —— 傳統做法對照組
# ============================================================
def lut_baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """對每個 (cell, corner) 群組獨立做雙線性 LUT 插值。

    做法：
        - 取這個群組在訓練集裡的所有 (slew, load, value)
        - 用 scipy 的 LinearNDInterpolator（Delaunay 三角化）建插值器
          注意：因為訓練集是「原始 7×7 網格隨機拿掉 20%」，剩下的點可能
          不再是規則網格，所以用 Delaunay 三角化的線性插值；這在論文中
          與雙線性內插等價（當網格完整時）。
        - 若測試點落在訓練點的凸包外，`LinearNDInterpolator` 會回 NaN，
          用最近鄰內插補上。
    """
    preds = np.full(len(test_df), np.nan, dtype=np.float64)
    for (cell, corner), test_g in test_df.groupby(["cell", "corner"], sort=False):
        train_g = train_df[(train_df["cell"] == cell) & (train_df["corner"] == corner)]
        if len(train_g) < 3:
            # 需要至少 3 點才能做 Delaunay
            continue
        pts = train_g[["slew", "load"]].to_numpy()
        vals = train_g["value"].to_numpy()
        try:
            f_lin = LinearNDInterpolator(pts, vals)
            f_near = NearestNDInterpolator(pts, vals)
        except Exception:
            continue

        q = test_g[["slew", "load"]].to_numpy()
        p = f_lin(q)
        mask = np.isnan(p)
        if mask.any():
            p[mask] = f_near(q[mask])
        preds[test_g.index.to_numpy()] = p
    return preds


# ============================================================
# Step 5: 計算對比指標
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    err = y_pred - y_true
    abs_err = np.abs(err)
    rel_err = abs_err / np.abs(y_true).clip(min=1e-9)
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {
        "name": name,
        "MAE": float(abs_err.mean()),
        "MaxE": float(abs_err.max()),
        "MRE_pct": float(rel_err.mean() * 100),
        "MaxRE_pct": float(rel_err.max() * 100),
        "R2": float(r2),
    }


def print_comparison(rows: list[dict]) -> None:
    print("\n" + "=" * 76)
    print(" MLP vs LUT 對比（測試集）")
    print("=" * 76)
    header = f"{'Method':<10} {'MAE':>10} {'MaxE':>10} {'MRE %':>10} {'MaxRE %':>10} {'R²':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<10} {r['MAE']:>10.4f} {r['MaxE']:>10.4f} "
            f"{r['MRE_pct']:>10.3f} {r['MaxRE_pct']:>10.3f} {r['R2']:>10.6f}"
        )
    print()


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print(" 真實 NLDM 資料訓練：MLP vs 傳統 LUT 插值")
    print("=" * 60)
    print(f"資料檔     : {PARQUET_PATH}")
    print(f"table_type : {TABLE_TYPE}")
    print(f"corners    : {CORNERS}")

    # 1. 資料
    train_df, test_df = load_and_split(PARQUET_PATH)

    # 2. DataLoader
    train_loader, test_loader, cell_vocab, stats = build_loaders(train_df, test_df)

    # 3. 訓練 MLP
    model = MLPNldmMultiCell(
        num_cells=len(cell_vocab),
        embedding_dim=EMBEDDING_DIM,
        in_features=2,
        hidden_dims=HIDDEN_DIMS,
        out_features=1,
    )
    train_mlp(model, train_loader, test_loader, N_EPOCHS)

    # 4. MLP 在測試集上預測（把正規化反過來）
    model.eval()
    with torch.no_grad():
        preds_norm = []
        trues_norm = []
        for x, cid, y in test_loader:
            preds_norm.append(model(x, cid).numpy())
            trues_norm.append(y.numpy())
    preds_norm = np.concatenate(preds_norm).flatten()
    trues_norm = np.concatenate(trues_norm).flatten()

    # 反正規化：先從 z-score 復原，再從 log 空間指數回原尺度
    mlp_pred_transformed = preds_norm * stats.y_std + stats.y_mean
    y_true_transformed = trues_norm * stats.y_std + stats.y_mean
    if LOG_TRANSFORM_Y:
        mlp_pred = np.exp(mlp_pred_transformed) - 1e-3
        y_true = np.exp(y_true_transformed) - 1e-3
    else:
        mlp_pred = mlp_pred_transformed
        y_true = y_true_transformed

    # 5. LUT baseline 在同一批測試集上預測
    print("\n計算 LUT 雙線性插值 baseline ...")
    lut_pred = lut_baseline_predict(train_df, test_df)

    # 對齊順序：LUT 是照 test_df 的 index，MLP 是照 DataLoader 順序
    # DataLoader shuffle=False，所以順序 = test_df 順序 → 可以直接比較
    valid = ~np.isnan(lut_pred)
    if not valid.all():
        print(f"  警告：{(~valid).sum()} 筆 LUT 預測為 NaN，已從評估中排除")
    lut_metrics = compute_metrics(y_true[valid], lut_pred[valid], "LUT")
    mlp_metrics = compute_metrics(y_true[valid], mlp_pred[valid], "MLP")

    # 6. 印對比結果
    print_comparison([lut_metrics, mlp_metrics])

    # 額外：算 MLP 相對 LUT 的改善幅度
    improve_mae = (lut_metrics["MAE"] - mlp_metrics["MAE"]) / lut_metrics["MAE"] * 100
    improve_maxe = (lut_metrics["MaxE"] - mlp_metrics["MaxE"]) / lut_metrics["MaxE"] * 100
    print(f"MLP 相對 LUT 的改善：")
    print(f"  MAE 降低   : {improve_mae:+.1f}%   (正值代表 MLP 較準)")
    print(f"  MaxE 降低  : {improve_maxe:+.1f}%")

    # 7. 存 checkpoint
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "cell_vocab": cell_vocab,
        "stats": {
            "x_mean": stats.x_mean, "x_std": stats.x_std,
            "y_mean": stats.y_mean, "y_std": stats.y_std,
        },
        "config": {
            "num_cells": len(cell_vocab),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dims": HIDDEN_DIMS,
        },
        "test_metrics": {"lut": lut_metrics, "mlp": mlp_metrics},
    }, CKPT_PATH)
    print(f"\n模型與結果已存至 {CKPT_PATH}")


if __name__ == "__main__":
    main()
