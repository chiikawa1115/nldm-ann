"""ANN + NLDM 教學型完整範例（合成資料）。

--------------------------------------------------------------------
這支腳本的目的：讓你在拿到真的 .lib 資料前，先看懂「用 ANN 逼近 NLDM
延遲函數」的完整流程。用假資料，但流程與真資料完全一樣。

流程共 5 步：
  1. 產生合成資料 —— 用一個「假 NLDM 公式」當真理，加點雜訊當觀測
  2. 切訓練/測試集
  3. 建立 MLP 模型
  4. 訓練迴圈（forward → loss → backward → update）
  5. 評估：對比 ANN 預測 vs 真理，並輸出 R² 與最大誤差

執行方式：
    conda activate nldm
    cd C:\\Users\\ruru8\\claude_project\\nldm_ann
    python src/tutorial_train.py
--------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 匯入我們自己寫的 MLP 模型
from models.mlp_nldm import MLPNldm


# ============================================================
# Step 1: 產生合成資料（模擬「SPICE 特徵化」的結果）
# ============================================================
def make_synthetic_nldm_data(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """製造假的 (slew, load) → delay 資料。

    真實 cell 的延遲大致長這樣：
        delay ≈ a + b*slew + c*load + d*slew*load + e*sqrt(load) + noise

    這個公式是我們的「上帝真理」。ANN 的任務是從一堆帶雜訊的觀測
    中學回這個函數。
    """
    rng = np.random.default_rng(seed)

    # 輸入範圍：模擬真實 NLDM 特徵化的 slew 與 load 掃描區間
    slew = rng.uniform(0.01, 0.5, size=n)      # ns
    load = rng.uniform(0.001, 0.05, size=n)    # pF

    # 假 NLDM 公式（真理）
    delay_true = (
        5.0
        + 8.0 * slew
        + 200.0 * load
        + 300.0 * slew * load
        + 30.0 * np.sqrt(load)
    )  # ps

    # 加上量測雜訊（1% Gaussian）
    delay_obs = delay_true * (1 + rng.normal(0, 0.01, size=n))

    X = np.stack([slew, load], axis=1).astype(np.float32)     # shape: (n, 2)
    y = delay_obs.astype(np.float32).reshape(-1, 1)            # shape: (n, 1)
    return X, y


# ============================================================
# Step 2: 資料前處理（正規化 + 切訓練/測試）
# ============================================================
def prepare_dataloaders(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, batch_size: int = 64
) -> tuple[DataLoader, DataLoader, dict]:
    """把 numpy 資料切分並包成 PyTorch DataLoader。

    正規化很重要 —— ANN 對數值尺度敏感，slew (0.01~0.5) 與 load (0.001~0.05)
    量級不同，若不正規化，收斂會慢很多。
    """
    n = len(X)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    # 用「訓練集」的統計量計算正規化參數（避免資料洩漏）
    x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0)
    y_mean, y_std = y[train_idx].mean(), y[train_idx].std()

    def norm_x(a):
        return (a - x_mean) / x_std

    def norm_y(a):
        return (a - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(norm_x(X[train_idx])),
        torch.from_numpy(norm_y(y[train_idx])),
    )
    test_ds = TensorDataset(
        torch.from_numpy(norm_x(X[test_idx])),
        torch.from_numpy(norm_y(y[test_idx])),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return train_loader, test_loader, stats


# ============================================================
# Step 3 & 4: 建模型 + 訓練迴圈
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """跑一個 epoch 的訓練。回傳這個 epoch 的平均 loss。"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()             # 清掉上一步累積的梯度
        pred = model(x_batch)             # 前向傳遞
        loss = loss_fn(pred, y_batch)     # 計算損失
        loss.backward()                   # 反向傳遞：算梯度
        optimizer.step()                  # 更新權重
        total_loss += loss.item() * len(x_batch)
        total_samples += len(x_batch)
    return total_loss / total_samples


@torch.no_grad()  # 評估時關掉梯度追蹤，加速
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        total_loss += loss.item() * len(x_batch)
        total_samples += len(x_batch)
    return total_loss / total_samples


# ============================================================
# Step 5: 完整流程
# ============================================================
def main() -> None:
    print("=" * 60)
    print(" NLDM + ANN 教學範例（合成資料）")
    print("=" * 60)

    # --- 1. 資料 ---
    X, y = make_synthetic_nldm_data(n=2000, seed=42)
    print(f"\n[Step 1] 產生 {len(X)} 筆合成資料")
    print(f"         輸入 shape: {X.shape}  (slew, load)")
    print(f"         輸出 shape: {y.shape}  (delay ps)")
    print(f"         delay 範圍: {y.min():.1f} ~ {y.max():.1f} ps")

    # --- 2. DataLoader ---
    train_loader, test_loader, stats = prepare_dataloaders(X, y, batch_size=64)
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"\n[Step 2] 訓練集 {n_train} 筆 / 測試集 {n_test} 筆")

    # --- 3. 模型 ---
    # 輸入 2 維 (slew, load)，輸出 1 維 (delay)
    # 隱藏層 (64, 64, 64) 是常見的中小型 MLP 設計
    model = MLPNldm(in_features=2, hidden_dims=(64, 64, 64), out_features=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Step 3] 建立 MLP 模型")
    print(f"         架構: 2 → 64 → 64 → 64 → 1")
    print(f"         參數量: {n_params:,}")

    # --- 4. 訓練 ---
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n[Step 4] 開始訓練（Adam, lr=1e-3, MSE loss）")
    print(f"         {'Epoch':>6} {'Train Loss':>14} {'Test Loss':>14}")
    n_epochs = 30
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        test_loss = evaluate(model, test_loader, loss_fn)
        if epoch == 1 or epoch % 5 == 0 or epoch == n_epochs:
            print(f"         {epoch:>6d} {train_loss:>14.6f} {test_loss:>14.6f}")

    # --- 5. 評估：把預測值反正規化回原尺度，計算實際誤差 ---
    print(f"\n[Step 5] 評估模型精度")
    model.eval()
    with torch.no_grad():
        all_pred_norm = []
        all_true_norm = []
        for x_batch, y_batch in test_loader:
            all_pred_norm.append(model(x_batch).numpy())
            all_true_norm.append(y_batch.numpy())
    pred_norm = np.concatenate(all_pred_norm)
    true_norm = np.concatenate(all_true_norm)

    # 反正規化
    pred = pred_norm * stats["y_std"] + stats["y_mean"]
    true = true_norm * stats["y_std"] + stats["y_mean"]

    abs_err = np.abs(pred - true)
    rel_err = abs_err / true

    # R² 決定係數
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"         測試集大小     : {len(true)}")
    print(f"         平均絕對誤差   : {abs_err.mean():.3f} ps")
    print(f"         最大絕對誤差   : {abs_err.max():.3f} ps")
    print(f"         平均相對誤差   : {rel_err.mean() * 100:.2f} %")
    print(f"         最大相對誤差   : {rel_err.max() * 100:.2f} %")
    print(f"         R² 決定係數    : {r2:.6f}   (越接近 1 越好)")

    # --- 6. 存權重（示範）---
    ckpt_path = "checkpoints/tutorial_mlp.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "stats": stats,
        "config": {"in_features": 2, "hidden_dims": (64, 64, 64), "out_features": 1},
    }, ckpt_path)
    print(f"\n[Step 6] 模型權重已儲存至 {ckpt_path}")
    print("\n完成！這就是 ANN 訓練 NLDM 的完整流程。")


if __name__ == "__main__":
    main()
