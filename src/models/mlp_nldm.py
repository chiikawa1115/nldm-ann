"""MLP-based NLDM baseline model.

輸入：input slew, output load (以及選配的 PVT 參數：Vdd, temperature, process corner)
輸出：cell delay, output transition

這是取代傳統 7x7 LUT 的最簡單基準。之後可擴充：
  - 多輸出頭（同時預測 rise/fall delay、rise/fall transition）
  - 使用顯式 cell 結構與連續 PVT 特徵進行 inductive generalization
  - 加入 physics-informed loss（例如強制單調性）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPNldm(nn.Module):
    """簡單的多層感知器 (MLP) 用於延遲預測。

    Args:
        in_features: 輸入特徵數。預設 2 對應 (input_slew, output_load)。
                     若加入 PVT 參數，可提升到 5 或更多。
        hidden_dims: 各隱藏層的神經元數量。
        out_features: 輸出維度。預設 2 對應 (delay, output_transition)。
        dropout: dropout 機率，避免過擬合。
        activation: 激活函數（預設 ReLU；也可用 GELU、SiLU 觀察差異）。
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_dims: tuple[int, ...] = (64, 64, 64),
        out_features: int = 2,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPNldmMultiCell(nn.Module):
    """Transductive baseline using cell IDs; invalid for unseen-cell claims.

    一個模型負責所有 cell —— 每個 cell 的差異由 embedding 學到。
    這比「每個 cell 一個獨立 MLP」更省參數、更容易泛化，也是
    DL-NLDM 論文中常見的做法。

    輸入：
        x        shape (B, 2)      連續特徵 (slew, load)
        cell_id  shape (B,)        每筆屬於哪個 cell 的整數索引

    輸出：
        pred     shape (B, 1)      預測值（delay 或 transition）
    """

    def __init__(
        self,
        num_cells: int,
        embedding_dim: int = 8,
        in_features: int = 2,
        hidden_dims: tuple[int, ...] = (64, 64, 64),
        out_features: int = 1,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        # cell embedding：每個 cell 用一個 8 維向量表示
        # 這個 embedding 是「學出來的」——訓練時會自己找出哪些 cell 相似
        self.cell_embed = nn.Embedding(num_cells, embedding_dim)

        # MLP 的輸入 = embedding + 連續特徵 (slew, load)
        mlp_in = embedding_dim + in_features

        layers: list[nn.Module] = []
        prev = mlp_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cell_id: torch.Tensor) -> torch.Tensor:
        emb = self.cell_embed(cell_id)        # (B, embedding_dim)
        h = torch.cat([emb, x], dim=-1)       # (B, embedding_dim + 2)
        return self.mlp(h)


class FeatureConditionedNldm(nn.Module):
    """Inductive MLP using only inference-available numeric/one-hot features.

    Cell and corner IDs are intentionally absent. Feature normalization must be
    fit on the training partition and persisted with the checkpoint.
    """

    def __init__(
        self, in_features: int, hidden_dims: tuple[int, ...] = (128, 128, 64),
        out_features: int = 1, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mlp = MLPNldm(
            in_features=in_features, hidden_dims=hidden_dims,
            out_features=out_features, dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


if __name__ == "__main__":
    # 快速自我測試
    print("--- MLPNldm ---")
    m1 = MLPNldm(in_features=2, out_features=1)
    x = torch.randn(4, 2)
    print(f"Input : {x.shape}")
    print(f"Output: {m1(x).shape}")
    print(f"Params: {sum(p.numel() for p in m1.parameters()):,}")

    print("\n--- MLPNldmMultiCell ---")
    m2 = MLPNldmMultiCell(num_cells=37, embedding_dim=8)
    x = torch.randn(4, 2)
    cell_id = torch.tensor([0, 1, 5, 36])
    print(f"Input : x={x.shape}, cell_id={cell_id.shape}")
    print(f"Output: {m2(x, cell_id).shape}")
    print(f"Params: {sum(p.numel() for p in m2.parameters()):,}")
