"""環境自我檢測腳本。

用途：確認 PyTorch、liberty-parser 等核心套件正確安裝，並顯示是否偵測到 GPU。
執行：
    conda activate nldm
    python src/sanity_check.py
"""

from __future__ import annotations

import platform
import sys


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_python() -> None:
    _print_header("Python 環境")
    print(f"Python 版本      : {sys.version.split()[0]}")
    print(f"作業系統         : {platform.system()} {platform.release()}")
    print(f"處理器架構       : {platform.machine()}")


def check_pytorch() -> None:
    _print_header("PyTorch")
    import torch

    print(f"torch 版本       : {torch.__version__}")
    print(f"CUDA 可用        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本        : {torch.version.cuda}")
        print(f"GPU 數量         : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("(目前使用 CPU — 這在筆電開發階段是預期的)")


def check_mini_mlp() -> None:
    _print_header("Mini MLP 前向/反向傳遞測試")
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)

    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()

    print(f"輸入 shape       : {tuple(x.shape)}")
    print(f"輸出 shape       : {tuple(pred.shape)}")
    print(f"初始 loss        : {loss.item():.4f}")
    print("反向傳遞         : OK")


def check_liberty_parser() -> None:
    _print_header("Liberty Parser")
    try:
        import liberty  # liberty-parser 套件的 import 名稱

        print(f"liberty-parser   : 已載入 ({liberty.__name__})")
    except ImportError as e:
        print(f"liberty-parser   : 未安裝 ({e})")


def check_scientific_stack() -> None:
    _print_header("科學運算套件")
    import numpy as np
    import pandas as pd
    import matplotlib
    import sklearn

    print(f"numpy            : {np.__version__}")
    print(f"pandas           : {pd.__version__}")
    print(f"matplotlib       : {matplotlib.__version__}")
    print(f"scikit-learn     : {sklearn.__version__}")


def main() -> int:
    check_python()
    check_pytorch()
    check_mini_mlp()
    check_liberty_parser()
    check_scientific_stack()
    _print_header("結果")
    print("所有檢查完成。若上方沒有 ImportError，環境即可用於後續開發。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
