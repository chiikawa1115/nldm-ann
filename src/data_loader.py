"""從 Liberty (.lib) 檔案抽取 NLDM 訓練資料。

兩層設計：
  1. `extract_lib_to_dataframe` / `extract_lib_to_parquet`
     離線一次性把 .lib 全部 cell 的所有 timing 表攤平成 tidy DataFrame，
     欄位：cell, output_pin, related_pin, table_type, slew, load, value, ...
     存成 parquet 之後訓練時載入極快。

  2. `NldmDataset` (PyTorch Dataset)
     訓練時使用，支援 cell/table_type 過濾、正規化、cell embedding 索引等。

CLI 用法：
    conda activate nldm
    cd C:\\Users\\ruru8\\claude_project\\nldm_ann

    # 抽 ASAP7 三個 corner 為訓練資料
    python src/data_loader.py `
        --lib data/external/asap7/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib `
        --lib data/external/asap7/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib `
        --lib data/external/asap7/asap7sc7p5t_INVBUF_RVT_SS_nldm_220122.lib `
        --out data/processed/asap7_invbuf.parquet

    # 抽 SkyWater
    python src/data_loader.py `
        --lib data/external/sky130hd/sky130_fd_sc_hd__tt_025C_1v80.lib `
        --out data/processed/sky130hd_tt.parquet
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from liberty.parser import parse_liberty
from torch.utils.data import Dataset


# ============================================================
# 1. 抽取器
# ============================================================

# 從檔名推斷 corner。ASAP7 檔名格式：..._RVT_TT_..., SkyWater：...__tt_025C_1v80...
_CORNER_RE = re.compile(r"(?:^|[_\W])(FF|SS|TT|ff|ss|tt)(?:$|[_\W])")


def _infer_corner(name: str) -> str:
    """從 lib 名稱推斷 process corner (FF/SS/TT)，找不到就回傳 'unknown'。"""
    m = _CORNER_RE.search(name)
    return m.group(1).upper() if m else "unknown"


# 我們關心的四種 NLDM 表格。每個 timing arc 都可能有這些子表。
_TABLE_TYPES = ("cell_rise", "cell_fall", "rise_transition", "fall_transition")


@dataclass
class Sample:
    """一筆訓練樣本（LUT 的一個格點）。"""
    lib_file: str
    library_name: str
    corner: str
    cell: str
    output_pin: str
    related_pin: str
    timing_sense: str
    timing_type: str     # combinational / rising_edge / falling_edge / etc.
    when: str            # conditional arc 的條件（例："!B" 表示 B=0 時此 arc 生效）
    table_type: str
    template: str
    slew: float          # index_1 (ps)
    load: float          # index_2 (fF)
    log_slew: float      # ln(slew)  —— 因 LUT index 是幾何級數，log 尺度更均勻
    log_load: float      # ln(load)
    value: float         # delay 或 transition 值


def _clean_str(v) -> str:
    """liberty-parser 有時把字串包在雙引號中；一併去空白。"""
    if v is None:
        return ""
    s = str(v).strip().strip('"')
    return s


def iter_cell_samples(lib_path: Path) -> Iterator[Sample]:
    """走訪一個 .lib 檔的所有 cell / 所有 timing arc / 所有格點，逐筆 yield Sample。"""
    text = lib_path.read_text(encoding="utf-8", errors="replace")
    library = parse_liberty(text)

    library_name = _clean_str(library.args[0]) if library.args else lib_path.stem
    corner = _infer_corner(library_name) or _infer_corner(lib_path.name)

    for cell in library.get_groups("cell"):
        cell_name = _clean_str(cell.args[0])
        for pin in cell.get_groups("pin"):
            direction = _clean_str(pin.get_attribute("direction"))
            if direction != "output":
                continue
            pin_name = _clean_str(pin.args[0])

            for timing in pin.get_groups("timing"):
                related_pin = _clean_str(timing.get_attribute("related_pin"))
                timing_sense = _clean_str(timing.get_attribute("timing_sense"))
                timing_type = _clean_str(timing.get_attribute("timing_type"))
                # when：條件式 arc 的觸發條件，例如 "!B" 表示 B=0 時本 arc 生效
                # 沒有 when 時空字串 —— 這就是「無條件 arc」的識別
                when = _clean_str(timing.get_attribute("when"))

                # 略過非組合 timing（例如 sequential setup/hold arcs）
                # combinational / rising_edge / falling_edge 都會有 cell_rise/cell_fall
                if timing_type and timing_type not in (
                    "", "combinational", "combinational_rise",
                    "combinational_fall", "rising_edge", "falling_edge",
                ):
                    continue

                for table_type in _TABLE_TYPES:
                    tables = list(timing.get_groups(table_type))
                    if not tables:
                        continue
                    tbl = tables[0]  # 只取第一個（通常一個 arc 只有一個）
                    template = _clean_str(tbl.args[0]) if tbl.args else ""

                    try:
                        idx1 = np.asarray(tbl.get_array("index_1")).flatten()
                        idx2 = np.asarray(tbl.get_array("index_2")).flatten()
                        vals = np.asarray(tbl.get_array("values"))
                    except (KeyError, AttributeError, TypeError):
                        continue

                    if vals.ndim != 2 or vals.shape != (len(idx1), len(idx2)):
                        # 有些 scalar 表或 shape 不匹配的，先跳過
                        continue

                    # log-scale：LUT index 通常是幾何級數（每格 ×2），log 後接近等距，
                    # 讓 MLP 較好學。加 1e-9 避免 log(0)。
                    for i, s in enumerate(idx1):
                        log_s = float(np.log(max(float(s), 1e-9)))
                        for j, ell in enumerate(idx2):
                            log_l = float(np.log(max(float(ell), 1e-9)))
                            yield Sample(
                                lib_file=lib_path.name,
                                library_name=library_name,
                                corner=corner,
                                cell=cell_name,
                                output_pin=pin_name,
                                related_pin=related_pin,
                                timing_sense=timing_sense,
                                timing_type=timing_type,
                                when=when,
                                table_type=table_type,
                                template=template,
                                slew=float(s),
                                load=float(ell),
                                log_slew=log_s,
                                log_load=log_l,
                                value=float(vals[i, j]),
                            )


def extract_lib_to_dataframe(lib_paths: Iterable[Path]) -> pd.DataFrame:
    """讀取一個或多個 .lib 檔，回傳合併的 DataFrame。"""
    rows: list[Sample] = []
    for p in lib_paths:
        p = Path(p)
        print(f"  解析 {p.name} ...", end=" ", flush=True)
        before = len(rows)
        rows.extend(iter_cell_samples(p))
        print(f"抽出 {len(rows) - before:,} 筆")
    return pd.DataFrame([s.__dict__ for s in rows])


def extract_lib_to_parquet(lib_paths: Iterable[Path], out_path: Path) -> pd.DataFrame:
    """抽取並存成 parquet；回傳同一份 DataFrame 供進一步使用。"""
    df = extract_lib_to_dataframe(lib_paths)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n已存成 {out_path}  ({out_path.stat().st_size / 1024:,.1f} KB)")
    return df


def summarize(df: pd.DataFrame) -> None:
    """印出 DataFrame 的重點統計，方便你確認資料合理。"""
    print("\n" + "=" * 60)
    print(" 資料統計")
    print("=" * 60)
    print(f"總樣本數     : {len(df):,}")
    print(f"cell 數      : {df['cell'].nunique()}")
    print(f"library 數   : {df['library_name'].nunique()}")
    print(f"corner 分佈  : {dict(df['corner'].value_counts())}")
    print(f"table_type   : {dict(df['table_type'].value_counts())}")
    if "when" in df.columns:
        n_cond = (df["when"] != "").sum()
        print(f"conditional  : {n_cond:,} 筆帶 when（{100*n_cond/len(df):.1f}%）")
    print(f"slew 範圍    : {df['slew'].min():.4f} ~ {df['slew'].max():.4f}")
    print(f"load 範圍    : {df['load'].min():.6f} ~ {df['load'].max():.6f}")
    print(f"value 範圍   : {df['value'].min():.4f} ~ {df['value'].max():.4f}")
    # arc = (cell, output_pin, related_pin, when, table_type)
    # 這個 tuple 就是「一張 LUT 表」的唯一識別，per-table 訓練會用它
    arc_cols = ["cell", "output_pin", "related_pin", "when", "table_type"]
    if all(c in df.columns for c in arc_cols):
        n_arcs = df.groupby(arc_cols, sort=False).ngroups
        pts_per_arc = df.groupby(arc_cols, sort=False).size()
        print(f"\n唯一 (cell,pin,related,when,table) 表格數: {n_arcs:,}")
        print(f"每張表的格點數     : min={pts_per_arc.min()} / "
              f"median={int(pts_per_arc.median())} / max={pts_per_arc.max()}")


# ============================================================
# 2. PyTorch Dataset
# ============================================================

@dataclass
class NormStats:
    """正規化用統計量。用訓練集算出來，測試集共用。"""
    x_mean: np.ndarray   # shape (n_features,)
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def norm_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def norm_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def denorm_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean


class NldmDataset(Dataset):
    """從 parquet 載入的 NLDM 訓練資料集。

    最簡設定：僅用 (slew, load) 當輸入，value 當輸出。
    如需支援多 cell / 多 corner 的統一模型，透過 `cell_index` 屬性做 embedding。

    Args:
        parquet_path: 由 extract_lib_to_parquet 產出的 parquet 檔
        cells: 只保留這些 cell（None = 全部）
        table_types: 只保留這些 table（例如 ["cell_rise"]）
        corners: 只保留這些 corner（例如 ["TT"]）
        stats: 若提供，用它做正規化（測試集通常用訓練集的 stats）
        return_cell_index: 是否額外回傳 cell 的整數索引（給 embedding 用）
    """

    def __init__(
        self,
        parquet_path: str | Path,
        cells: list[str] | None = None,
        table_types: list[str] | None = None,
        corners: list[str] | None = None,
        stats: NormStats | None = None,
        return_cell_index: bool = False,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        if cells is not None:
            df = df[df["cell"].isin(cells)]
        if table_types is not None:
            df = df[df["table_type"].isin(table_types)]
        if corners is not None:
            df = df[df["corner"].isin(corners)]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("過濾後沒有樣本，請放寬條件。")

        self.df = df
        self.return_cell_index = return_cell_index

        # 建立 cell → 索引映射
        self._cell_vocab = {c: i for i, c in enumerate(sorted(df["cell"].unique()))}

        X_raw = df[["slew", "load"]].to_numpy(dtype=np.float32)
        y_raw = df["value"].to_numpy(dtype=np.float32).reshape(-1, 1)

        if stats is None:
            stats = NormStats(
                x_mean=X_raw.mean(axis=0),
                x_std=X_raw.std(axis=0) + 1e-8,
                y_mean=float(y_raw.mean()),
                y_std=float(y_raw.std()) + 1e-8,
            )
        self.stats = stats

        self.X = stats.norm_x(X_raw).astype(np.float32)
        self.y = stats.norm_y(y_raw).astype(np.float32)
        self.cell_idx = np.array(
            [self._cell_vocab[c] for c in df["cell"]], dtype=np.int64
        )

    # 讓外部能取到 cell 詞彙表（給 embedding 層決定大小用）
    @property
    def num_cells(self) -> int:
        return len(self._cell_vocab)

    @property
    def cell_vocab(self) -> dict[str, int]:
        return dict(self._cell_vocab)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        if self.return_cell_index:
            return x, torch.tensor(self.cell_idx[idx], dtype=torch.long), y
        return x, y


def train_test_split_by_grid(
    df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把資料隨機切訓練/測試。

    注意：這是「格點層次」的隨機切，適合驗證內插精度。
    若要驗證外推能力，應改成留下 slew/load 的邊界格點做測試（可另寫函式）。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_ratio)
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


# ============================================================
# 3. CLI
# ============================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Extract NLDM tables from .lib files.")
    parser.add_argument("--lib", action="append", required=True, help="輸入 .lib 檔（可重複指定）")
    parser.add_argument("--out", required=True, help="輸出 parquet 路徑")
    parser.add_argument("--no-summary", action="store_true", help="不印統計")
    args = parser.parse_args()

    lib_paths = [Path(p) for p in args.lib]
    for p in lib_paths:
        if not p.exists():
            raise SystemExit(f"檔案不存在：{p}")

    print(f"抽取 {len(lib_paths)} 個 .lib 檔：")
    df = extract_lib_to_parquet(lib_paths, Path(args.out))
    if not args.no_summary:
        summarize(df)


if __name__ == "__main__":
    _cli()
