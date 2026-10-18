# Feature-Conditioned NLDM + ANN 論文專案

> 使用 cell 結構與連續 PVT 特徵的 ANN 建模 NLDM，研究 unseen cell、
> unseen corner 與 joint-unseen 泛化。ASAP7 為主資料集，GF180MCU 為
> external validation；完整契約見 [`docs/RESEARCH_CONTRACT.md`](docs/RESEARCH_CONTRACT.md)。

---

## 目錄

1. [研究背景](#1-研究背景)
2. [專案架構](#2-專案架構)
3. [環境安裝與啟用](#3-環境安裝與啟用)
4. [從筆電遷移到桌電（GPU）](#4-從筆電遷移到桌電gpu)
5. [常用指令速查](#5-常用指令速查)
6. [下一步實作方向](#6-下一步實作方向)
7. [參考文獻與資源](#7-參考文獻與資源)

---

## 1. 研究背景

### 1.1 什麼是 NLDM？

**NLDM（Non-Linear Delay Model）** 是 EDA 工具（Static Timing Analysis, STA）
用來預測標準單元延遲的模型，寫在 Liberty（`.lib`）檔案中。它本質上是一張**查表**：

```
delay(input_slew, output_load) → 從 7×7 表格中雙線性插值
```

**限制**：
- 只在特徵化點附近準確，離網格點越遠越不準
- 表格越大越慢，但太小又不準
- 每個 PVT（Process-Voltage-Temperature）角點都要重跑 SPICE，成本極高
- 無法自然泛化到訓練範圍外的操作點

### 1.2 為什麼用 ANN？

近年（2022–2025）多篇論文顯示，用神經網路取代 LUT 可以：

| 方法 | 對比 SPICE 誤差 | 加速比 |
|---|---|---|
| 傳統 7×7 NLDM LUT | 較高 | 基準 |
| **DL-NLDM**（MLP） | ~1.4% | 顯著提升 |
| **DL-WFDM**（波形編碼 + ANN） | 更低 | 最佳 |
| **GNN-based**（含 cell 拓樸） | 誤差降 67–80% | **100× vs SPICE** |

### 1.3 本專案的研究定位

由於目前**沒有現成的完整端到端 DL-NLDM 開源實作**，這個專案的價值在於：

1. **建立可重現的基準**：liberty-parser → MLP → 對比 LUT 精度
2. **驗證論文結果**：復現 DL-NLDM / DL-WFDM 的主要發現
3. **擴充原創方向**：加入 GNN、參數化建模、Active Learning 等進階方法

---

## 2. 專案架構

```
nldm_ann/
├── README.md                    ← 本檔案
├── environment.yml              ← Conda 環境定義（跨機器遷移用）
├── requirements.txt             ← pip 備援（不用 conda 時）
├── .gitignore
│
├── data/                        ← 資料
│   ├── raw/                     ← 原始 .lib、SPICE 輸出
│   ├── processed/               ← 清洗後的訓練資料
│   └── external/                ← 開源 PDK（SkyWater 130nm、ASAP7）
│
├── src/                         ← 主要程式碼
│   ├── sanity_check.py          ← 環境自我檢測
│   ├── data_loader.py           ← .lib 解析與 dataset 建構（待寫）
│   ├── train.py                 ← 訓練腳本（待寫）
│   ├── evaluate.py              ← 精度評估 vs 傳統 LUT（待寫）
│   └── models/
│       ├── mlp_nldm.py          ← MLP 基準模型
│       └── gnn_nldm.py          ← GNN 進階模型（待寫）
│
├── notebooks/                   ← Jupyter 探索性分析
├── experiments/                 ← 每次實驗的設定與結果
├── checkpoints/                 ← 訓練好的模型權重（不進 git）
├── logs/                        ← TensorBoard / W&B 紀錄（不進 git）
└── docs/                        ← 論文相關筆記、圖表
```

**設計原則**：
- `src/` 只放**可重複執行**的程式碼；探索性工作留在 `notebooks/`
- `data/` 底下三層區隔（raw / processed / external）避免手動處理過的資料被覆蓋
- `experiments/` 每個實驗獨立資料夾，方便對比不同超參數的結果
- 大檔案（權重、資料集）**不進 git**，透過 `.gitignore` 排除

---

## 3. 環境安裝與啟用

### 3.1 目前狀態（筆電，CPU only）

Miniconda 已安裝於 `C:\Users\ruru8\miniconda3`，未加入系統 PATH（避免污染系統環境）。

### 3.2 啟用環境（每次開新 PowerShell 都要）

```powershell
# 方法一：直接呼叫 activate 腳本
& "C:\Users\ruru8\miniconda3\Scripts\activate.ps1"
conda activate nldm

# 方法二：一勞永逸（讓 conda 指令永久可用）
& "C:\Users\ruru8\miniconda3\Scripts\conda.exe" init powershell
# 之後重開 PowerShell 就能直接用 `conda activate nldm`
```

### 3.3 驗證環境

```powershell
conda activate nldm
python src/sanity_check.py
```

預期輸出會顯示 Python 版本、PyTorch 版本、GPU 狀態、以及所有核心套件是否可載入。

> **PowerShell 中文亂碼？**
> 若輸出的中文顯示為 `????` 或亂碼，是 Windows PowerShell 預設 codepage 為 cp950/cp1252 造成，
> 執行前先切成 UTF-8 即可：
> ```powershell
> chcp 65001
> $env:PYTHONIOENCODING = "utf-8"
> ```
> 或改用 VS Code 內建 terminal（預設 UTF-8）也不會有這問題。

---

## 4. 從筆電遷移到桌電（GPU）

> **PLAN v4（`plan-v4` branch）請改看 [`docs/PLAN_v4_GPU_HOST_RUNBOOK.md`](docs/PLAN_v4_GPU_HOST_RUNBOOK.md)。**
> GPU 主機只靠 git 與公開 ORFS 重建資料，不從開發筆電複製任何資料檔。以下為舊流程。

這是本專案設計時**特別注意**的：`environment.yml` 讓桌電只要一行指令就能重建環境。

### 4.1 桌電操作步驟

```powershell
# 1. 把整個 nldm_ann 資料夾複製到桌電（用 git、隨身碟、雲端都可）

# 2. 桌電也安裝 Miniconda（同樣的靜默安裝方式）

# 3. 修改 environment.yml：
#    移除 `- pytorch::cpuonly`
#    加入 `- pytorch::pytorch-cuda=12.1`（依桌電 CUDA 版本調整）

# 4. 建立環境
conda env create -f environment.yml

# 5. 啟用並驗證
conda activate nldm
python src/sanity_check.py
# 這時 CUDA 可用應顯示 True
```

### 4.2 查桌電 CUDA 版本

在桌電上執行 `nvidia-smi`，右上角會顯示驅動支援的最高 CUDA 版本。
選擇 PyTorch 官網有支援的版本（目前常見：11.8、12.1、12.4）。

---

## 5. 常用指令速查

| 目的 | 指令 |
|---|---|
| 啟用環境 | `conda activate nldm` |
| 離開環境 | `conda deactivate` |
| 列出所有環境 | `conda env list` |
| 更新環境（改了 yml 後） | `conda env update -f environment.yml --prune` |
| 刪掉環境重來 | `conda env remove -n nldm` |
| 匯出當前環境 | `conda env export --no-builds > environment.yml` |
| 啟動 Jupyter Lab | `jupyter lab` |
| 開啟 TensorBoard | `tensorboard --logdir logs/` |

---

## 6. 下一步實作方向

> 新實驗一律依 [`docs/GOAL_DRIVEN_PLAN.md`](docs/GOAL_DRIVEN_PLAN.md) 的 gate
> 驗收。既有 `experiments/expA` 使用 random-point holdout 與 scattered
> interpolation，保留作歷史結果，不可當作標準 NLDM bilinear baseline 的論文證據。

按建議實作順序：

### 階段 1：資料 pipeline（先做這個！）
1. 到 [SkyWater 130nm PDK](https://github.com/google/skywater-pdk) 或
   [ASAP7](https://github.com/The-OpenROAD-Project/asap7) 下載開源 `.lib` 檔案
2. 用 `liberty-parser` 抽出所有 cell 的 `cell_rise`, `cell_fall`, `rise_transition`, `fall_transition`
3. 建立 (input_slew, output_load) → (delay, transition) 的資料集
4. 存成 `data/processed/*.parquet`

### 階段 2：MLP 基準
1. 用 `src/models/mlp_nldm.py` 訓練 per-cell MLP
2. 用 hold-out 網格點測試（訓練時遮住某些 slew/load 點）
3. 對比雙線性插值的誤差 → 這是論文的第一個實驗結果

訓練 CLI 會在 checkpoint 旁建立 `*.split.csv` 與 metadata JSON，記錄完整
Liberty table identity、dataset/manifest SHA-256、seed、套件版本與 git commit。
評估只接受 hash 驗證通過的新版 manifest，避免重切資料或跨 PVT 洩漏。

Experiment A v2 以單一設定檔執行 5 個固定 seeds、彙整 95% confidence
interval，並產生論文圖表：

```powershell
python src/run_experiment.py --config experiments/configs/expA_v2.json
```

程序可續跑：已存在的 `result.json` 不會重算。只重新彙整現有結果可加
`--aggregate-only`。輸出包含逐 seed metrics、平均與 CI、paired
`MLP - LUT` 差值及 `accuracy_ci.png`。

### 階段 3：泛化與參數化
1. 加入 process、voltage、temperature 數值與 cell 結構特徵
2. 執行 random-point、unseen-cell、unseen-corner、joint-unseen split
3. 報告 micro-average 與 per-cell macro-average，多 seeds mean ± SD/CI
4. cell/corner ID embedding 僅可作 transductive baseline，不可支持 unseen claim

### 階段 4：進階（選做）
1. GNN 加入 cell 內部電晶體拓樸
2. Active Learning 減少 SPICE 訓練資料需求
3. Physics-informed loss（例如強制 delay 對 load 單調遞增）

---

## 7. 參考文獻與資源

### 論文
- Klemme & Amrouch, "Deep-learning cell-delay modeling for STA", 2022
  <https://www.sciencedirect.com/science/article/pii/S2090447922001393>
- "Parametric Subthreshold Cell Delay Modeling Based on ANN", 2024
  <https://www.researchgate.net/publication/388090644>
- "Fast Cell Library Characterization via GNN", arXiv 2312.12784
  <https://arxiv.org/html/2312.12784v2>
- "CCS Modeling with GP + Active Learning", arXiv 2505.10799
  <https://arxiv.org/pdf/2505.10799>

### 工具
- [liberty-parser](https://pypi.org/project/liberty-parser/) — 純 Python `.lib` 解析器
- [OpenSTA](https://github.com/parallaxsw/OpenSTA) — 開源 STA 工具，可用來對照 EDA 標準做法
- [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) — 完整開源 EDA 流程

### 開源 PDK（訓練資料來源）
- [SkyWater 130nm](https://github.com/google/skywater-pdk) — Google 主導，教學資源多
- [ASAP7](https://github.com/The-OpenROAD-Project/asap7) — ASU 學術 7nm PDK
- [FreePDK45](https://eda.ncsu.edu/freepdk/freepdk45/) — NCSU 45nm

---

_最後更新：2026-07-12_
# GPU formal training

The main unseen-cell/PVT experiment uses the streaming global trainer, not the legacy
per-table Experiment A CLI. Follow [`docs/GPU_TRAINING_RUNBOOK.md`](docs/GPU_TRAINING_RUNBOOK.md).
The repository contains 28 immutable split contracts that expand to 40 formal model runs.
The ASAP7 Parquet is excluded from Git and must be copied or rebuilt and hash-verified on
the GPU host. Run `scripts/gpu_preflight.py` before any long job.
