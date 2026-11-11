# NLDM-ANN 專案接力指南 (AGY Handoff)

## 目前接續點：Global ANN / Ridge 精度比較（2026-11-11）

- 使用者已確認研究問題：**在相同輸入資訊與測試資料下，global ANN 相較 global Ridge，是否降低預測誤差，以及改善幅度多少？**
- 本輪以精度為主；硬體不同的執行時間僅記錄，不作模型加速比較。
- 完整計畫：[PLAN_GLOBAL_ANN_RIDGE_COMPARISON.md](docs/PLAN_GLOBAL_ANN_RIDGE_COMPARISON.md)。可機讀規格：[global_ann_ridge_v1.json](experiments/configs/global_ann_ridge_v1.json)。
- 第一階段：`random_point`、`unseen_cell` × seeds `11,23,37,53,71`，新增 **10 個 Ridge runs**，逐一對照既有 ANN 結果。
- 固定原 42 維特徵、200 個 cell_base 的來源範圍、delay + transition、原 manifest / validation 規則；不在此輪改 L1、v4 arc 特徵或 delay-only。
- **目前狀態：Ridge 正式訓練與評估 10/10 完成；2026-11-11 接回提交 `17a3bdc` 的 ANN metadata 並重新 aggregate，十組來源資格全部 matched。** 本次沒有重訓或重跑 ANN 推論。實作版本 `af9e137`，原有 46 項相關測試通過。詳細結果見 [中文結果報告](docs/RESULTS_GLOBAL_ANN_RIDGE_COMPARISON.md) 與 [metadata 核對報告](docs/ANN_METADATA_VALIDATION_20261111.md)。
- 主要結果（五 seeds 的 per-cell macro MAE 均值）保持不變：random-point ANN 13.06 / Ridge 67.82 ps，逐 seed 平均改善 80.75%；unseen-cell ANN 22.33 / Ridge 77.12 ps，改善 71.33%。十組主要差值均支持 ANN 誤差較低。預定雙尾配對 t／Holm 已輸出，校正 p 分別為 5.1574e-9、1.0563e-5；n=5 且切分重疊，推論仍受獨立性與分布假設限制。
- 入口：`src/run_ridge_comparison.py`；[執行指南](docs/GLOBAL_ANN_RIDGE_RUNBOOK.md) 保留聚合命令。已完成來源核對，不需重訓；再次聚合必須提供 `--ann-metadata-dir experiments/global_ann_ridge_metadata`，以免退回 pending。
- **ANN 主機交接已完成：**十份 metadata、十份 history 與十份 per-cell CSV 均已帶回。原 [匯出交接](docs/ANN_METADATA_EXPORT_HANDOFF.md) 保留為操作紀錄；原訓練 commit 為 `0032cdfeb1457bda850ff5752d1f1bf00ad3dd62`，六份核心 source hashes 與既定來源相符。
- 實作包含 float64 串流統計、fit-only 正規化、validation 選 lambda、精確 percentile、逐 cell 指標、配對 CI / Holm、進度與產物 checksum；原 ANN 六個凍結來源檔未修改。
- 執行主機選目前 Ryzen 7 PRO 8840U 筆電（8C/16T、系統可用總 RAM 約 14.7 GiB）；使用 CPU 串流處理，不一次載入全矩陣。
- 十組訓練加評估合計 1,873.73 秒（31.23 分鐘），單組 124.69–248.50 秒，程序峰值記憶體 1.13 GiB；不含開發、等待核准與報告整理。所有 test n、groups、negative targets 均與對應 ANN 吻合。
- ANN 40 份 `metrics.json` 與彙總先前已核對一致；本次十組配對的 history／per-cell 檔也與 metadata／既有指標一致。逐 cell MAE：random-point 1,000/1,000、unseen-cell 197/200 個 cell/run 觀測為 ANN 較低；三個例外均在 unseen-cell seed 23。該 run 最佳 epoch=4、完成 epoch=12，符合 patience=8，誤差原因尚未確定。其餘 30 組不在本次 metadata 核對範圍。
- ANN 權重未帶回，checkpoint hash 只保留原主機記錄，沒有在本機重算；Git blob 也不能證明原訓練沒有未提交修改。不得把 metadata matched 說成完整重現 ANN。
- 原 40-run 實驗是每 run 一個 global model，**沒有按 type 做 L1，也沒有挑少量 cells 縮減來源資料**。以下歷史紀錄的 MAE/R2 混合 delay 與 transition，並非 v4 delay-only 主實驗。

接續方向：本輪 metadata 交接、來源核對與配對彙整已完成，可使用新補充資料分析 seed 23／尾端誤差，再與老師確認後續研究範圍；尚未啟動 L1 或減少資料的新實驗。config 保留執行前鎖定內容；最新比較資格請讀 `aggregate/comparison_status.json` 與 `aggregate/ann_metadata_audit.json`，原 training metadata/status 與 preflight audit 保留各階段的 pending 紀錄，不覆寫為訓練當時已 matched。模型與資料仍留本機，Git 保存小型結果。不要直接呼叫舊 ANN orchestrator，避免在沒有本機 checkpoint 時意外重訓 ANN。

---

## 歷史紀錄：2026-10-29 GPU 主機 ANN 基準

> **更新時間**：2026-10-29 19:20  
> **專案位置**：`C:\Users\User\claude_project\nldm-ann`  
> **Git 分支**：`plan-v4`  
> **Python 環境**：`F:\ana\envs\nldm-v4\python.exe` (`conda activate nldm-v4`)  
> **硬體環境**：本機 GTX 1070 8GB (CUDA 12.6, PyTorch cu126)  
> **訓練狀態**：🎉 **全量 40 Runs 泛化基準訓練已 100% 完成！**

---

## 🎯 任務目標與完成狀態
ASAP7 標準元件庫之神經網路非線性延遲模型 (NLDM-ANN) 的**全量泛化基準訓練 (40 Runs)** 已全數執行完畢，並成功產出最終評估報表。

* **總進度**：**40 / 40 Runs 完工 (100%)**
  * `random_point`：5 / 5 完成 ✅
  * `unseen_cell`：5 / 5 完成 ✅
  * `joint_unseen`：15 / 15 完成 ✅
  * `unseen_corner`：15 / 15 完成 ✅ (FF, SS, TT 各 5 seeds)

---

## 📊 最終泛化成果總覽 (Final Benchmark Results)

詳細報表位於：[`experiments/asap7_generalization_v1/runs/aggregate/`](file:///C:/Users/User/claude_project/nldm-ann/experiments/asap7_generalization_v1/runs/aggregate/)

| 泛化維度 (Mode) | Runs | Micro $R^2$ | Micro MAE (ps) | Micro RMSE (ps) | Macro $R^2$ | Macro MAE (ps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Point** | 5 | **0.9797** | **9.99** | **19.12** | **0.9595** | **13.06** |
| **Unseen Cell** | 5 | **0.9378** | **14.48** | **31.56** | **0.8755** | **22.33** |
| **Unseen Corner** | 15 | **0.8966** | **20.94** | **39.82** | **0.8265** | **25.63** |
| **Joint Unseen** | 15 | **0.8469** | **24.66** | **45.42** | **0.6399** | **35.42** |

---

## 📁 核心產出檔案清單
1. **各 Run 模型與個別評估**：
   * `experiments/asap7_generalization_v1/runs/<run_name>/model.pt` (共 40 份權重)
   * `experiments/asap7_generalization_v1/runs/<run_name>/evaluation/metrics.json`
2. **全域匯總報表**：
   * [`summary_by_mode.csv`](file:///C:/Users/User/claude_project/nldm-ann/experiments/asap7_generalization_v1/runs/aggregate/summary_by_mode.csv)：各模式統計（均值、標準差、95% 信賴區間）
   * [`summary_by_mode_corner.csv`](file:///C:/Users/User/claude_project/nldm-ann/experiments/asap7_generalization_v1/runs/aggregate/summary_by_mode_corner.csv)：細分 PVT Corner 的表現
   * [`run_metrics.csv`](file:///C:/Users/User/claude_project/nldm-ann/experiments/asap7_generalization_v1/runs/aggregate/run_metrics.csv)：全量 40 個 Runs 的單項明細表

---

## 🔍 新 Session 快速檢查指令
```powershell
# 確認 40 個 Runs 模型齊全
(Get-ChildItem C:\Users\User\claude_project\nldm-ann\experiments\asap7_generalization_v1\runs\*\model.pt).Count

# 查看匯總結果
Get-Content C:\Users\User\claude_project\nldm-ann\experiments\asap7_generalization_v1\runs\aggregate\summary_by_mode.csv
```
