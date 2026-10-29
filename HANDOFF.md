# NLDM-ANN 專案接力指南 (AGY Handoff)

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
