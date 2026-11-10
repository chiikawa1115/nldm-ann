# ANN Checkpoint Metadata Export Notes for Ridge Comparison

> **匯出日期**：2026-11-10  
> **匯出主機**：本機 Windows (GTX 1070 訓練主機)  
> **Python 環境**：Conda `nldm-v4` (Python 3.12.15, PyTorch 2.14.1+cu126)  
> **目標分支**：`plan-v4`  
> **參考文檔**：`docs/ANN_METADATA_EXPORT_HANDOFF.md`、`docs/PLAN_GLOBAL_ANN_RIDGE_COMPARISON.md`

---

## 匯出結果摘要

依據 `docs/ANN_METADATA_EXPORT_HANDOFF.md` 之要求，使用 `scripts/export_ann_ridge_metadata.py` 成功匯出全量指定之 10 組配對實驗的 ANN checkpoint metadata。十組皆完整導出，無缺失欄位，無錯誤。

### 匯出清單（共 10 組）

| Mode | Seed | Run ID | Completed Epochs | 狀態 |
| :--- | :---: | :--- | :---: | :---: |
| `random_point` | 11 | `random_point_seed11_modelseed11` | 34 | 成功匯出 ✅ |
| `random_point` | 23 | `random_point_seed23_modelseed23` | 50 | 成功匯出 ✅ |
| `random_point` | 37 | `random_point_seed37_modelseed37` | 50 | 成功匯出 ✅ |
| `random_point` | 53 | `random_point_seed53_modelseed53` | 47 | 成功匯出 ✅ |
| `random_point` | 71 | `random_point_seed71_modelseed71` | 42 | 成功匯出 ✅ |
| `unseen_cell` | 11 | `unseen_cell_seed11_modelseed11` | 38 | 成功匯出 ✅ |
| `unseen_cell` | 23 | `unseen_cell_seed23_modelseed23` | 12 | 成功匯出 ✅ |
| `unseen_cell` | 37 | `unseen_cell_seed37_modelseed37` | 24 | 成功匯出 ✅ |
| `unseen_cell` | 53 | `unseen_cell_seed53_modelseed53` | 31 | 成功匯出 ✅ |
| `unseen_cell` | 71 | `unseen_cell_seed71_modelseed71` | 17 | 成功匯出 ✅ |

---

## 欄位與指紋核對說明

1. **資料集契約 (Contract)**：
   * `dataset_sha256`: `79ae211cec386a170455f6fdf0d38c1043d835786408d5aa93c94a1bec855ba5` (與基準 Parquet 完全一致)。
   * `eligible_cell_bases`: 200。
2. **特徵與正規化 (Normalization)**：
   * 特徵名稱與順序為標準 42 維度。
   * 各 run 之 `x_mean`, `x_std`, `y_mean`, `y_std`, `n_fit` 均完整由 checkpoint 保留記錄。
3. **訓練版本 (Git Commit & Sources)**：
   * 原 checkpoint 記錄 commit 為 `0032cdfeb1457bda850ff5752d1f1bf00ad3dd62`。
   * 六份核心訓練源碼之 SHA-256 指紋均成功由該 commit 提取並記錄。
4. **截取標記**：
   * `max_rows`: `null` (全量資料訓練，無截取)。

---

## 補充檔案 (Supplemental Data)

依照 Handoff 建議，另外一併提供了這 10 組 Run 的個別訓練收斂歷史與單 cell 指標：
* 路徑：`experiments/global_ann_ridge_metadata/supplemental/<run_id>/`
  * `model.history.json`
  * `per_cell_metrics.csv`
供後續深入分析各 cell 表現與訓練曲線。
