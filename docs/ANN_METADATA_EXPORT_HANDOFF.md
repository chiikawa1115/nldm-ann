# 給原 ANN 訓練主機 AI 的交接：匯出比較所需 metadata

更新：2026-11-05。目標 repo：`https://github.com/chiikawa1115/nldm-ann.git`，分支：`plan-v4`。

**完成註記（2026-11-11）：**原主機已於提交 `17a3bdc` 提供十份 metadata 及十組 supplemental history／per-cell 檔案；Ridge 主機接回後十組均 matched。結果見 [核對報告](ANN_METADATA_VALIDATION_20261111.md)。以下保留原交接需求與命令，無需再次匯出或訓練。

## 任務與目前狀態

請在**保存原 ANN checkpoint 的訓練電腦**上，匯出下列十組既有模型的 metadata，檢查完整性後 commit / push 到 `plan-v4`，供 Ridge 執行主機 pull 回核對。

Ridge 十組已完成，結果 commit 為 `48118bb`；目前十組比較資格皆為 `ann_provenance_pending`，原因是缺少原 ANN checkpoint 的實際訓練資訊。本任務是匯出既有檔案，**不需要重訓 ANN、重跑 ANN 推論或執行 Ridge**。既有 Parquet、ANN metrics 與 manifests 已在 Ridge 主機。

先讀本文件，再視需要參照：

- [執行規格](PLAN_GLOBAL_ANN_RIDGE_COMPARISON.md)
- [固定十組清單與 hashes](../experiments/configs/global_ann_ridge_v1.json)
- [既有匯出工具](../scripts/export_ann_ridge_metadata.py)
- [目前結果與限制](RESULTS_GLOBAL_ANN_RIDGE_COMPARISON.md)

## 要匯出的十組

| Mode | Seed | Run ID／輸出 JSON 檔名（加 `.json`） |
|---|---:|---|
| random_point | 11 | `random_point_seed11_modelseed11` |
| random_point | 23 | `random_point_seed23_modelseed23` |
| random_point | 37 | `random_point_seed37_modelseed37` |
| random_point | 53 | `random_point_seed53_modelseed53` |
| random_point | 71 | `random_point_seed71_modelseed71` |
| unseen_cell | 11 | `unseen_cell_seed11_modelseed11` |
| unseen_cell | 23 | `unseen_cell_seed23_modelseed23` |
| unseen_cell | 37 | `unseen_cell_seed37_modelseed37` |
| unseen_cell | 53 | `unseen_cell_seed53_modelseed53` |
| unseen_cell | 71 | `unseen_cell_seed71_modelseed71` |

原 checkpoint 通常位於：

```text
experiments/asap7_generalization_v1/runs/<run_id>/model.pt
```

匯出工具實際依各 run 的 `evaluation/metrics.json` 中 `checkpoint` 路徑讀取模型。應使用產生該份 ANN 指標的原 checkpoint，不能以其他 seed、其他訓練版本或 `.last.pt` 代替。

## 操作步驟

在原電腦找到此 repo，啟用原本訓練 ANN 的 Python 環境。此環境應能載入 PyTorch checkpoint，並有 numpy、pandas、pyarrow。先檢查工作目錄與分支；若有本機修改，保留並處理後再同步，勿覆蓋原始訓練資料。

```powershell
git status --short
git switch plan-v4
git pull --ff-only origin plan-v4
python scripts/export_ann_ridge_metadata.py
```

成功會輸出十行 `exported=<run_id>`，產生：

```text
experiments/global_ann_ridge_metadata/<run_id>.json
```

工具載入原 checkpoint，複製其記錄的設定，並查詢**checkpoint 所記錄 commit**的來源檔 hash。它不會訓練模型，也不需要讀取完整 Parquet。只載入本專案自己產生且可信任的 checkpoint。

若模型不存在或路徑已移動，先確認對應原檔位置；不要修改已發布的 ANN metrics/manifests 來假裝相符。工具遇錯會停止，前面已匯出的 JSON 會保留。回報缺少或失敗的 run ID 與錯誤，解決原檔位置後可重新執行。

## JSON 需要保留的資訊

| 欄位 | 核對用途 |
|---|---|
| `run_id`、`seed` | 對應正確的實驗與重複次數 |
| `contract`、`manifest_sha256` | 原切分規格；contract 內含 dataset hash |
| `normalization.feature_names` | 原 ANN 使用的 42 維特徵與順序 |
| `normalization.x_mean/x_std/y_mean/y_std/n_fit` | fit partition 的正規化與筆數 |
| `max_rows`、`training_config` | 當時是否截取資料，以及 epochs、batch、學習率、架構等設定 |
| `git_commit`、`training_source_sha256` | 實際記錄的訓練版本及該版本來源檔 hash |
| `checkpoint_sha256` | 匯出資訊對應哪一份模型檔 |
| `completed_epochs`、`best_validation_mse_norm`、版本資訊 | 補充訓練紀錄；依原 checkpoint 實際可取得內容保存 |

**照實匯出，不為通過核對而補造或修改欄位。** `max_rows` 非 null、特徵不同或 source hash 不同時仍應保留原值並回報。checkpoint 沒有保存的欄位保留缺失；不可把現在的 Git HEAD 填成當時訓練 commit，也不可用 Ridge 的 normalization 代填 ANN 欄位。`completed_epochs` 是工具由 checkpoint history 長度取得，若 history 缺失，不能把輸出的 0 當作已證實的實際訓練 epoch 數。

若 checkpoint 指向的 Git commit 不在本機歷史中，可以從原 repo 補取所需歷史後重匯出；仍不可取得時，保留 hash 缺失並說明。Git commit hash 本身不能獨立證明當年訓練沒有未提交的來源碼修改。

## 檢查與交付

逐一確認上述十個 Run ID 的 JSON 可解析、檔名與內容的 run/seed 一致，並列出缺少的欄位、原檔或來源歷史。**十個檔案成功輸出，不等於比較資格已 matched。** 最後的 normalization 與來源核對由 Ridge 主機執行。

請附一份 `experiments/global_ann_ridge_metadata/EXPORT_NOTES.md`，簡短記錄匯出日期、使用的 Python 環境、成功／失敗 run ID、缺欄位或已知差異；沒有缺失就明確記錄十組成功。不要將密碼、token 或無關環境資訊寫入文件。

只提交十份小型 metadata JSON 與匯出說明：

```powershell
git add -- experiments/global_ann_ridge_metadata
git diff --cached --stat
git commit -m "data: export ANN checkpoint metadata for Ridge comparison"
git push origin HEAD:plan-v4
```

提交前確認 staging 沒有混入原主機其他工作；若遠端已有新提交，正常同步整合後再 push，不使用 force push。缺失無法解決時，交付已取得的 metadata 與完整缺失清單，不能把十組狀態寫成全部成功。

不需要上傳 `.pt`、Parquet、快取或逐點預測。完成後回覆：commit hash、分支、成功份數及未解決項目。這就是原訓練主機本次任務的完成點。

## 可選的後續分析資料

若容易取得，可另提供這十組 ANN 的 `evaluation/per_cell_metrics.csv` 與 `model.history.json`，供分析個別 cells 與收斂情況；不影響本次 metadata 優先交付。原 `runs/` 路徑被 Git 忽略，若另外提供，複製到 `experiments/global_ann_ridge_metadata/supplemental/<run_id>/`，保留來源說明，不搬動原檔。

## Ridge 主機接回後的動作（供接手者了解）

這一步在保有 Ridge `.npz` 模型與完整結果的主機執行，原 ANN 訓練主機不必執行：

```powershell
git pull --ff-only origin plan-v4
.\.venv\Scripts\python.exe src/run_ridge_comparison.py aggregate --ann-metadata-dir experiments/global_ann_ridge_metadata
```

此命令重新核對來源與產物，再產生配對摘要，**不重訓**。正規化容差為 `rtol=1e-6, atol=1e-8`，`n_fit` 必須完全相同；manifest 核對容許 LF/CRLF 的換行差異。若出現 mismatch，保留原始證據並釐清差異，不直接修改凍結 config 或放寬資格以保留有利結果。
