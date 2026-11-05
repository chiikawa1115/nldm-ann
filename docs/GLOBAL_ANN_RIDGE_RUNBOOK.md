# Global ANN / Ridge 執行指南

更新：2026-11-03。研究規格見 [比較計畫](PLAN_GLOBAL_ANN_RIDGE_COMPARISON.md)，鎖定設定見 [config](../experiments/configs/global_ann_ridge_v1.json)。

## 目前執行狀態

2026-11-03 使用者授權後，**十組 Ridge 全部完成，重新執行 aggregate 的輸入與產物 checksum 核對也已通過。** 沒有重訓或重跑 ANN 推論；十組 ANN checkpoint metadata 仍 pending。結果見 [中文結果報告](RESULTS_GLOBAL_ANN_RIDGE_COMPARISON.md)。以下保留操作命令；接續只需補 metadata 後重新聚合，不需重訓。

十組訓練加評估合計 1,873.73 秒（31.23 分鐘），單組 124.69–248.50 秒，程序峰值記憶體 1,214,738,432 bytes（約 1.13 GiB）。時間不包含開發、等待核准、額外聚合及報告整理。此為本機 Ridge 執行紀錄，不用來宣稱相對於另一台電腦 ANN 的加速比。

首個完整 run 實測 248.50 秒（fit 134.42 秒、evaluate 114.07 秒）、peak working set 1.06 GiB，test n=2,463,920、groups=200、negative targets=872 均核對通過。這是完整訓練加評估的耗時，不包含程式開發或等待核准時間。

- 46 項測試通過（17.30 秒）：Ridge 新增 27 項，既有特徵／切分／指標相關 19 項。
- 合成資料為 432 rows、4 cells、3 corners、4 table types。結果只留在忽略的 `tests/runtime/`，不混入正式實驗。
- 以獨立的 augmented least squares 驗算六個 lambda、截距、validation MSE；包含共線、常數欄位、常數及負 target。
- 驗證跨 table 的 batch 邊界、fit-only 正規化、原 ANN normalization、精確 micro/macro percentile、近零值與 undefined groups。
- 驗證來源與產物變更攔截、pending/matched/mismatch、完成結果重用、中斷後重用模型、配對差值與 CI / Holm。
- 正式資料 preflight PASS：12,073,208 rows、42 features、10 組固定配對、來源檔與輸入 hash 相符；**10 組 ANN metadata 仍 pending**。preflight 只讀 Parquet metadata 與檔案 hash，不掃描全部 row 以擬合模型。

驗證命令（專案根目錄，PowerShell）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ridge_comparison.py tests/test_global_engine.py tests/test_generalization.py tests/test_experiment.py -q --basetemp tests/runtime/ridge_validation_20261103
.\.venv\Scripts\python.exe src/run_ridge_comparison.py preflight
```

`--basetemp` 是 pytest 專用暫存目錄，重新執行可能清理該目錄，請勿放個人資料。Windows 沙箱曾阻擋 pytest 暫存存取；上述成功測試在核准的沙箱外環境執行。CLI 和 metadata exporter 的 `--help` 均已確認可用。

## 啟動正式比較

從 `nldm-ann` 根目錄執行。使用現有 `.venv`，所需 numpy、pandas、scipy、pyarrow、torch、matplotlib 已存在；求解器不依賴 scikit-learn。Torch 僅因沿用原 ANN helper 而載入，不會啟動 ANN 訓練。

先跑一個完整 seed（已獲授權並完成）：

```powershell
.\.venv\Scripts\python.exe src/run_ridge_comparison.py run --run-id random_point_seed11_modelseed11
```

首個 run 依序：串流 fit/validation moments → 六組候選解與 validation 選參數 → 串流 test 推論 → 精確指標 → 配對摘要。它會讀完全部來源資料，沒有 `max_rows` 截斷。預設 batch 100,000 rows，保留完整 table；test targets/predictions 暫存磁碟，不保留全量特徵矩陣。

檢查 `evaluations/random_point_seed11_modelseed11/` 的：

- `status.json`：必須為 `complete`，記錄此程序的峰值記憶體與評估耗時。
- `training_metadata.json`：fit/validation/test 筆數、六個 validation losses、lambda/alpha、fit 耗時與環境版本。
- `metrics.json`：test n = 2,463,920、cell groups = 200、negative targets = 872；程式會自動核對，差異即失敗。

峰值記憶體是本程序啟動至今的 peak working set/RSS；同一程序連跑多個 seeds 時不是每個 run 的獨立峰值。前次 1–3 小時是全 10 runs 的初估，本次已取得上述實測值。凍結 config 的 `planned_not_run` / `formal_training_started=false` 保留為執行前快照，實際完成狀態請讀各 run 的 `status.json` 與 `aggregate/comparison_status.json`，不要修改 config 使既有 signature 失效。

首個 run 確認後：

```powershell
.\.venv\Scripts\python.exe src/run_ridge_comparison.py run --all
```

已完成的首個 run 通過 signature/checksum 檢查後會直接重用；只擬合其餘九個。各 mode 分別彙總五組配對。`--all` 遇失敗即停止，已完成結果保留；失敗原因記錄於該 run 的 status。也可重複傳入 `--run-id` 指定接續 runs。

正式輸出根目錄為 `experiments/global_ann_ridge_v1/`。只用一個程序寫同一個輸出目錄。

## 補齊 ANN 訓練來源

交給原訓練電腦上的 AI 執行時，請先讀 [ANN metadata 匯出交接](ANN_METADATA_EXPORT_HANDOFF.md)，其中列出十個 run、核對欄位、Git 交付方式與缺失處理。

在持有原十個 `model.pt` 的訓練電腦先取得本次程式，再用該電腦的 Python 環境執行：

```powershell
python scripts/export_ann_ridge_metadata.py
```

輸出到 `experiments/global_ann_ridge_metadata/<run_id>.json`，可透過 Git 帶回本機；不需要上傳全部 ANN 權重。匯出會載入自己產生且可信任的 PyTorch checkpoint，保留 normalization、training config、seed、contract、訓練 commit、該 commit 的來源 hash，以及 checkpoint checksum。若原始欄位或 Git 歷史缺失，保留缺失狀態。

回到本機後，可在 `preflight`、`run`、`aggregate` 任一命令附上：

```powershell
--ann-metadata-dir experiments/global_ann_ridge_metadata
```

即使 metadata 齊全，preflight 尚未計算 Ridge normalization，因此仍可能 pending。完整 run 的 fit 階段完成後才能比較所有 normalization 欄位。容差 `rtol=1e-6, atol=1e-8`，`n_fit` 完全相等。來源碼與 metadata 相符也不能獨立證明當年訓練沒有未提交修改；報告會保留這項限制。

若 Ridge 已完成才補 metadata，可只重新聚合，不需重訓：

```powershell
.\.venv\Scripts\python.exe src/run_ridge_comparison.py aggregate --ann-metadata-dir experiments/global_ann_ridge_metadata
```

未提供 `--ann-metadata-dir` 時不會自動猜測位置，配對會維持 pending。發現不一致時不輸出正式比較結論，先查看 `input_audit.json` 或 `aggregate/comparison_status.json`。

## 中斷與結果核對

```powershell
.\.venv\Scripts\python.exe src/run_ridge_comparison.py run --run-id random_point_seed11_modelseed11 --retry-incomplete
```

只有明確加 `--retry-incomplete` 才重試未完成 run。若已有 `fit_complete.json` 且檔案 checksum 相符，重用模型，重新執行評估；若 fit 尚未完成，從頭累積 moments。signature 包含 config、來源程式、套件版本與 batch 大小，任一變更會拒絕重用。需換設定時，使用另一個獨立 `--out-dir`，重新從首個 run 開始；不要修改 marker 假裝相符。

按 Ctrl+C 會盡可能保存 interrupted 狀態並清理該次暫存。作業系統強制終止可能留下 `.scratch/ridge-eval-*`，它們不參與續跑也不進 Git。程式不自動清理不明目錄。

主要產物：`aggregate/paired_metrics.csv`、`aggregate/summary_by_mode.csv`、`aggregate/comparison_status.json`、`figures/` 與 `report.md`。完整比較需 10/10；pending、失敗、缺失與不一致不得省略。兩個 mode 各五組來源 matched、差值變異有限且非零時，才提供主要 macro MAE 的雙尾配對 t / Holm；其他數值屬描述性分析。

`--no-plots` 可略過製圖，既有圖可能過時，report/status 會標註。本機模型 `.npz` 與暫存不進 Git；小型 JSON、逐 cell 指標、配對 CSV、圖及報告可保存。跨機續跑或重新聚合需同時具備 checksum 清單中的模型等原始產物；只有 Git 摘要時，可閱讀已保存報告，不能偽裝為重新驗證過的 run。
