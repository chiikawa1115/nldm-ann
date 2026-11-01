# Global ANN 與 Global Ridge 配對精度比較計畫

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent / plan
- Origin Date: 2026-11-01
- Version Label: global_ann_ridge_v1
- Verification Status: UNVERIFIED（比較尚未執行；既有 ANN 報表已做一致性核對）
- 使用者決策：以預測精度與改善幅度為主；耗時僅作執行紀錄，不作主要比較結論。
- 本次工作邊界：先保存並推送計畫與交接紀錄；正式 Ridge 訓練尚未啟動。
- 本計畫在已閱讀 ANN 結果、尚未取得 Ridge 結果後訂定，不宣稱盲態事前註冊。

## 1. 研究問題與主張範圍

> 在相同輸入資訊與測試資料下，global ANN 相較 global Ridge，是否降低預測誤差，以及改善幅度多少？

評估兩個獨立情境：同表 random-point 預測與 unseen-cell 泛化。假設 ANN 可能降低誤差，但不預設其必須勝出；Ridge 相近或較好同樣是有效結果。

本輪不回答 L0/L1/L2 合併效果、跨 VT/size 的 v4 主實驗、物理約束、sign-off 安全或跨製程泛化。兩種模型在不同電腦執行，耗時不能直接解讀為模型加速比。

## 2. 既有實驗的正確定位

- 參考 Git commit：`4178b2f8d74cfd2ebe3454283fea969da3c44a75`。這是結果發布版本，不冒充各 ANN checkpoint 的實際訓練版本。
- 資料：ASAP7，12,073,208 rows，200 個 `cell_base`，含 SEQ。
- 目標：`cell_rise`、`cell_fall`、`rise_transition`、`fall_transition` 的 `value`，單位 ps；混合評估不能稱為 delay-only 結果。
- 模型：每個 run 以一個 global ANN 學習該次 fit partition；沒有按 function type 訓練 L1，也沒有先挑少數 cells 縮小來源資料。
- 特徵：`src/features.py` 的 42 欄；不使用 v4 的 69 欄候選特徵快取。舊特徵缺少 `when`、pin role、timing sense 的限制對本輪兩種模型保持一致。
- 既有 40 runs 中，本輪只配對 random-point 5 組與 unseen-cell 5 組；不混入 unseen-corner / joint-unseen 的平均。
- 目前已核對 40 份 JSON 與三份 aggregate CSV 相符、資料集 hash 相符、測試筆數相符；沒有重跑 ANN 推論。

## 3. 固定輸入與實驗矩陣

| 項目 | 設定 |
|---|---|
| Dataset | `data/processed/asap7_nldm.parquet` |
| Dataset SHA-256 | `79ae211cec386a170455f6fdf0d38c1043d835786408d5aa93c94a1bec855ba5` |
| Manifests | `experiments/asap7_generalization_v1/manifests/` |
| ANN results | `experiments/asap7_generalization_v1/runs/<run>/evaluation/metrics.json` |
| Split modes | `random_point`, `unseen_cell` |
| Split / model seeds | `11, 23, 37, 53, 71`，逐 seed 配對 |
| 新增 Ridge runs | 10；第一個完整 run 為 `random_point_seed11_modelseed11` |
| 可機讀計畫 | `experiments/configs/global_ann_ridge_v1.json`（計畫規格，尚非已實作 CLI） |

`random_point` 保留每張完整 table 的 round(n × 0.2) 個點作 test（49 點表格為 10 點）。`unseen_cell` 沿用各 manifest 指定的 40 個 held-out cell_base，其餘 160 個進入 train partition。保留 cell_base 的全部 VT variants 分組規則。

資料以完整 table 為邊界串流處理；調整 batch 大小不得改變點的身分或切分結果。`--batch-rows` 表示分批，不能用 `max_rows` 截取正式實驗。跨 batch 的切分不變性必須在小型測試驗證。

## 4. ANN 來源核對與比較資格

從原訓練電腦為十個對應 checkpoint 匯出小型 metadata JSON：

- run ID、seed、完整 contract、dataset / manifest SHA-256。
- `git_commit`、`training_config`、`max_rows`。
- `normalization`：feature_names、x_mean/x_std、y_mean/y_std、n_fit。
- 實際 epochs、best validation loss、Python / package versions（若原 checkpoint 有保存）。

不要求先搬移全部模型權重。訓練 history 與 per-cell metrics 可另行補上，補充收斂與分層分析。

資格狀態分開保存：

1. **matched**：資料、特徵、實際切分、訓練範圍與前處理已核對相符。
2. **ann_provenance_pending**：可執行 Ridge 並產生暫定配對表，但缺 ANN 實際訓練設定，不能宣稱已完全核對或重現 ANN。
3. **mismatch**：發現不一致，停止將該 run 納入正式配對摘要；記錄原因，先解決差異。

JSON 語意 hash 用於跨平台內容核對（UTF-8、sort_keys、緊湊 separators），另外保留實際檔案 hash。不能因 Windows CRLF / LF 差異而偷偷更新舊 manifest。資料集使用原始檔案 SHA-256。

## 5. Ridge 定義、前處理與選參數

固定沿用 `src/features.py`、`src/global_engine.py` 的輸入資訊、切分與 validation 規則。模型學習的是同一個 scalar target，table type 保留在輸入。

- 在 manifest 的 train partition 中，以對應 run seed 的既有 `validation_mask(..., ratio=0.1)` 留出 validation；其餘稱為 **fit partition**。
- x 與 y 的正規化只由 fit partition 計算，沿用既有常數欄位與近零變異處理規則。記錄並比較 ANN normalization；數值累積差異須使用事前指定容差。
- 原始 feature builder 產生 float32 特徵；Ridge 統計累積及求解使用 float64，保存數值精度與求解器設定。
- 所有 fit rows 等權；不新增 per-type reweighting、target log transform、特徵交互項或目標資訊。
- 保留負 target；預測不新增裁切或物理約束。

在正規化座標中擬合帶截距的 Ridge：

`mean((y - (Xw + b))^2) + lambda * ||w||_2^2`

截距 b 不正則化。固定候選 `lambda = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]`。若使用 sklearn 的 sum-squared-error 定義，換算 `alpha = n_fit * lambda`；保存兩個數值及 n_fit。

- 每個 run 以 validation MSE 最低選 lambda，與 ANN 的 validation MSE 選 checkpoint 原則一致。
- 若 MSE 在明確記錄的數值容差內相同，選較大的 lambda；全部候選與 loss 都保存。
- 不使用 test 結果挑 lambda、不擴張候選範圍、不在選參數後用 fit+validation 重訓。
- 正規化、截距、候選解與 validation MSE 必須先以小型資料對照可信的直接解或套件解；不能只驗證輸出檔存在。

## 6. 執行資源與驗證順序

目標主機：目前筆電 Ryzen 7 PRO 8840U，8 cores / 16 threads，系統可用總 RAM 約 14.7 GiB。查詢時空閒約 3.7 GiB，這不是執行時保證。

- 使用 CPU，按完整 table 分批；先以約 100,000 rows 為 batch 目標，必要時縮小，不能因此丟棄資料。
- 不一次 materialize 全量 42 維矩陣（float64 約 3.78 GiB，尚未計中間副本）。逐批累積充分統計量，針對多個 lambda 重用統計量。
- Test 評估也須控制記憶體。MAE / RMSE / R2 可累積統計；P90 / P99 必須保持既有 exact percentile 定義，可使用暫存檔或逐 cell 計算，不擅自替換為近似 quantile。
- 先以合成小資料驗證數值、資料邊界與統計程式，再做第一個完整 seed；小型測試結果不可混入正式摘要。
- 首個完整 seed 記錄峰值記憶體與各階段耗時，核對 test n、cell groups、負 target 數後，才執行其餘九組。
- 每階段記錄進度；失敗 run 留下原因，不靜默跳過或將部分結果標為 10/10。
- 先前 1–3 小時估計指全部 10 runs，不含程式實作；只是局部計時推估。預估不是 hard timeout，不依它自動終止或刪除產物。
- 實作後才能提供正式 entry command；此版本計畫不宣稱已有可執行 Ridge CLI。

## 7. 指標與統計分析

**主要指標：per-cell macro MAE**，group key 為 `cell_base`，先算每個 cell 的 MAE，再等權平均。模型仍以 point-weighted MSE 訓練及選參數，不能因報告 macro MAE 而改變其中一方訓練權重。

次要指標：micro MAE、micro / macro RMSE、P90RE、P99RE、R2、sMAPE；P50RE 可作補充。沿用 `src/metrics.py`，尤其保留近零分母規則。若 macro R2 排除 undefined groups，必須記錄數量，不以缺值改善平均。

逐 mode × seed × scope × metric 配對，禁止只比較兩組未配對平均：

- 誤差指標：`delta = Ridge - ANN`，正值代表 ANN 較好。
- 誤差改善率：`100 * (Ridge - ANN) / Ridge`，先逐 seed 計算，再報均值；Ridge 分母為 0 時記為 undefined，另報絕對差。
- R2：報 `ANN - Ridge`，不套用誤差百分比改善公式。
- P90RE / P99RE 的絕對差單位為 percentage points；不要與相對改善百分比混淆。

各 mode 分開以五個 seeds 報 mean、sample SD、配對 delta 的 95% Student-t CI（df=4）與五筆原始差值。不同 mode 不合成 n=10；數百萬 test rows 也不當作獨立重複。

主要呈現效應大小與不確定性。若提出正式顯著性結論，對兩個 mode 的 macro MAE 配對雙尾 t-test 做 Holm 校正（family alpha=0.05）；次要指標僅作描述性／探索性分析。n=5 且不同 seed 的資料有重疊，CI 與 p-value 的假設及限制須揭露，不能推廣為跨 library 的保證。

不事後設定最低改善百分比。ANN 較差、CI 跨 0、MAE 改善但 P99 惡化都照實報告。只有資料匹配且證據支持時才描述 ANN 優勢；CI 跨 0 表示此比較未明確區分，不代表等效。

## 8. 預期產物與驗收

獨立輸出根目錄：`experiments/global_ann_ridge_v1/`，不覆寫 ANN 結果。

| 產物 | 內容與驗收 |
|---|---|
| `input_audit.json` | Dataset / feature / split / ANN result hash、十個配對的資格狀態 |
| `evaluations/<run>/training_metadata.json` | n_fit/n_validation/n_test、normalization、候選 lambda/alpha、validation losses、selected lambda、程式與環境版本 |
| `evaluations/<run>/metrics.json` | 與 ANN 同定義的 micro / macro metrics，n 與 groups 相符 |
| `evaluations/<run>/per_cell_metrics.csv` | 逐 cell 誤差與樣本數，用於後續定位問題 |
| `models/<run>.npz` | 係數、截距與必要數值資訊；依既有忽略規則留本機 |
| `aggregate/paired_metrics.csv` | 每個 seed 的 ANN / Ridge / delta / relative improvement |
| `aggregate/summary_by_mode.csv` | 分 mode 的 mean / SD / CI，含有效配對數與來源核對狀態 |
| `figures/` | 配對 macro MAE 圖與 P90/P99 圖；展示五個 seeds，避免只畫平均柱狀圖 |
| `report.md` | RQ、方法、結果、差值方向、限制、負結果及後續 L1 的問題 |

小型規格、metadata、結果摘要可進 Git；全量資料、快取、模型與逐點預測不隨本計畫上傳。保存摘要時注意 repo 的 `runs/` 忽略規則，新結果目錄刻意使用 `evaluations/`。

執行完成條件是十個預定配對都具有可檢查結果或明確失敗狀態；完整比較摘要要求 10/10 成功，不能隱藏失敗 run。驗收以資料與方法正確為準，不以 ANN 勝出為條件。

## 9. 接續工作清單

- [x] 確認 RQ、比較範圍、固定 seeds、主要指標與時間比較邊界。
- [x] 保存本計畫及可機讀輸入鎖定清單。
- [ ] Commit / push 計畫至 `plan-v4`，並核對遠端 commit。
- [ ] 核對或補匯出 ANN checkpoint metadata，保留 pending 與 mismatch 狀態。
- [ ] 實作串流 Ridge、選參數、評估、配對統計與可續跑的單一入口。
- [ ] 完成小型數值／切分／統計測試與 preflight；提供正式命令。
- [ ] 首個完整 Ridge run、記憶體與筆數驗證。
- [ ] 其餘九個 runs、配對摘要、圖與報告。

接續時先讀本文件、`HANDOFF.md` 與 config；不要將「計畫已保存」或「小型測試通過」改寫為「Ridge 已訓練完成」。
