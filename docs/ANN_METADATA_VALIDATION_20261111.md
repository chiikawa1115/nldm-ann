# ANN metadata 接收與核對紀錄

日期：2026-11-11。Verification Status：**ANALYZED**；核對範圍為保存的 metadata、來源版本、指標與 history，一律未重訓或重跑 ANN 推論。

## 資料與結果

- 資料提交：`17a3bdc`，分支 `plan-v4`。收到預定十份 metadata JSON、十份 history、十份 per-cell CSV 及 `EXPORT_NOTES.md`，沒有缺少或多出的 metadata run ID。
- 原訓練 checkpoint 記錄版本：`0032cdfeb1457bda850ff5752d1f1bf00ad3dd62`；六個凍結來源檔的 Git blob SHA-256 在本機另行計算，與 metadata 及凍結設定一致。
- 執行 `python src/run_ridge_comparison.py aggregate --ann-metadata-dir experiments/global_ann_ridge_metadata` 成功，10/10 complete，10/10 matched，沒有 pending/mismatch。
- Dataset、manifest（容許 LF/CRLF 差異）、42 維 feature 順序、`max_rows=null`、fit 筆數及 normalization 均相符。Normalization 各欄位跨十組的最大絕對差：x_mean `2.13e-14`、x_std `1.42e-14`、y_mean `1.85e-13`、y_std `5.68e-14`，符合原容差 `rtol=1e-6, atol=1e-8`；n_fit 完全相等。
- 全部 Ridge 保存產物通過既定 signature/checksum。與提交 `17a3bdc` 所含的原配對表相比，140 列指標所有數值完全相同，只有來源資格更新。
- 預定兩項 primary macro MAE 配對 t 與 Holm 結果已輸出，另以 SciPy `ttest_rel` 與兩項 Holm 公式核對一致。校正 p：random-point `5.1573777976e-9`、unseen-cell `1.0562636918e-5`。獨立性與分布假設未因此獲得驗證。

## 補充資料一致性與逐 cell 結果

十份 ANN per-cell CSV 的 cell ID／各 cell 測試筆數與 Ridge 一致。七項 macro 指標皆能還原已發布 ANN metrics；micro 的 n、MAE、RMSE、sMAPE 也一致。這些檢查不會從 per-cell percentile 還原 micro percentile，也沒有從 per-cell R² 平均值推算 micro R²。

| Mode | 每 seed 測試 cells | ANN MAE 較低的 cell/run 觀測 | 例外 |
|---|---:|---:|---|
| random_point | 200 | 1,000 / 1,000 | 無 |
| unseen_cell | 40 | 197 / 200 | 三筆均來自 seed 23 |

這是已測試 cell/run 的描述性計數，包含跨 seed 重複的 cell，不是 1,000 或 200 個獨立樣本，也不保證未來 cell 都有相同結果。

| Unseen-cell seed 23 的例外 | ANN MAE（ps） | Ridge MAE（ps） | ANN − Ridge（ps） |
|---|---:|---:|---:|
| BUFx4_ASAP7_75t | 88.2939 | 78.7171 | 9.5769 |
| DFFLQNx3_ASAP7_75t | 58.1881 | 54.3772 | 3.8110 |
| DFFLQNx2_ASAP7_75t | 57.4064 | 55.5064 | 1.9001 |

## 訓練歷史

十組 history 的 epoch 編號連續，fit/validation loss 均有限，長度與 metadata 的 completed_epochs 一致，最低 validation loss 與 checkpoint 記錄一致。

| Mode | Seed | 完成 epochs | 最佳 validation epoch | 停止情況 |
|---|---:|---:|---:|---|
| random_point | 11 | 34 | 26 | 符合 patience=8 |
| random_point | 23 | 50 | 46 | 到達 50 epochs 上限 |
| random_point | 37 | 50 | 43 | 到達 50 epochs 上限 |
| random_point | 53 | 47 | 39 | 符合 patience=8 |
| random_point | 71 | 42 | 34 | 符合 patience=8 |
| unseen_cell | 11 | 38 | 30 | 符合 patience=8 |
| unseen_cell | 23 | 12 | 4 | 符合 patience=8 |
| unseen_cell | 37 | 24 | 16 | 符合 patience=8 |
| unseen_cell | 53 | 31 | 23 | 符合 patience=8 |
| unseen_cell | 71 | 17 | 9 | 符合 patience=8 |

這些結果支持紀錄內部一致；不能單憑較早停止推論收斂充分、欠擬合，或解釋 seed 23 的較高測試誤差。其 checkpoint 使用最佳 validation epoch 的權重，並非最後 epoch 的權重，與核對過的訓練程式一致。

## 統計與方法檢查

學術研究 skill 的 11 項統計誤用類型均已檢視（11/11）；「不適用」及現有資料無法排除的限制如下，並非宣告所有風險已消失。

| 項目 | 本次判讀 |
|---|---|
| Simpson's paradox | micro/macro 的 MAE 方向一致；逐 cell 發現三筆例外並保留。沒有 point-level delay/transition 分層資料，本次未完整排除更細分組反轉。 |
| Ecological fallacy | 平均改善不推論每個 cell 都改善；上述三筆例外明列。 |
| Berkson's paradox | 僅涵蓋既定 ASAP7 eligible cells，不外推所有 library；未以模型好壞另行選 cell。 |
| Collider bias | 本次為固定資料與指標比較，未新增因結果而定的條件篩選或共變數調整。 |
| Base rate neglect | 非分類診斷；保留每 cell 樣本數，區分 micro 與 macro。 |
| Regression to the mean | 配對清單事前固定；不把事後選出的 seed 23 當作改善策略已驗證的證據。 |
| Survivorship bias | 預定十組全數納入，沒有失敗或較差 seed 被排除。 |
| Look-elsewhere effect | 只對預定兩項 primary macro MAE 做 Holm；其他指標與逐 cell/history 分析為描述性。 |
| Garden of forking paths | 凍結 config、來源與指標不變；未因新 metadata 調整 normalization 容差、lambda grid 或測試資料。 |
| Correlation/causation | 比較的是既定模型流程的預測誤差，不能把差值歸因於單一架構因素，也未證明 L1 效果。 |
| Reverse causality | 非觀察性因果估計；不從訓練 epoch 與測試誤差的關聯推導因果機制。 |

每 mode 僅五個 seed，切分來自相同 library 且重疊。配對 t 的差值獨立性／分布假設尚未建立，p 值與 t 區間須附帶此限制；配對檢定不以兩模型的獨立樣本等變異假設為前提。來源 matched 解決比較來源資格，未解決跨 library 泛化或 timing sign-off 可用性。ANN 尾端相對誤差仍高，需與平均誤差一起閱讀。

## 可追溯性與限制

- [機讀核對紀錄](../experiments/global_ann_ridge_v1/aggregate/ann_metadata_audit.json) 保存逐 run 資格、normalization 差值、補充資料 LF-normalized hashes、訓練歷史摘要與統計核對。
- [最新比較狀態](../experiments/global_ann_ridge_v1/aggregate/comparison_status.json)、[配對表](../experiments/global_ann_ridge_v1/aggregate/paired_metrics.csv)、[中文結果報告](RESULTS_GLOBAL_ANN_RIDGE_COMPARISON.md)。
- 原 training metadata/status 是訓練當時的紀錄，保留 pending；`input_audit.json` 是尚未比較 fit normalization 的聚合前檢查。最終資格以 `aggregate/` 下的狀態及本次核對紀錄為準。
- 原主機的 `EXPORT_NOTES.md` 與所有交付資料均未修改。本機沒有原 ANN 權重，無法獨立重算 checkpoint hash；匯出內容與 Git commit 也不能證明原訓練時沒有未提交修改。
- 本次完成的是保存資料的一致性核對（ANALYZED），並非重新執行 ANN 的完整重現（VERIFIED）。其餘 30 組 ANN 實驗不在本次十組 metadata 核對範圍。

本輪交接沒有待補欄位或 mismatch。可直接使用本次摘要向老師說明比較結果；seed 23 與尾端誤差可作後續分析題目，尚未啟動新的訓練。
