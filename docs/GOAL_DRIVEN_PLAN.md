# NLDM-ANN Goal-Driven Implementation Plan

## Goal

完成一套論文等級、可重現且無資料洩漏的 feature-conditioned NLDM + ANN
實作，量化模型在 unseen cell、unseen PVT corner，以及兩者同時 unseen 時的
泛化能力與 STA 影響。ASAP7 為主資料集，GF180MCU 為 external validation，
IHP SG13G2 為選配。Nangate45 sparse-grid 結果只作 pilot，不作主 RQ 證據。

## Definition of Done

只有下列證據全部存在，才算達成最終目標：

- 資料來源、單位、解析規則與資料版本可追溯，split manifest 可重現每個實驗。
- ANN 與規則網格 bilinear LUT 使用完全相同的 train/test points；不得以 nearest fallback 混入主要結果。
- 至少以 5 個固定 seeds 報告 MAE、RMSE、sMAPE、P50/P90/P99 relative error 與 95% confidence interval。
- cell 輸入使用可泛化結構特徵，不以 cell ID embedding 支撐 unseen-cell claim。
- PVT 使用 process、voltage、temperature 數值，不以 corner ID 支撐 unseen-corner claim。
- 完成 random-point、unseen-cell、unseen-corner、joint-unseen 四組 split；
  同時報告 micro-average 與 per-cell macro-average。
- 完成 cell/PVT feature、模型架構與資料量消融。
- 至少完成一個 OpenSTA end-to-end 驗證，報告 path delay/slack 誤差與 runtime/model-size trade-off。
- 所有論文表格與圖片能從 versioned config 和原始結果一鍵重建。
- 核心 parser、split、normalization、baseline、metrics 具自動測試，CI 全數通過。

## Milestones and Gates

### M0 — Baseline audit

Gate：找出並測試 split leakage、單位、插值定義、極小真值下的 metric 行為；建立 immutable Experiment A 重現紀錄。

### M1 — Reproducible experiment harness

Gate：單一 CLI/config 能產生 split manifest、checkpoint、逐點預測、摘要、環境與 git provenance；相同 seed 的結果可重現。

### M2 — Fair sparse-grid pilot (complete, not the main RQ)

Gate：規則子網格 bilinear baseline 與 ANN 同資料比較；5 seeds 與信賴區間完成，不以單次 MAPE 宣稱勝負。

### M3 — Feature-conditioned generalization (main RQ)

Gate：ASAP7 上以至少 3 seeds 完成 random-point、unseen-cell、unseen-corner、
joint-unseen；split manifest 證明 group disjointness；cell/PVT 特徵皆可在 inference
時由 held-out sample 取得；報告 MAE、RMSE、sMAPE、P90/P99、R2 的 micro 與
per-cell macro mean ± SD/CI。

### M3b — External technology validation

Gate：在不以 GF180 cell ID fine-tune 的條件下完成 external validation；清楚揭露
ASAP7/GF180 feature schema、unit 與 library coverage 差異。IHP 僅在時間允許時加入。

### M4 — STA validation

Gate：將預測 timing tables 接入一個 OpenSTA design，與原始 Liberty 比較 path delay/slack、runtime 與模型大小。

### M5 — Paper artifact

Gate：fresh environment 可依 README 重跑主要實驗與圖表；結果、限制與 statistical claims 一致，測試全過。

## Operating Rule

每次 implementation 先指定所屬 milestone、可執行驗收命令與預期 artifact。完成後由驗收證據判定 gate；未通過就繼續修正，不因程式已寫完或單次 run 成功而關閉 milestone。若研究假設改變，先更新 Goal/Definition of Done，再改 implementation。
