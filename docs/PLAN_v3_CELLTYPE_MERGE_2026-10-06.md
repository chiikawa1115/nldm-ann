# 研究計畫 v3：同 Cell Type 合併模型（Per-Cell-Type Merge）

日期：2026-10-06。狀態：草案，待 Rudy 確認後鎖定。
本文件不覆寫 `RESEARCH_CONTRACT.md` 與 `CROSS_CELL_VT_RESEARCH_PLAN.md`；既有 manifests 保留。

## Material Passport

- Artifact type: experiment plan (ARS experiment-agent `plan` mode, code path)
- Inputs: 老師 2026-10-06 口頭反饋、LIBMixer (ASP-DAC 2025) 精讀、既有 repo 狀態
- Verification status: 資料統計數字已於 2026-10-06 由 parquet 實際查詢；其餘為規劃，未執行

## 1. 老師反饋與定位

- 反饋：LIBMixer 已很完整，需要有「不同的突出點」（尚未具體指定）。
- 近期指示：先改進泛化，將同種 cell type 合併成一個 model，減少模型數量；
  現況為一張 table 一個 model。
- 與 LIBMixer 的邊界（依原文 Table 2/3）：LIBMixer 的 train/test 為同一批 198 cells，
  且需要 reference library 作輸入；test PVT 的各維度值皆出現在訓練集，屬組合內插。
  其未檢驗 unseen size、unseen VT、unseen cell，也未報告 table 物理一致性。

## 2. 資料事實（ASAP7，2026-10-06 查詢）

- 800 cells = 200 `cell_base` × 4 VT（LVT/RVT/SLVT/SRAM）；每 cell 皆有 FF/SS/TT。
- 移除 drive-strength 後綴後得 **101 個 function type**。
- Size 數分布：55 types 只有 1 個 size、22 types 有 2 個、24 types 有 ≥3 個
  （INV 11、CKINVDC 10、BUF 8、NAND2 6、NOR2/ICG 5、SDFH/SDFL 4 ...）。
- 每 cell 每 corner 平均約 39 張 table（含 delay 與 transition；delay-only 數量待確認）。

含意：合併後每個 type model 必然跨 VT × corner × arc × rise/fall；跨 size 只在 46 個
types 可檢驗，leave-one-size-out 只在 24 個 ≥3 size 的 types 有內插意義。

## 3. 研究問題

- RQ1：將同 function type 的所有 size/VT/corner/arc 合併成單一模型，相對於
  per-table 模型與 bilinear LUT，精度損失多少？模型數與參數量減少多少？
- RQ2：合併模型能否預測該 type 內未見的 VT、未見的 size、未見的 corner？
- RQ3：合併粒度（per-table → per-type → per-family → global）與精度、負遷移的取捨為何？

## 4. 模型設計（L1 per-type model）

- 輸入：log_slew、log_load、drive_strength、VT（one-hot 與 ordinal 兩版做消融）、
  process proxy / voltage / temperature、arc 描述（related pin 序號、rise/fall、timing_sense）。
- 輸出：cell delay（主）；output transition 不納入，除非 OpenSTA 驗證需要。
- Target transform：比較 raw z-score 與 log/比例形式；負值（4,280 點）處理規則先鎖定再訓練。
- 架構：小型 MLP（例如 2–3 層 × 32–64），參數量必須報告。

## 5. 實驗矩陣

| ID | Split | 目的 | 比較對象 |
|---|---|---|---|
| E1 | 同表 random-point（0.2/0.5 holdout） | 合併是否損失精度 | L0 per-table MLP、bilinear LUT、Ridge、Ridge+ |
| E2 | Leave-one-VT-out（輪替 4 VT） | 跨 VT 泛化 | Ridge/Ridge+、VT 比例縮放 `k_vt × delay_RVT` |
| E3 | Leave-one-size-out（≥3 size types，分內插/外推） | 跨 size 泛化 | Ridge/Ridge+、drive-strength 縮放 |
| E4 | Leave-one-corner-out | 跨 corner 泛化 | Ridge/Ridge+、corner 比例縮放 |
| E5 | L0/L1/L2/L3 粒度掃描（E1/E2 設定） | 模型數 vs 精度曲線、負遷移 | 同上 |

- Seeds：11、23、37、53、71；指標沿用 contract（MAE、RMSE、P90/P99 rel. error、R²，
  micro + per-cell macro，mean ± SD / 95% CI）。
- 完全未見的 VT/size 不存在可插值的 LUT，不得虛構 LUT baseline。
- 比例縮放 baseline 為必做：若 ANN 未勝過，照實報告。

預先鎖定的成功門檻（訓練前確認，事後不改）：

- E1：L1 的 MAE 不超過 L0 的 1.10 倍（或明確報告差距）。
- E2/E3：L1 優於最佳簡單縮放 baseline，且 paired seed difference 的 95% CI 不跨 0。

## 6. 亮點候選（與 LIBMixer 區隔，待老師選定）

| 代號 | 亮點 | 量化指標 | 可行性 |
|---|---|---|---|
| H1 | Model-as-library：101 個小模型取代全部 LUT | 參數總量 vs LUT 浮點數總量（壓縮比）、推論時間 | 高；E5 順帶產出 |
| H2 | EDA 適配：單調性保證（slew/load ↑ → delay ↑）+ 寫回 .lib + OpenSTA | 單調性違反率、lib sanity 通過率、path slack 誤差、違例數 | 中高 |
| H3 | Sign-off 安全：尾端誤差與低估比例，可選非對稱 loss | P99/max error、under-prediction ratio | 高 |
| H4 | 跨 VT 少量資料校準（few-shot） | 達指定誤差所需 label 數 | 中；只能稱 label-budget reduction |

建議主軸：RQ1–RQ3 + H1（呼應減少模型數量）+ H2（EDA 工具適配度）。

## 7. 里程碑

| 週次 | 里程碑 | 驗收 artifact |
|---|---|---|
| W1（10/06–10/12） | M-a：cell-type 對照表、資料盤點、sequential 是否納入的決策 | `cell_type_catalog.csv` + 決策紀錄 |
| W2–W3 | M-b：L1 trainer + E1 smoke（少數 types），L0/LUT/Ridge/Ridge+ baseline | 單一 type 的 E1 表格 |
| W4–W5 | M-c：E1–E4 全部 types × 5 seeds | aggregate CSV + CI |
| W6 | M-d：E5 粒度曲線（L2 新增；L3 沿用 global engine） | 模型數-精度圖 |
| W7–W8 | M-e：亮點 pilot（H1 壓縮比 + H2 單調性/OpenSTA 一個 design） | 單調性報告 + slack 比較表 |
| 之後 | 寫作、GF180 外部驗證（選配） | 章節草稿 |

時程以每週 8–10 小時估計，為規劃值，需在 M-b smoke 後依實測 runtime 修正。

## 8. 待決事項

1. Sequential cells（DFF/latch/ICG/SDF*）是否納入 L1 第一輪（LIBMixer 強調其重要性）。
2. 老師對亮點 H1–H4 的選擇。
3. 論文繳交期限，用以決定 GF180 外部驗證是否保留。
4. 執行環境：GPU 主機是否可用（per-type 小模型可能 CPU 即可）。

---

## 9. 2026-10-17 更新：Rudy 決策與物理一致性約束（亮點 A 擴充）

### 9.1 決策

- 合併策略：可跨 size 合併的 types 用 L1 per-type model；無法合併者（單一 size 的 55 types）
  改由 L2 family model 處理（待 Rudy 確認此解讀）。
- 亮點 A（EDA 適配度）採用，擴充為「物理一致性約束」：除單調性外，加入 rise/fall、
  transition 合理範圍、cell 間關係等規則。

### 9.2 規則先以 golden ASAP7 驗證（2026-10-17 實測）

原則：golden 幾乎 100% 遵守的才可做硬約束；否則只能當 soft penalty 或評估指標，
避免把物理上不成立的規則強加給模型。

| 規則 | Golden 遵守率 | 建議用法 |
|---|---|---|
| delay 對 load 單調遞增 | 100% | 硬約束（monotonic 架構） |
| transition 對 load 單調遞增 | ≈99.999% | 硬約束 |
| delay 對 slew 單調遞增 | ≈98.6%（違反 1.3–1.5%） | soft penalty；違反點需分析（小 slew 區） |
| transition 對 slew 單調遞增 | ≈89%（違反 10–11%） | 不約束，僅報告 |
| Corner：FF ≤ TT ≤ SS | 99.93% | 硬約束或強 penalty |
| VT：SLVT ≤ LVT ≤ RVT ≤ SRAM（TT） | 99.56% | 強 penalty |
| 同 type 大 size 的 delay 不大於小 size（同 slew/load） | 78.9% | 不成立為通則；需改寫成負載相關條件 |
| 同 type 大 size 的 input pin cap 不小於小 size | 70.1% | 不成立為通則；需檢查 pin 對應 |
| 同 type+pin 的 rise/fall 大小方向一致 | 中位數 83%，P10 55% | 不可硬約束；改為報告預測是否保留 golden 的 rise/fall 大小關係 |

Transition 合理範圍：ASAP7 `max_transition = 320 ps`，但 transition table 值最大可達
約 6,300 ps（P99 約 630–800 ps）。Table 本來就會延伸到 max_transition 以外。
「依 clock 頻率決定合理 transition」屬於 design constraint，不是 library 性質，建議用法：

- 定義有效操作區（slew ≤ max_transition、load ≤ max_capacitance），作為評估分層與 loss 權重；
- 依 clock period 的 transition 上限放在 OpenSTA 階段，檢查 max_transition 違例數是否與 golden 一致。

Cell 間關係（如 multi-bit reg 的 load 大於單 bit reg）：ASAP7 沒有 multi-bit flop cell，
無法直接驗證。可用的替代關係為 size、VT、corner 排序（見上表）；MBR 規則留待有 MBR 的
library（如商用製程）再做。

### 9.3 新增評估指標

- Constraint violation rate：依 9.2 表逐條報告（預測 vs golden）。
- Liberty sanity：寫回 .lib 後可被 OpenSTA 讀入、無負 delay、無非單調 load 軸。
- OpenSTA：path slack 誤差、max_transition 違例數與 golden 的一致性。
