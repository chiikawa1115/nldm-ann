# 研究計畫 v4：階層式模型合併 + Sign-off 安全的 NLDM Delay 預測

日期：2026-10-17。修訂：2026-10-18（v4.1 納入獨立審查 14 點；v4.2 納入 W1 結果與 Rudy 決策，見第 12 節）。
狀態：**LOCKED v4.2**（Rudy 2026-10-18 確認流程與 W1 決策）。時程不在本次鎖定範圍內。
取代 `PLAN_v3_CELLTYPE_MERGE_2026-10-06.md` 第 1–8 節的規劃內容；v3 第 9 節的約束實測保留為依據。
不覆寫 `RESEARCH_CONTRACT.md`、`CROSS_CELL_VT_RESEARCH_PLAN.md` 與既有 28 份 manifests。

## Material Passport

- Artifact type: experiment plan（ARS experiment-agent `plan` mode, code path）
- Inputs: 老師 2026-10-06 反饋、LIBMixer（ASP-DAC 2025, DOI 10.1145/3658617.3697602）全文精讀、
  ASAP7 parquet 實際查詢（2026-10-06、10-17）、Rudy 2026-10-17 決策
- Verification status: 第 2 節與第 6.1 節數字由 parquet 查詢取得（VERIFIED-DATA）；
  其餘為規劃（UNEXECUTED）。不得把本文件任何數字當作模型結果引用。

## 1. 背景與定位

### 1.1 老師反饋（2026-10-06）

- LIBMixer 已很完整，需要「不同的突出點」。
- 近期：改進泛化，將同種 cell 合併成一個 model，減少模型數量（現況一張 table 一個 model）。

### 1.2 既有負結果

Nangate45 Experiment A v2：per-table 64×64 MLP 在所有測試 sparsity 下劣於 regular-grid
bilinear LUT（5 seeds）。這是合併的動機：單張 7×7 table 資料太少，ANN 無優勢；合併讓不同
size/VT/corner 共享資料。此負結果在論文中照實報告。

### 1.3 與 LIBMixer 的區隔（依原文 Table 2、3、5、7）

| 面向 | LIBMixer | 本研究 |
|---|---|---|
| 輸入 | 目標 PVT + 同一 cell 的 reference library | cell 結構特徵 + PVT/VT，無 reference table（主線） |
| 泛化測試 | 同一批 198 cells；test PVT 各維度值皆在訓練集出現（組合內插） | 跨 VT、跨 size、跨 corner，group-disjoint split |
| 模型數 | 單一模型，未分析合併粒度 | L0→L1→L2 粒度與精度取捨、負遷移分析 |
| 輸出 | 單一點預測 | nominal + early/late 區間（亮點 B） |
| 物理一致性 | 未檢查 | 以 golden 驗證的約束 + violation rate（亮點 A） |
| 統計 | 單一 split，無 CI | 5 seeds，mean ± SD / 95% CI，paired difference |
| Baseline | Ridge、Ridge+ | LUT、Ridge、Ridge+、比例縮放、L0 per-table |

LIBMixer 的 timing MAPE 0.11% 來自自建 BSIM-CMG/SiliconSmart library 與不同任務，
不與本研究數字直接比較。

## 2. 資料範圍（ASAP7，VERIFIED-DATA）

- 主線：combinational 4 個 family，排除 SEQ（Rudy 2026-10-17 決定）。
- 目標：cell delay（`cell_rise`、`cell_fall`）。Output transition 不作為主要目標；
  僅在亮點 A/OpenSTA 驗證需要時另行處理（見第 9 節風險）。

| Family | cell_base | function types | 多 size types | delay tables | delay rows | 負值 |
|---|---|---|---|---|---|---|
| SIMPLE | 54 | 22 | 17 | 5,140 | 251,860 | 435 |
| AO | 42 | 28 | 12 | 60,936 | 2,985,864 | 1,755 |
| OA | 34 | 25 | 7 | 55,056 | 2,697,744 | 2,075 |
| INVBUF | 37 | 11 | 3 | 888 | 43,512 | 15 |
| **小計（主線）** | **167** | **86** | **39** | **122,020** | **5,978,980** | **4,280** |
| SEQ（排除） | 33 | 15 | 7 | 1,176 | 57,624 | 0 |

- 每個 cell_base 皆有 4 VT（LVT/RVT/SLVT/SRAM）× 3 corner（FF/SS/TT）。
- Family 對應沿用 `experiments/asap7_generalization_v1/manifests/eligible_cell_catalog.csv`。
- Function type = `cell_base` 去除 `_ASAP7_75t` 與 drive-strength 後綴（如 `x1`、`xp33`）；
  此 regex 實際得到的是 topology type：`BUFx4f/6f/12f/16f`、`HB1–4` 等會各自成為單一 size 的
  type。W1 產出對照表時人工決定是否合併，並記錄決策。
- **W1 決策（Rudy 2026-10-18）**：`BUFx4f/6f/12f/16f` 各自為獨立 type（不併入 BUF，
  同 drive pin cap 約 2 倍、delay 較低）；`HB1–HB4` 各自獨立（編號為 delay 等級而非 drive）；
  `CKINVDC` 自成一個多 size type（不併入 INV）。結果：主線 86 types（39 多 size、47 單 size），
  對照表 `experiments/plan_v4/w1/cell_type_catalog.csv`，10 筆抽查經 Rudy 確認無誤。
- **資料不平衡**：AO/OA 佔主線 delay rows 95.06%，主因是 state-dependent（`when`）tables
  （例如 AO333 每個 cell/corner 有 630 張 delay tables＝315 arcs × rise/fall，其中 612 張帶 `when`）。
  INVBUF 僅 888 張 table。
  L2 訓練與評估必須處理此不平衡（見 4.3、6.2）。

## 3. 研究問題與假設

- **RQ1（合併）**：將 delay tables 從 per-table（L0）合併為 per-function-type（L1）、
  per-family（L2）時，精度變化與模型數/參數量的取捨為何？何處出現負遷移？
- **RQ2（泛化）**：合併模型對 held-out VT、held-out size、held-out corner 的預測能力為何？
- **RQ3（亮點 A）**：在不明顯犧牲精度的前提下，能否讓預測 table 滿足經 golden 驗證的物理約束？
- **RQ4（亮點 B）**：能否輸出 early/late 區間，在指定 coverage 下讓 setup 與 hold 檢查皆不漏報，
  其悲觀代價（區間寬度、誤報）為何？

待檢驗假設（不預設成立）：

- H1：L1 相對 L0 在 random-point 設定下誤差不顯著變差，且在 held-out VT/size 上優於簡單縮放 baseline。
- H2：L2 相對 L1 的精度損失集中於少數 types（可識別的負遷移來源）。
- H3：硬約束（load 單調、corner 排序）使 violation rate 降為 0，nominal MAE 增加不超過 5%。
- H4：conformal 校準後的區間在 test 上達到名目 coverage（±1%）。

## 4. 模型階層

### 4.1 層級定義

| 層級 | 共享範圍 | 模型數（主線） | 角色 |
|---|---|---|---|
| L0 | 單一 delay table | 122,020（訓練時分層抽樣約 2,000 張） | 基準；僅在 random-point 有定義 |
| L1 | 單一 function type（跨 size × VT × corner × arc） | 86 | 主實驗 |
| L2 | 單一 family | 4 | **主要目標** |
| L3 | 全部 combinational | 1 | 選做；沿用既有 global engine 作對照 |

### 4.2 合併 gate（訓練前鎖定，事後不改）

由 L(k) 往 L(k+1) 前，以 E1 random-point 與 E2 leave-one-VT-out 的 5-seed 結果判定：

- **PASS**：non-inferiority 檢定——L(k+1) 與 L(k) 的 macro MAE 比值，其 paired 95% CI 上界
  ≤ 1.10。E1 以 seed 為配對單位；E2 以（seed × held-out VT fold）為配對單位。
- **FAIL**：停在 L(k)，並以 per-type error 分解找出負遷移來源，作為分析結果報告。
- L0→L1 的 gate 只用 E1（L0 無法做 held-out VT），且只在 L0 的抽樣 tables 上比較
  （抽樣依 family × type 分層，固定 seed，清單寫入 manifest）。

Gate 失敗不代表研究失敗；「哪些 cell 不宜合併」是 RQ1 的答案之一。

### 4.3 特徵（inference-available）

- 連續：log_slew、log_load、process proxy（SS=-1/TT=0/FF=+1）、voltage、temperature、
  drive strength、related pin capacitance。
- 類別：rise/fall、timing_sense、related pin 角色。
- **VT 編碼**：E1 可用 one-hot；**E2 主線必須用 ordinal（SLVT=0 < LVT=1 < RVT=2 < SRAM=3）
  或連續 VT 描述子**。One-hot 在 held-out VT 的那一欄於訓練中恆為 0，權重未被訓練，
  不構成跨 VT 資訊；one-hot 於 E2 僅作負對照消融。
- **`when`、related_pin、timing_sense（新增，必要）**：現有 `src/features.py`
  （`feature_columns()`）三者皆未編碼。審查查詢顯示主線 5,707,315 / 5,978,980 rows（95.5%）
  在缺少這些欄位時「特徵相同但 target 不同」。L1/L2 必須：
  - 將 `when` 的 side-input 狀態編碼為固定長度特徵（每個 side input 的 0/1/don't-care）；
  - 將 related_pin 轉為跨 cell 一致的 canonical role（例如依 Boolean 結構中的位置／階層），
    不得用 pin 名稱字串或 ID embedding；
  - 加入 timing_sense one-hot。
  W1 以 script 重現 95.5% 衝突比例並寫入報告，作為此需求的證據。
  **W1 結果**：5,707,315 / 5,978,980（95.46%）重現；只加 `when` 剩 196 rows 衝突，
  `when`+`related_pin` 降為 0；`timing_sense` 單獨加入幾乎無效，但仍依物理意義保留。
  （`experiments/plan_v4/w1/conflict_report.json`）
- L2 另加 Boolean 結構特徵（沿用 `boolean_function_features`）與 function type 描述
  （輸入數、AND/OR 階層），**不使用 cell ID 或 type ID embedding**。
- **Pin capacitance 洩漏規則**：同一 (cell_base, pin) 的 pin cap 在 4 VT 間 100% 不同、
  在 3 corner 間也 100% 不同（2026-10-18 查詢）。因此：
  - E1、E3：可使用該 cell 本身的 pin cap（揭露其來自 characterization）。
  - E2：主線使用 reference VT（RVT；held-out 為 RVT 時用 LVT）的 pin cap，並做「不含 pin cap」消融；
    禁止使用 held-out VT 的 pin cap。
  - E4：主線使用 TT 的 pin cap（held-out 為 TT 時不用 pin cap），同樣做消融。

### 4.4 Target 與 loss

- 比較 raw z-score 與 log 形式；負值（4,280 rows）的處理規則在訓練前鎖定
  （候選：signed log、對 table 最小值平移、或改學相對 reference 比例），並報告受影響 rows。
- 不平衡處理：L2 以 per-table 或 per-type 等權重抽樣／加權，避免 AO/OA 主導；
  消融比較「無加權」與「type 等權」。
- 架構：小型 MLP（2–3 層 × 32–128），所有模型報告參數量。

## 5. 實驗矩陣

Seeds：11、23、37、53、71。指標：MAE、RMSE、P50/P90/P99 relative error、R²；
micro 與 per-cell macro；mean ± SD 與 95% CI；模型比較用 paired seed difference。
近零 delay 的 relative error 分母規則沿用既有 `metrics.py` 並揭露。

| ID | Split | 用途 | 適用層級 | Baseline |
|---|---|---|---|---|
| E1 | 同表 random-point（holdout 0.2、0.5） | 合併是否損失精度 | L0、L1、L2 | bilinear LUT、Ridge、Ridge+ |
| E2 | Leave-one-VT-out（輪替 4 VT） | 跨 VT | L1、L2 | Ridge、Ridge+、VT 比例縮放 `k_vt × delay_RVT` |
| E3 | Leave-one-size-out（39 個多 size types；內插約 42 folds、外插約 78 folds） | 跨 size | L1、L2 | Ridge、Ridge+、drive-strength 縮放 |
| E4（探索性） | Leave-one-corner-out；主張僅限 leave-TT-out 內插 | 跨 corner | L1、L2 | Ridge、Ridge+、corner 比例縮放 |
| E5 | 合併粒度彙整 | 模型數／參數量 vs 精度 | L0–L2（L3 選做） | — |

規則：

- 所有 split 在 table/group 層建立後才展開到點；同一 group 的所有點同屬一個 partition。
- Normalization、early stopping、超參數、conformal calibration 皆不使用 test labels。
- Held-out VT/size/corner 沒有 target table，不存在可插值的 LUT；不得虛構 LUT baseline。
- 比例縮放 baseline 為必做；ANN 未勝出時照實報告。
- E2 的「VT 比例縮放」使用同 cell 同 corner 的 RVT table，屬於有 reference 的 baseline；
  held-out VT 為 RVT 時改用 LVT 為 reference，並標註。
- **Baseline 分兩類，分開報告**：
  - 無 reference：Ridge、Ridge-log（Ridge + log normalization）、ANN 主線。
  - 有 reference：比例縮放、Ridge+（依 LIBMixer p.3 Fig.2、p.4 §4.1.3：log normalization +
    對 reference 值的 residual，即預測 = reference + 修正量）。Reference 定義同 E2/E4 的 pin cap 規則
    （E2 用 RVT／LVT，E4 用 TT）；E1 的 reference 為同表訓練點的 bilinear 結果。
  ANN 主線與「有 reference」baseline 比較時，須標註資訊量不對等。
- **E3 樣本限制**：22 個 types 只有 2 種 size（只能外插）、12 個有 3 種、其餘 5 個各 5/6/8/10/11 種。
  報告 per-type 結果，不對少樣本 type 下普遍結論。
- **E4 限制**：三個 corner 的 P/V/T 完全共線（FF=0.77 V/0 °C、TT=0.70 V/25 °C、SS=0.63 V/100 °C），
  leave-FF/SS-out 為兩點外插。論文明寫「本資料無法區分 P、V、T 各維度的效果，也無法驗證
  PVT 維度泛化」；E4 只作探索性結果，不納入 gate 判定。

## 6. 亮點 A：物理一致性（EDA 適配度）

### 6.1 規則與 golden 遵守率（VERIFIED-DATA，2026-10-17，全 library 含 transition）

| 規則 | Golden 遵守率 | 用法 |
|---|---|---|
| delay 對 load 單調遞增 | 100%（主線 delay 重算：0 違反） | 硬約束 |
| delay 對 slew 單調遞增 | 主線 delay 98.36% | soft penalty；違反集中於大 slew 端（見下） |
| Corner：FF ≤ TT ≤ SS（delay） | 主線 delay 99.9993% | 硬約束（對 corner ordinal 單調，非對 process_code 單獨單調） |
| VT：SLVT ≤ LVT ≤ RVT ≤ SRAM（TT delay） | 99.56% | 強 penalty |
| 同 type 大 size delay 不大於小 size | 主線 delay 79.26%（相鄰 size） | 不約束 |
| 同 type 大 size pin cap 不小於小 size | 主線 70.67%（相鄰 size） | 不約束 |
| 同 type+pin rise/fall 大小方向一致 | 主線中位數 84.2%、P10 55.7% | 不約束；評估是否保留 golden 方向 |

約束原則：只有 golden 遵守率 ≥ 99.9% 的規則可做硬約束；其餘為 soft penalty 或只報告。
全部七列已由 W1 script（`scripts/plan_v4_w1.py`，`src/constraint_check.py`）以主線 delay 重算並鎖定
（VERIFIED-DATA，2026-10-18）；分母定義寫在輸出的 `unit` 欄位，結果見 `experiments/plan_v4/w1/`。

**Slew 違反分布（W1 新發現，修正 v3 第 9.2 節「小 slew 區」的描述）**：違反集中於大 slew 端。
相鄰 slew 段（ps）的違反數：5→10：1,657；10→20：1,006；20→40：1,704；40→80：7,123；
80→160：22,880；160→320：49,852（佔 59%）。整張 table 完全單調的比例為 89.9%。
Slew soft penalty 的設計與「有效操作區」分層評估須依此分布處理，不得假設違反只在小 slew。

### 6.2 實作

- Load 單調：對 log_load 輸入採 monotonic 結構（非負權重路徑或 lattice/輸入凸性設計），
  或 penalty 版本作消融。
- Corner 排序：以 corner ordinal（FF=0 < TT=1 < SS=2）作為單調輸入，與 V/T 數值綁定為一組，
  不對 process_code、V、T 個別施加單調。
- VT 排序：pairwise ranking penalty（同 cell/arc/slew/load 的 VT 配對）。
- 指標：每條規則的 violation rate（預測 vs golden）、rise/fall 方向保留率、
  有效操作區（slew ≤ max_transition 320 ps、load ≤ max_capacitance）內外分層誤差。

### 6.3 EDA 驗證

- 將預測 delay 寫回 .lib（transition 用 golden，明確標註），OpenSTA 讀入無錯誤。
- 至少一個 OpenROAD ASAP7 design（候選 ibex 或 aes）：path slack 誤差、violating endpoints 數、
  與 golden 的一致性；格式參考 LIBMixer Table 7。
- 依 clock period 的 transition 上限屬 design constraint，只在 OpenSTA 階段檢查 max_transition 違例。
- MBR 與單 bit reg 的關係：ASAP7 無 multi-bit flop，列 future work。

## 7. 亮點 B：Sign-off 安全的 early/late 區間

### 7.1 方向性

| 檢查 | 危險方向 | 需要的保守值 |
|---|---|---|
| 檢查 | Launch clock path | Data path | Capture clock path |
|---|---|---|---|
| Setup | late（上界） | late（上界） | early（下界） |
| Hold | early（下界） | early（下界） | late（上界） |

Launch 與 capture 的共同 clock 路徑若同時套 early 與 late，會產生人為悲觀，
OpenSTA 驗證時必須啟用 CRPR（clock reconvergence pessimism removal）並說明。

Hold 違例無法以降頻修復，hold 側 coverage 目標設得比 setup 高（候選：setup 99%、hold 99.9%，訓練前鎖定）。

### 7.2 方法

- Quantile heads（pinball loss）輸出 q_low、median、q_high。
- Split conformal calibration，calibration set 必須與 test 的切分方式一致（group-disjoint）：
  - E1：從 training partition 以 **table 為單位**切出 calibration tables（同一張表的點高度相關，不可以點為單位）。
  - E2：在訓練用的 VT 中再 leave-one-VT-out 當 calibration（模擬「未見 VT」的誤差分布）。
  - E3：在訓練用的 size 中再 leave-one-size-out 當 calibration。
  不使用 test labels；報告 conformal 前後的 coverage。
- 合併層級 L1、L2 都做；held-out 情境下 coverage 仍可能失效，照實報告。
- Coverage 是 per-arc（per-point）性質，不等於 path-level 保證；path-level 只以 OpenSTA 結果呈現。

### 7.3 指標

- Empirical coverage（整體、per-family、FF corner × 小 load 區域）。
- 區間寬度（絕對與相對）。
- 分方向的低估／高估比例與尾端（P99、max）誤差。
- OpenSTA：依 7.1 表套用 early/late，報告漏報違例數（目標 0）與誤報數（悲觀代價）。
- 寫回 .lib 的 early/late tables 本身也要滿足亮點 A 的硬約束（對 load 單調、corner 排序），
  且 q_low 不得低於 golden 允許的下限（負值依 4.4 的規則處理）；區間 head 一併套用單調結構。

### 7.4 主張邊界

- 區間表示**模型誤差**的不確定性，不是製程變異；實務上應疊加於 foundry OCV/LVF derate，不取代。
  定位：給「ML 產生的 library」用的安全邊界——LVF 涵蓋製程變異，但不涵蓋 ML 預測誤差。
- 資料只有 delay/transition tables，無 flop setup/hold constraint tables；本研究不預測 flop 的 hold requirement。

## 8. 延伸（非主線）

- C：跨 VT few-shot 校準（0%/1%/5%/10% target-VT budget）；只能主張 label-budget reduction，不主張 characterization runtime speedup。
- L3 全 combinational 單模型對照。
- SEQ family 納入。
- GF180 外部驗證（依論文期限決定）。

## 9. 風險與緩解

| 風險 | 緩解 |
|---|---|
| `when` arcs 未編碼導致 L1/L2 target 衝突 | W1 量化衝突比例；設計固定長度 state 編碼並加測試 |
| AO/OA 主導 L2 訓練 | type 等權抽樣；報告 per-family macro |
| 跨 size 樣本少（39 types、多數僅 2–3 size） | 分內插／外推報告；不對少樣本 type 下普遍結論 |
| OpenSTA 需要 transition，但主目標只有 delay | 寫回 .lib 時 transition 用 golden 並揭露；若時間允許另訓 transition 模型 |
| Conformal 在 held-out 分布偏移下 coverage 失效 | 照實報告；可加 group-conditional calibration |
| Monotonic 架構降低精度 | penalty 版本作消融；報告精度代價 |
| GPU 環境（1070 Pascal 與新版 PyTorch 相容性） | 先 CPU smoke；特徵預先快取；優先 3060 |
| 特徵前處理為瓶頸（逐列 Boolean 解析） | W2 前完成特徵快取（.npy/parquet） |

## 10. 里程碑（每週 8–10 小時估計，W2–3 smoke 後依實測修正）

| 週 | 里程碑 | 驗收 artifact |
|---|---|---|
| W1 | 資料準備 | `cell_type_catalog.csv`（type/family/size/VT，人工抽查 10 筆）；`when` 衝突比例報告；約束檢查 script（主線 delay 重算 6.1 表）+ 測試 |
| W2 | 特徵與 split | `when` 編碼、特徵快取；E1–E4 manifests（含 hash、group disjointness 測試） |
| W3 | L1 smoke + baselines | 3–5 個 types 的 E1/E2 smoke；LUT、Ridge、Ridge+、比例縮放實作與測試 |
| W4–W5 | L0/L1 全量 | E1–E4 × 5 seeds；L0→L1 gate 判定 |
| W6 | L2 | E1–E4 × 5 seeds；L1→L2 gate 判定；E5 粒度圖 |
| W7 | 亮點 A | 約束訓練 + violation 報告；OpenSTA 一個 design |
| W8 | 亮點 B | quantile + conformal；coverage 與 setup/hold 漏報分析 |
| W9+ | 寫作與延伸 | 論文圖表一鍵重建；視時間做 C／L3／SEQ |

## 11. 待確認

1. 論文繳交期限（決定 W9+ 延伸範圍與 GF180）。
2. GPU 主機（1070 或 3060）可用時程。
3. 老師對主軸「L0→L2 合併 + A + B」的認可。

## 12. 修訂紀錄

### v4.1（2026-10-18）— 納入獨立審查（Opus subagent，總評 PASS WITH FIXES）

審查已用 parquet 重算第 2 節全部數字，結果吻合；LIBMixer 三項描述經原文核對正確。
以下為修正項目（Critical 2、3 與 E4 共線另經 2026-10-18 本機查詢再次確認）：

| # | 嚴重度 | 修正 | 位置 |
|---|---|---|---|
| 1 | Critical | E2 的 VT 改為 ordinal／連續描述子，one-hot 只用於 E1 | 4.3 |
| 2 | Critical | Pin cap 跨 VT、跨 corner 皆 100% 不同；E2/E4 改用 reference 的 pin cap 並做消融 | 4.3 |
| 3 | Critical | 三 corner 的 P/V/T 共線；E4 降為探索性，僅主張 leave-TT-out 內插 | 5 |
| 4 | Major | Ridge+ 的 residual 需 reference；baseline 分「有／無 reference」兩類 | 5 |
| 5 | Major | Setup/hold 改為 launch/capture/data 分列，加註 CRPR | 7.1 |
| 6 | Major | Conformal calibration 改 group-disjoint、以 table 為單位 | 7.2 |
| 7 | Major | Gate 改為 non-inferiority（CI 上界 ≤ 1.10） | 4.2 |
| 8 | Major | L0 改分層抽樣約 2,000 張 tables | 4.1、4.2 |
| 9 | Major | 除 `when` 外，related_pin、timing_sense 也未編碼；95.5% rows 衝突；需 pin canonical role | 4.3 |
| 10 | Major | Corner 排序主線 delay 為 99.9993%，改為硬約束並對 corner ordinal 單調 | 6.1、6.2 |
| 11 | Minor | AO333 的 630 為 tables 而非 arcs | 2 |
| 12 | Minor | E3 fold 數與 size 分布 | 5 |
| 13 | Minor | Regex 得到 topology type（BUFx*f、HB* 待人工處理） | 2 |
| 14 | Minor | Early/late tables 亦需單調、q_low 下限 | 7.3 |

時程（第 10 節）未依審查調整，依 Rudy 指示另行處理。

### v4.2（2026-10-18）— 納入 W1 結果與 Rudy 決策

| # | 修正 | 位置 |
|---|---|---|
| 1 | 主線範圍確認做到 L2；L3 維持選做（Rudy 2026-10-18） | 4.1、8 |
| 2 | BUFx*f、HB1–4 各自獨立 type；CKINVDC 自成 type 不併 INV；86 types 確認 | 2 |
| 3 | 6.1 表後三列改為主線 delay 重算值（79.26%／70.67%／84.2%、55.7%），全表 VERIFIED-DATA | 6.1 |
| 4 | Slew 違反集中於大 slew 端（160→320 ps 佔 59%），修正 v3 的「小 slew 區」描述 | 6.1 |
| 5 | `when` 衝突 95.46% 重現；只加 `when` 剩 196 rows，加 `when`+`related_pin` 為 0 | 4.3 |
