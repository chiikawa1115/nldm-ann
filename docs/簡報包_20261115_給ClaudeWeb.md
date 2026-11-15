# 簡報製作素材包：Global ANN 泛化實驗與線性 Baseline 比較（2026-11-15）

> **給使用者（貼給 Claude 前）**：請一併上傳以下三張圖，對應第 3 節的 F2、F3、F5：
> - `experiments/global_ann_ridge_v1/figures/macro_mae.png`（F2）
> - `experiments/global_ann_ridge_v1/figures/macro_tail_error.png`（F3）
> - `experiments/expA_v2/accuracy_ci.png`（F5）

---

## 【給 Claude 的任務指示】

請根據**本文件的全部內容**，製作一份**研究進度簡報**，對象是指導教授與實驗室同學（研究進度報告，約 20 分鐘）。

**輸出方式（擇一，預設 A）**：
- **A. 逐頁內容版**：輸出每一頁的「標題／條列重點／圖表說明／講者備忘稿」，我會自己放進 PowerPoint。
- **B. 程式版**：若我要求，請用 `python-pptx` 產生完整 Python 程式碼。已上傳的圖（F2、F3、F5）直接插入；其餘圖表（F0、F1、F4）用 `matplotlib` 依第 3 節資料畫出。風格乾淨、學術。

**要求**：
1. 語言：**繁體中文**，用字精簡、學術；專有名詞（macro MAE、unseen-cell、Ridge 等）保留英文。
2. **以第 2 節的逐頁藍圖為骨架**（共 16 頁），可潤飾文字，但不要增減主張。
3. **所有數字只能用第 3 節的資料**，不要自行推算新的數字或補上文件沒有的數據。
4. **嚴格遵守第 4 節「不可宣稱」清單**。這份簡報的重點之一是「誠實呈現限制」，語氣請保持克制，不要使用「全面」「顯著優異」「證實」「突破」這類字眼。
5. 圖表請依第 3 節的「圖表規格」使用或繪製。
6. 最後一頁（第 16 頁）為講者專用的 Q&A 小抄，取自第 5 節。

---

# 1. 一頁摘要（給 Claude 建立整體理解）

- **主題**：用 ANN 取代靜態時序分析（STA）中 Liberty NLDM 的查表雙線性插值。
- **上一階段（Exp A，Nangate45）的結論**：在表格存在時，per-table MLP 的誤差是雙線性插值的 4.7–9.9 倍，查表法明顯較好。→ 研究方向轉向「**表格不存在**」的情境：新元件、新 PVT 角點。
- **本階段做了什麼**：
  1. 在 ASAP7（7nm）上訓練不使用 cell/corner ID 的 Global ANN，只用 42 維連續物理/電氣特徵，做 4 種切分模式共 40 個 runs 的泛化測試。
  2. 以相同特徵、相同測試集訓練 Global Ridge（線性 baseline），做 10 組配對比較。
  3. 核對 ANN checkpoint 的 metadata，確認兩者使用相同的資料、特徵與 normalization。
- **主要結果**：
  - 10/10 組配對中 ANN 的 macro MAE 都低於 Ridge（random-point 平均改善 80.75%，unseen-cell 71.33%）。
  - 未見角點中，SS 角點的誤差明顯最高。
- **主要限制**：ANN 的 P99 相對誤差仍高於 100%；尚未和「業界在沒有表時的實務作法」比較，這是下一步最優先的實驗。

---

# 2. 逐頁藍圖（共 16 頁）

### 第 1 頁：標題頁
- 標題：以物理特徵 Global ANN 預測 NLDM 時序值：泛化能力與線性 Baseline 比較
- 副標：研究進度報告｜2026-11-15

### 第 2 頁：大綱
1. 回顧與研究轉向
2. Global ANN 與實驗設計
3. 泛化結果（40 runs）
4. 與線性 baseline 的比較
5. 結論、限制與下一步

### 第 3 頁：回顧：NLDM 與研究問題
- NLDM：每個 timing arc 用 7×7 表格（input slew × output load）記錄 delay 與 transition，STA 用雙線性插值查表。
- 每個新元件、每個新 PVT 角點都要重新跑 SPICE 建表，成本高。
- 研究問題：能否用一個模型，對「沒有表」的元件或角點直接給出預測？
- 講者備忘：一句帶過即可，重點在於說明「沒有表」才是本研究的切入點。

### 第 4 頁：上一階段結論與研究轉向（Exp A，圖 F5 + 表 T0）
- Nangate45：125 cells、3,540 張表；每張表訓練一個 64×64 MLP，在 4 種 holdout 比例下與雙線性插值比較，各 5 seeds。
- 結果：所有 holdout 比例下，MLP 的 overall MAE 都是 LUT 的 4.7–9.9 倍，差值的 95% CI 皆不含 0。
- 轉向：表格存在時，查表法明顯較好；ANN 的價值應在表格**不存在**時。
- 由此改為：一個 Global 模型，輸入物理特徵而非 cell ID，才可能套用到未見元件與角點。
- 講者備忘：主動呈現負面結果，說明它如何決定了現在的研究方向。

### 第 5 頁：Global ANN 與資料
- 資料：ASAP7 7nm，12,073,208 筆，200 個 cell_base，FF/SS/TT 三個角點，含組合邏輯與循序元件（DFF/LATCH），delay 與 transition 表皆有。
- 輸入 42 維特徵（詳見表 T6）：表格軸（log slew、log load）、PVT（製程、電壓、溫度）、元件電氣屬性（drive strength、pin 電容、pin 數）、邏輯功能（function 計數與 logic type one-hot）、Vt 類別、表格類型（rise/fall delay/transition）。
- 刻意**不使用** cell ID / corner ID embedding，否則模型無法處理沒看過的元件或角點。
- 保留負的時序值，一併納入評估。

### 第 6 頁：四種切分模式（圖 F0）
- Random-point：測試點隨機抽，元件與角點在訓練時都看過。
- Unseen-cell：每個 seed 隨機抽 40 個 cell（20%），訓練時完全沒看過。
- Unseen-corner：整個 PVT 角點在訓練時沒看過，FF、SS、TT 各 hold-out 一次；每個角點只有一個固定測試集，5 個 seed 只改變模型初始化。
- Joint-unseen：元件與角點都沒看過；每個 seed 抽一組 cell，再分別與三個角點組合。
- 共 40 runs（5 + 5 + 15 + 15）。

### 第 7 頁：泛化結果（圖 F1 + 表 T1）
- 誤差隨外推難度上升：random-point 13.06 → unseen-cell 22.33 / unseen-corner 25.63 → joint 35.42 ps。
- Unseen-cell 與 unseen-corner 的 CI 大幅重疊，目前**無法判定**哪一個比較難。
- Joint-unseen：micro R² 0.847，但 macro R² 只有 0.640，代表部分 cell 預測品質明顯較差。
- 講者備忘：unseen-corner 與 joint 的 CI 以 n=15 計算，但 runs 之間共用測試集或 hold-out cell，CI 可能偏窄。

### 第 8 頁：逐角點結果：SS 最難（圖 F4 + 表 T5）
- Unseen-corner：FF 19.99、TT 23.20、SS 33.70 ps。
- Joint-unseen：FF 31.79、TT 27.30、SS 47.18 ps。
- 兩種模式下，hold-out SS（0.63 V / 100 °C）的誤差都最高。
- FF 雖然同樣位於電壓/溫度範圍的端點，在 unseen-corner 中卻最容易，所以難度不能單純用「內插 vs. 外插」解釋。
- 原因尚待分析（列入下一步）。

### 第 9 頁：為什麼要和線性 Baseline 比較？實驗設計
- 問題：42 維特徵已含大量物理資訊，ANN 的效果會不會只是特徵好，用線性模型也做得到？
- 設計：Global Ridge，與 ANN 使用**相同的 42 維特徵、相同測試 manifests、相同 validation mask 與指標程式**。
- 防止資料外洩：
  - fit-only normalization（只用訓練集統計量）。
  - Ridge 的 lambda 只依 validation MSE 從固定 6 個候選值中選，不看 test set。
- 2 種模式（random-point、unseen-cell）× 5 seeds = 10 組配對。

### 第 10 頁：主要結果：ANN vs. Ridge（圖 F2 + 表 T2）
- Random-point：13.06 vs. 67.82 ps，平均改善 80.75%，5 個 seed 介於 79.88%–81.83%，非常穩定。
- Unseen-cell：22.33 vs. 77.12 ps，平均改善 71.33%，但各 seed 介於 59.19%–77.26%，變異較大。
- 10/10 組配對方向一致。
- 講者備忘：p 值放在備忘即可（random-point Holm 校正後 5.16×10⁻⁹，unseen-cell 1.06×10⁻⁵），口頭說明只有 5 個 seed 且切分重疊，p 值僅供參考。

### 第 11 頁：逐 cell 與尾端誤差（圖 F3 + 表 T4）
- 逐 cell：random-point 1,000/1,000 個 cell/run 觀測都是 ANN 較低；unseen-cell 197/200。
- 3 個 ANN 較差的 cell 全部來自 seed 23：BUFx4、DFFLQNx3、DFFLQNx2。
- 尾端誤差：ANN 的 macro P90RE / P99RE 在 random-point 約為 Ridge 的 1/9 至 1/10，在 unseen-cell 約為 1/7。
- **但** ANN 的 macro P99RE 仍為 103%（random-point）與 168%（unseen-cell），尚未達到 sign-off 要求。
- 講者備忘：seed 23 最佳 epoch 為 4、第 12 epoch 停止，符合 early stopping 規則，目前不能據此判定原因，已列入下一步分析。

### 第 12 頁：可追溯性與比較資格核對
- 10 組 ANN checkpoint 的 metadata 全部與 Ridge 使用的資料吻合（matched）：dataset/manifest hash、42 維特徵順序、normalization 統計量（最大差約 1.85×10⁻¹³）、訓練設定、核心原始碼 Git blob hash。
- 這確認了兩者使用相同的資料、特徵與 normalization。
- Ridge 以串流方式計算（峰值記憶體約 1.13 GiB），10 組共 31 分鐘。
- 限制：這是 metadata 一致性核對；ANN 權重不在 repo 中，未重跑推論，不等於完整重現。

### 第 13 頁：結論
- **目前可以說**：
  - 在本次資料與切分設定下，Global ANN 的主要誤差與尾端相對誤差均低於 Global Ridge，10/10 組配對方向一致。
  - 不使用 ID embedding 的連續特徵表示，讓單一模型能對未見元件與角點給出預測，誤差隨外推難度上升。
  - 未見角點中，SS 角點明顯最難。
- **目前還不能說**：
  - 已優於業界在沒有表時的實務作法（尚未比較）。
  - 已可用於 sign-off（P99 相對誤差仍高）。
  - 結果可推廣到其他製程（只做了 ASAP7）。

### 第 14 頁：下一步（依優先順序）
1. **實務 baseline 比較**：unseen-corner 用相鄰角點插值或 derating；unseen-cell 用同類型、drive strength 最接近元件的表代用。
2. SS 角點分析：比較 SS 與 FF/TT 的數值範圍與誤差分布，找出 SS 較難的原因。
3. Seed 23 與尾端誤差歸因：誤差集中在哪些 cell 類型與 slew/load 區域。
4. OpenSTA 端到端整合：接入 `TableModel.cc` 的 `GateTableModel::gateDelay`，比較實際電路 critical path 的 slack 誤差。
5. 跨製程驗證：Sky130 或 GF180MCU。

### 第 15 頁：備用頁（不主動投影）
- 放表 T6（42 維特徵完整清單）與表 T3（逐 seed macro MAE），供提問時翻出。

### 第 16 頁：Q&A 小抄（講者專用，不投影）
- 內容取自第 5 節。

---

# 3. 數據與圖表規格

### 表 T0：Exp A（Nangate45）overall MAE，5 seeds 平均

| Holdout 比例 | LUT（雙線性插值） | MLP | MLP / LUT |
|---:|---:|---:|---:|
| 1/7（約 14%） | 0.69 | 3.21 | 4.7× |
| 2/7（約 29%） | 0.91 | 6.44 | 7.1× |
| 24/49（約 49%） | 0.77 | 7.64 | 9.9× |
| 34/49（約 69%） | 1.64 | 9.54 | 5.8× |

（單位與原始 Exp A 結果相同；各比例下差值的 95% CI 皆不含 0。）

### 表 T1：40-run 泛化結果（ANN）

| 模式 | Runs | Macro MAE (ps) | 95% CI (ps) | Macro R² | Micro MAE (ps) | Micro R² |
|---|---:|---:|---:|---:|---:|---:|
| Random-point | 5 | 13.06 ± 0.65 | [12.24, 13.87] | 0.9595 | 9.99 | 0.9797 |
| Unseen-cell | 5 | 22.33 ± 6.96 | [13.69, 30.98] | 0.8755 | 14.48 | 0.9378 |
| Unseen-corner | 15 | 25.63 ± 7.70 | [21.36, 29.90]* | 0.8265 | 20.94 | 0.8966 |
| Joint-unseen | 15 | 35.42 ± 12.01 | [28.77, 42.07]* | 0.6399 | 24.66 | 0.8469 |

（± 為樣本標準差；macro 指標為先逐 cell_base 計算再等權平均。* 以 n=15 計算，runs 間不完全獨立，可能偏窄。）

### 表 T2：ANN vs. Ridge（5 seeds 平均）

| 指標 | Random-point ANN | Random-point Ridge | Unseen-cell ANN | Unseen-cell Ridge |
|---|---:|---:|---:|---:|
| Macro MAE (ps) | 13.06 ± 0.65 | 67.82 ± 0.12 | 22.33 ± 6.96 | 77.12 ± 5.90 |
| Micro MAE (ps) | 9.99 | 56.22 | 14.48 | 56.27 |
| Macro R² | 0.9595 | 0.2802 | 0.8755 | -0.0359 |
| Micro R² | 0.9797 | 0.5582 | 0.9378 | 0.5411 |
| Macro P90RE (%) | 38.57 | 347.20 | 61.92 | 421.56 |
| Macro P99RE (%) | 103.28 | 1069.12 | 167.61 | 1227.19 |
| 差值 Ridge − ANN 的 95% CI (ps) | [54.07, 55.46] | | [49.23, 60.34] | |
| 平均相對改善 | 80.75% | | 71.33% | |

### 表 T3：逐 seed macro MAE（ps）

| 模式 | Seed | ANN | Ridge | 相對改善 |
|---|---:|---:|---:|---:|
| Random-point | 11 | 13.23 | 67.95 | 80.53% |
| Random-point | 23 | 13.65 | 67.84 | 79.88% |
| Random-point | 37 | 13.67 | 67.92 | 79.88% |
| Random-point | 53 | 12.43 | 67.71 | 81.63% |
| Random-point | 71 | 12.30 | 67.69 | 81.83% |
| Unseen-cell | 11 | 24.56 | 84.27 | 70.85% |
| Unseen-cell | 23 | 33.25 | 81.48 | 59.19% |
| Unseen-cell | 37 | 17.51 | 75.60 | 76.84% |
| Unseen-cell | 53 | 15.74 | 69.23 | 77.26% |
| Unseen-cell | 71 | 20.60 | 75.01 | 72.54% |

### 表 T4：ANN 較差的 3 個 cell（皆為 unseen-cell seed 23）

| Cell | ANN MAE (ps) | Ridge MAE (ps) |
|---|---:|---:|
| BUFx4_ASAP7_75t | 88.29 | 78.72 |
| DFFLQNx3_ASAP7_75t | 58.19 | 54.38 |
| DFFLQNx2_ASAP7_75t | 57.41 | 55.51 |

### 表 T5：逐角點 macro MAE（ps，每格 5 runs，mean ± SD [95% CI]）

| Hold-out 角點 | 電壓 / 溫度 | Unseen-corner | Joint-unseen |
|---|---|---:|---:|
| FF | 0.77 V / 0 °C | 19.99 ± 3.08 [16.16, 23.82] | 31.79 ± 9.23 [20.34, 43.25] |
| TT | 0.70 V / 25 °C | 23.20 ± 4.05 [18.17, 28.24] | 27.30 ± 4.40 [21.83, 32.76] |
| SS | 0.63 V / 100 °C | 33.70 ± 7.30 [24.64, 42.76] | 47.18 ± 11.34 [33.10, 61.25] |

（unseen-corner 每格的 5 runs 共用同一測試集，SD 只反映模型初始化的變異。）

### 表 T6：42 維輸入特徵

| 類別 | 特徵 |
|---|---|
| 表格軸 | log_slew、log_load |
| PVT | process_code、nom_voltage_v、nom_temperature_c |
| 元件電氣屬性 | drive_strength、related_pin_cap_ff（及兩者的缺值旗標）、n_input_pins、n_output_pins |
| 邏輯功能 | function 的 input / NOT / AND / OR / XOR 計數；logic type one-hot（INV、BUF、AND、NAND、OR、NOR、XOR、XNOR、AO、AOI、OA、OAI、MUX、DFF、LATCH、CLOCK、OTHER） |
| Vt 類別 | one-hot（SLVT、LVT、RVT、SRAM、UNKNOWN） |
| 表格類型 | one-hot（cell_rise、cell_fall、rise_transition、fall_transition） |

### 圖表規格

- **F0（第 6 頁）切分示意圖**：2×2 方格，橫軸「元件：看過／沒看過」，縱軸「角點：看過／沒看過」，四格分別標上 random-point、unseen-cell、unseen-corner、joint-unseen，可用由淺到深的顏色表示難度。
- **F1（第 7 頁）泛化長條圖**：x 軸為四種模式（依 T1 順序），y 軸為 macro MAE (ps)，誤差棒為 95% CI，長條上方標數值，長條下方標註 runs 數（5 / 5 / 15 / 15）。
- **F2（第 10 頁）**：使用上傳的 `macro_mae.png`（五 seeds 的 ANN 與 Ridge macro MAE 配對圖）。
- **F3（第 11 頁）**：使用上傳的 `macro_tail_error.png`（五 seeds 的 macro P90/P99 相對誤差配對圖）。
- **F4（第 8 頁）逐角點分組長條圖**：x 軸為 FF、TT、SS（標上電壓/溫度），每組兩根長條（unseen-corner、joint-unseen），y 軸 macro MAE (ps)，誤差棒為 T5 的 95% CI。
- **F5（第 4 頁）**：使用上傳的 `accuracy_ci.png`（Exp A 在各 holdout 比例下 MLP 與 LUT 的誤差）。若圖中資訊與 T0 不同，以 T0 為準。

---

# 4. 不可宣稱清單（請嚴格遵守）

1. 不可說「證實非線性模型是必要的」：只和一個線性模型比較過。
2. 不可說「跨角點比跨元件更難」或歸因於電壓/溫度非線性：兩者 CI 重疊，切分結構也不同。
3. 不可說「內插的角點比外插的角點容易」：逐角點結果不支持（FF 在 unseen-corner 中最容易）。也不可替 SS 較難下因果解釋，原因尚未分析。
4. 不可說「可用於 sign-off」或「滿足業界精度需求」：P99 相對誤差仍超過 100%。
5. 不可說「已完整重現 ANN」或「驗證了模型權重」：只做了 metadata 核對。
6. 不可說「ANN 優於傳統 NLDM 查表」：Exp A 顯示表格存在時查表法明顯較好；表格不存在時尚未與實務作法比較。
7. 不可說 seed 23 是「欠擬合」或「early stopping 太早」造成的：原因尚未分析。
8. 不可單獨強調 joint-unseen 的 micro R² 0.847，需與 macro R² 0.640 並列。
9. 不可把 p 值當成主要證據：每種模式只有 5 個 seed，切分重疊。
10. 不可說「確認比較完全公平」：metadata 核對只確認兩者使用相同的資料、特徵與 normalization。
11. 不可使用「zero-shot」「突破」「全面優於」等字眼。

---

# 5. Q&A 預備（講者小抄）

**Q1：只贏過 Ridge，能說明 ANN 有用嗎？業界在沒有表的時候怎麼做？**
A：目前只能說 ANN 優於使用相同特徵的線性模型。業界在缺角點時常用相鄰角點插值或 derating，缺元件時用相近元件的表代用，這正是下一步最優先要比較的 baseline。

**Q2：Exp A 不是說 ANN 贏不了查表？**
A：是的，在表格存在時，MLP 的誤差是查表的 4.7–9.9 倍，所以本研究不打算在那個情境取代查表，而是處理沒有表的情況。

**Q3：為什麼 unseen-cell 的變異這麼大？**
A：可能原因之一是每個 seed hold-out 的 40 個 cell 組合不同，難度不同。seed 23 改善率最低（59.19%），3 個 ANN 較差的 cell 都在這組，目前還在分析是切分難度還是訓練問題。

**Q4：只有三個角點，怎麼談 PVT 泛化？為什麼 SS 特別難？**
A：這是目前的限制，unseen-corner 實際上只有三個測試集。SS 在兩種模式下都最難，但原因還沒分析；FF 同樣是端點卻最容易，所以不是單純的外插問題。長期需要更多角點或其他製程的資料。

**Q5：P99 相對誤差超過 100%，這能用嗎？**
A：目前還不能用於 sign-off。相對誤差在延遲接近零時容易放大，下一步會分析高誤差點集中在哪裡，並用 OpenSTA 看實際路徑的 slack 誤差，那才是實際影響。

**Q6：怎麼確定 ANN 和 Ridge 的比較是公平的？**
A：兩者使用相同特徵、相同測試 manifests、相同指標程式；normalization 只用訓練集；Ridge 的 lambda 只看 validation。另外逐項核對了 ANN checkpoint 的 metadata，10 組都確認與 Ridge 使用相同的資料、特徵與 normalization。

**Q7：Ridge 的 macro R² 是負的，代表什麼？**
A：代表對平均每個 cell 而言，Ridge 比直接猜該 cell 的平均值還差；它的 micro R² 約 0.54，說明只看 point-weighted 指標會高估模型品質，所以兩者都要報。
