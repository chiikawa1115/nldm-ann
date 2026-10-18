# Research Contract: Feature-Conditioned NLDM Generalization

## Research question

使用 cell 結構特徵與連續 PVT 特徵的 ANN，對 ASAP7 NLDM delay/transition
進行預測時，在 unseen cell、unseen corner，以及兩者同時 unseen 的條件下，
其預測精度與泛化能力如何？模型在 GF180MCU 上的 external technology
validation 表現如何？

## Dataset roles

- **ASAP7**：主實驗；使用同一 standard-cell library 的 FF/SS/TT 與可取得的
  LVT/RVT/SLVT/SRAM Vt 類別。
- **GF180MCU**：external validation；7-track 與 9-track 必須分層報告，不把
  technology/library shift 誤稱為 unseen-corner。
- **IHP SG13G2**：選配 robustness check。
- **Nangate45**：方法與重現性 pilot；不回答主 RQ。

## Inputs available at inference

Continuous operating features:

- input slew (ps)
- output load (fF)
- process numeric descriptor（ASAP7 的 `nom_process` 在 FF/TT/SS 皆可能為 1，
  因此主實驗使用 versioned SS=-1/TT=0/FF=+1 ordered proxy，並強制做 ablation）
- voltage (V)
- temperature (°C)

Cell/arc features:

- logic function/category
- drive strength
- Vt class
- input pin capacitance (fF)
- input/output pin counts
- timing sense/type、rise/fall table type
- transistor/topology features 僅在 CDL/SPICE netlist 可追溯取得時加入

禁止使用 cell ID embedding 或 corner ID embedding 作為 unseen group 的必要輸入。
ID 可保留於 manifest、grouping 與誤差分析，但不得進入主模型 feature tensor。

ASAP7 unseen-cell 以移除 `_L/_R/_SL/_SRAM` 後的 `cell_base` 分組；同一
topology 的所有 Vt variants 必須一起進 train 或 test。Boolean function 同時轉為
固定 operator-count features，避免 unseen cell 依賴新的 function vocabulary ID。

資料清理規則：`related_pin` 必須解析到 direction=`input` 且有 capacitance。
ASAP7 FA/HA 中少數 `related_pin=CON` 指向 output pin 的 arcs 排除，不以 output
capacitance 或全域平均值冒充 input pin capacitance。

## Leakage-safe experiment matrix

| Experiment | Training groups | Test groups | Claim boundary |
|---|---|---|---|
| Random point | 每個 cell × PVT table 的部分 grid points | 同 table held-out points | interpolation upper bound；不代表泛化 |
| Unseen cell | seen cells × all eligible PVTs | held-out cells × same eligible PVTs | cell generalization |
| Unseen corner | all eligible cells × seen PVTs | same eligible cells × held-out PVT | PVT generalization |
| Joint unseen | seen cells × seen PVTs | held-out cells × held-out PVTs | strict compositional generalization |

Joint split 的兩個 cross quadrants（held-cell/seen-PVT、seen-cell/held-PVT）須標為
diagnostic 或 excluded，不可混進 train/test。所有 split 先在 table/group 層建立，再
展開到 grid points；normalization 只由 training partition fit。

## Repetition and reporting

- 固定 seeds：至少 3；主設定沿用 5 seeds（11, 23, 37, 53, 71）。
- 指標：MAE、RMSE、sMAPE、P90/P99 relative error、R2。
- **Micro**：所有 test points 合併計算。
- **Macro**：先對每個 held-out cell 計算，再對 cell 等權平均。
- 報告 mean ± standard deviation 與 95% confidence interval；比較模型時使用相同
  split 的 paired seed differences。

## Primary hypotheses

- H1：顯式 cell/PVT features 在 unseen-cell 與 unseen-corner 上優於不含該類特徵的
  MLP ablation。
- H2：joint-unseen 誤差高於單獨 unseen-cell 或 unseen-corner。
- H3：GF180 external validation 的 domain-shift 誤差高於 ASAP7 internal splits。

以上皆為待檢驗假設，不預設 ANN 必然優於 LUT。LUT 只在同一 cell/corner 有足夠
characterization points 時構成 interpolation baseline；對完全 unseen cell/corner，
不得虛構不存在的 LUT baseline。
