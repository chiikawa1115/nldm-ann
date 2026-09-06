# 跨 Cell 與跨 VT 的 NLDM Delay 研究計畫

日期：2026-09-06。狀態：使用者已同意方向；文獻全文比較與正式實驗待執行。

## 研究主軸

建立跨 cell 共用的 delay 模型，研究對未見目標 VT 使用少量 characterization 資料校準的效果。不同 drive strength、function family 與全部 combinational cells 的合併是比較與消融軸；跨 PVT 是後續驗證。Nangate45 保留為 pilot，優先盤點既有 ASAP7 多 VT 資料作主實驗。

研究問題：在共享 cell 結構與 operating-condition 表徵下，能否以少量目標 VT characterization 資料，預測未取樣 cell／timing arc 的 NLDM delay，並降低達到指定精度所需的資料成本？

候選貢獻為資料效率、真正未見條件的驗證與負遷移分析；不是僅以合併 MLP 為創新。新穎性尚待近期文獻全文比對，不宣稱首創。主目標先限定 cell delay；output transition 不自動納入。

## 階段與完成條件

| 階段 | 工作 | 產出與完成條件 |
|---|---|---|
| 0 文獻與資料盤點 | 精讀必讀四篇；確認 VT、PVT、cell/arc 覆蓋與 inference-available features | 文獻比較表、資料涵蓋表、可執行 RQ |
| 1 共享模型 | 比較獨立模型、同功能跨 size、family、global MLP、shared backbone + family head | 同資料預算與五 seeds 的誤差/模型成本比較 |
| 2 跨 size/cell | 分開 size 插值、size 外推與 unseen topology | 整組 holdout manifests、洩漏稽核與分群誤差 |
| 3 跨 VT 主實驗 | 保留整個 VT；分開 zero-shot 與 few-shot；輪替 target VT | calibration-budget curves 與公平 baseline |
| 4 跨 PVT 驗證 | 保留整個 corner；可行時增加 cell + condition joint holdout | 分離條件遷移與 topology 遷移的證據 |
| 5 實用驗證 | 評估 path delay；如需完整 STA，另補 transition 模型或說明使用 reference transition | end-to-end 誤差與成本、限制、可重現資料 |

## 分組與特徵

- 同功能跨 size 是起點；X1/X2 不直接當作精確線性 transistor width。
- AND、NAND、MUX 不預設共享 timing 行為。保留 pin arc、rise/fall、timing sense；區分 MUX select/data arcs。
- 特徵包括 slew/load、可取得的結構/Boolean/drive 資訊與 PVT/VT 描述；逐項列出推論時是否可取得及其取得成本。
- 真正 unseen cell 不以 cell ID embedding 作唯一表徵。只有 VT 類別名稱不能提供可靠的 zero-shot 物理對應；缺少 device descriptors 時，優先 few-shot。
- Pin capacitance 若由目標 characterization 取得，須揭露其成本；不得把目標 delay table 衍生特徵帶入零樣本任務。
- SS/TT/FF 類別或順序代理不是可獨立連續外推的物理製程參數。若 P/V/T 耦合，只主張 corner transfer。

## 實驗切分與 baseline

| 任務 | 切分 | 公平比較 |
|---|---|---|
| 稀疏同表預測 | 保留 table 中部分點 | Bilinear LUT、獨立 MLP、共享模型 |
| 未見 size | 全部相關 arcs/points 隨該 size 保留 | 同功能模型、global 模型、可用的簡單尺寸模型 |
| 未見 topology | 同 topology 的 size/VT variants 按任務定義整組隔離 | family/global/結構特徵模型 |
| 跨 VT zero-shot | 目標 VT labels 完全不參與訓練/調參 | 只用 source 資料的可部署模型 |
| 跨 VT few-shot | 固定 target calibration set，test 完全分離 | target-only scratch、pretrain + fine-tune、簡單 scaling/低階校準 |

跨 VT 任務可刻意保留相同 topology 在 source/target，以隔離 VT 效果；這不能同時稱為 unseen topology。另設 joint 任務才隔離兩者。

候選 calibration budgets 為 0%、1%、5%、10%，實際比例依資料盤點預先鎖定。明確定義單位為 points、tables、arcs 或 cells；分開「每顆 target cell 少量點」與「少數 target cells 提供資料，其餘 cell 無 labels」。

完全沒有 target table 時，bilinear interpolation 不具有直接可用的輸入，不能當作同條件 baseline。所有 few-shot 方法使用同一 calibration 資料預算。

## 指標與有效性

- MAE、RMSE、per-cell macro error、micro error、P90/P99 relative error；負值及近零 delay 保留並說明 relative-error 分母規則。
- 沿用五 seeds：11、23、37、53、71；報跨 seeds 變異，不能把同表高度相關點視為獨立重複實驗。
- Normalization 只 fit training；early stopping、超參數與 calibration selection 不使用 test labels。
- 比較固定 target-label budget 的誤差，及固定預先定義誤差門檻所需的資料量；門檻不事後挑選。
- 若只遮蔽現成 Liberty 表格，稱 label-budget reduction。只有實測 SPICE、特徵取得、訓練與推論總成本後，才能宣稱 characterization runtime speedup。
- 檢查負遷移、family 尾端誤差、模型數量/容量與訓練成本；不只比較全域平均。

## 既有結果與決策

本機交接紀錄記載 Nangate45 per-table 64×64 MLP 在已測 sparsity 下劣於 regular-grid bilinear LUT；本次未重跑驗證。保留負結果，不能宣稱 ANN 已勝出。

優先順序：同功能跨 size → 全 comb 共享 → 跨 VT few-shot → 跨 PVT/STA 驗證。跨製程驗證不是本輪主目標，不能以 Nangate45 → ASAP7 取代跨 VT 實驗。

現有程式與 manifests 不因本文件自動修改。正式執行前比對既有 RESEARCH_CONTRACT 與本輪設計，新增版本化 manifests，不覆寫既有實驗定義。

## 文獻與待確認事項

必讀清單見 [SELECTED_PAPERS.md](SELECTED_PAPERS.md)。逐篇擷取 input/output、cell 共享單位、target labels、holdout、VT/PVT 範圍、baseline、成本與限制；未讀到的欄位保留 unknown。

搜尋日期 2026-09-06；來源 Crossref、Semantic Scholar、出版社與作者公開頁面；搜尋詞涵蓋 cell library characterization、knowledge transfer、LIBMixer、GNN、generative model。這是定向初篩，不是完整 systematic review。

此計畫由 AI 協助整理，使用者已同意研究方向；文獻書目已初步核對，全文方法與新穎性尚待查證。
