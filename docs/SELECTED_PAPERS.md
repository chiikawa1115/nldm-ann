# 跨 Cell / VT 研究：精選必讀論文

更新：2026-09-06。配合 [研究計畫](CROSS_CELL_VT_RESEARCH_PLAN.md)。

## 必讀順序與取得狀態

| 優先 | 論文 | 閱讀目的 | 全文狀態 |
|---|---|---|---|
| 1 | Tang et al., Knowledge transferring framework for cell library characterization, Microelectronics Journal, 2025. [DOI](https://doi.org/10.1016/j.mejo.2024.106542) | 查明 transfer 的 source/target domain、VT/PVT 範圍、目標 labels 預算與 split | 待取得；出版社請求未回傳 PDF，OpenAlex/Semantic Scholar 未提供可用公開 PDF |
| 2 | Lee et al., LIBMixer: An all-MLP Architecture for Cell Library Characterization towards Design Space Optimization, ASP-DAC 2025. [DOI](https://doi.org/10.1145/3658617.3697602) | 比較共用 MLP、cell/device 特徵、PVT 測試與 characterization 成本 | 已下載作者公開 PDF（7 頁）及作者簡報（21 頁），標題與可讀性已核對 |
| 3 | Wu et al., Generative Model Based Standard Cell Timing Library Characterization, DAC 2025. [DOI](https://doi.org/10.1109/DAC63849.2025.11133303) | 查明生成對象、conditioning 與目標 library 所需資訊 | 待取得；IEEE 下載回傳網頁，OpenAlex 未提供公開 PDF |
| 4 | Ma et al., Fast Cell Library Characterization for Design Technology Co-Optimization Based on Graph Neural Networks, ASP-DAC 2024. [DOI](https://doi.org/10.1109/ASP-DAC58780.2024.10473933) | 比較 transistor graph、size 表徵與 unseen topology 驗證 | 待取得；IEEE 下載回傳網頁，OpenAlex 未提供公開 PDF |

「待取得」僅指本次未取得，並不表示論文必然付費或不存在作者公開版本。書目存在性與全文方法已驗證是不同狀態；本次沒有完成全文精讀。

Tang 論文 Crossref 正式出版年份為 2025，Semantic Scholar 為 2024；後續引用核對 online-first 與卷期年份。

## 本機閱讀檔案

下載位置為 workspace 的 `papers/cross_cell_vt/`（在本 repository 外）：

- `LIBMixer_2025.pdf`：7 頁，1,601,522 bytes。
- `LIBMixer_2025_slides.pdf`：21 頁，1,892,981 bytes。作者簡報，不算另一篇論文。
- `download_validation.json`：檔案大小、頁數、SHA-256 與首頁文字，用於確認下載完整性與論文身分；不是精讀結果。

公開來源：[POSTECH CSDL 作者出版頁](https://csdl.postech.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=246)。使用頁面提供的 Paper / Slide 正常 session 下載連結；ACM 直接 PDF 請求遇到驗證頁。

## 精讀擷取欄位

1. Input/output 與預測單位：point、table、arc 或 cell。
2. 每個 cell 獨立或共用模型；function、size、topology 特徵。
3. Source/target library、VT/PVT 條件及 device 資訊。
4. 訓練、驗證、測試的 group-disjoint 單位。
5. Target labels 數量、取得方式、是否包含所有 target cells。
6. Zero-shot、few-shot 或已見條件插值；是否混用名稱。
7. Baseline 是否使用相同資料與特徵預算。
8. 平均與尾端誤差、negative transfer、STA/path 驗證。
9. SPICE、feature acquisition、training/inference 各項成本。
10. 對本研究可重現的比較及仍未被證實的候選貢獻。

## 書目修正備註

既有 LITERATURE_NOTES.md 的起始論文作者標示有誤：
Deep-learning cell-delay modeling for static timing analysis 的作者為 Waseem Raslan 與 Yehea Ismail，正式卷期年份 2023，DOI [10.1016/j.asej.2022.101828](https://doi.org/10.1016/j.asej.2022.101828)。本次僅記錄修正，不修改該未提交舊文件。

本清單由 AI 協助檢索與整理。搜尋來源為 Crossref、OpenAlex、Semantic Scholar、出版社及作者機構頁；搜尋引擎另遇到驗證頁或不相關結果，未將其作為方法證據。不能由這份定向清單推論文獻已窮盡。
