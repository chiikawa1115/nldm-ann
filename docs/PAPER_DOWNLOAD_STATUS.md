# 額外論文下載紀錄

日期：2026-09-06。範圍：四篇必讀以外的 11 篇相關研究，涵蓋直接 characterization 研究與 timing/PDK 背景。

本輪結果：1 篇取得有效 PDF，10 篇未取得。未取得不代表論文不存在或必然付費。所有項目先查 OpenAlex 公開全文位置；再依可用來源嘗試出版社、作者網站或書目查詢。搜尋過程沒有使用訂閱帳號，也沒有繞過登入或驗證。

| 論文 | DOI | 本輪結果 |
|---|---|---|
| Deep-learning cell-delay modeling for static timing analysis | [10.1016/j.asej.2022.101828](https://doi.org/10.1016/j.asej.2022.101828) | OpenAlex 提供公開 PDF URL，但 ScienceDirect 回傳 HTML；Elsevier API PDF 請求回傳 INVALID_INPUT，不能據此判定缺訂閱 |
| Cell library characterization using machine learning for design technology co-optimization | [10.1145/3400302.3415713](https://doi.org/10.1145/3400302.3415713) | ACM PDF 回傳 Just a moment 驗證頁；Semantic Scholar 未提供公開 PDF |
| Machine Learning for On-the-Fly Reliability-Aware Cell Library Characterization | [10.1109/TCSI.2021.3069664](https://doi.org/10.1109/TCSI.2021.3069664) | IEEE PDF endpoint 回傳 HTML/JavaScript；未取得 PDF |
| Efficient Learning Strategies for Machine Learning-Based Characterization of Aging-Aware Cell Libraries | [10.1109/TCSI.2022.3201431](https://doi.org/10.1109/TCSI.2022.3201431) | IEEE PDF endpoint 回傳 HTML/JavaScript；未取得 PDF |
| Invited Paper: Boosting Standard Cell Library Characterization with Machine Learning | [10.1145/3658617.3703638](https://doi.org/10.1145/3658617.3703638) | ACM PDF 回傳 Just a moment 驗證頁 |
| Graph neural network based cell library characterization method for fast design technology co-optimization | [10.1016/j.vlsi.2024.102316](https://doi.org/10.1016/j.vlsi.2024.102316) | ScienceDirect PDF 請求回傳 HTML；未取得 PDF |
| Efficient Aging-Aware Standard Cell Library Characterization Based on Sensitivity Analysis | [10.1109/TCSII.2022.3212123](https://doi.org/10.1109/TCSII.2022.3212123) | IEEE PDF endpoint 回傳 HTML/JavaScript；未取得 PDF |
| Modeling Multiple-Input Switching in Timing Analysis Using Machine Learning | [10.1109/TCAD.2020.3009624](https://doi.org/10.1109/TCAD.2020.3009624) | IEEE PDF endpoint 回傳 HTML/JavaScript；未取得 PDF |
| PreRoutGNN for Timing Prediction with Order Preserving Partition: Global Circuit Pre-training, Local Delay Learning and Attentional Cell Modeling | [10.1609/aaai.v38i15.29653](https://doi.org/10.1609/aaai.v38i15.29653) | **成功**，AAAI 官方 PDF，9 頁；本機 `papers/cross_cell_vt/PreRoutGNN_2024.pdf` |
| A timing engine inspired graph neural network model for pre-routing slack prediction | [10.1145/3489517.3530597](https://doi.org/10.1145/3489517.3530597) | OpenAlex/Semantic Scholar 未提供公開 PDF；嘗試專案 README URL 回傳 404，未找到有效下載連結 |
| ASAP7: A 7-nm FinFET predictive process design kit | [10.1016/j.mejo.2016.04.006](https://doi.org/10.1016/j.mejo.2016.04.006) | OpenAlex 提供 PDF URL，但 ScienceDirect 回傳 HTML；未取得 PDF |

## 新增成功項目

PreRoutGNN 官方來源：https://ojs.aaai.org/index.php/AAAI/article/download/29653/31111

- 本機位置為 workspace 的 `papers/cross_cell_vt/PreRoutGNN_2024.pdf`，不在本 repository 內；本輪只授權上傳 LIBMixer PDF。
- 檔案大小：501,665 bytes；9 頁。
- SHA-256：`7db2deeb607143586d4915f7693dc8f4ac4807bcee61ba800aa39684ed1d10a3`。
- 已核對標題、PDF 格式與可解析性；驗證紀錄與 read-integrity sidecar 保存在本機。
- 這是 pre-routing timing prediction 背景，不能由其存在推論已證明跨 VT delay 泛化。

## VPN 與權限的區別

- 網站機器人驗證或 JavaScript/session 需求：即使開放存取也可能阻擋命令列下載。一般 VPN 不保證改善，換 IP 也可能觸發更多驗證。
- 出版社訂閱：如果網頁明確要求機構登入或購買，學校/公司的圖書館登入、proxy 或有訂閱资格的校園 VPN 才可能有幫助；一般商用 VPN 不會提供訂閱權限。
- 本次 IEEE/ScienceDirect 非 PDF 回應尚不足以確認是哪一种情況，不能直接斷言需要 VPN。
- 可先在正常瀏覽器透過 DOI 開啟，若有機構權限再使用正式登入；也可尋找作者公開版本或圖書館館際服務。

AI 協助檢索；下載與書目核對不是全文精讀或方法驗證。
