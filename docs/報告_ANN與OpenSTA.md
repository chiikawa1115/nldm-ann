# 週報：OpenSTA 環境建置 + ANN 延遲模型

> 報告範圍：(1) 建立 OpenSTA 靜態時序分析環境並定位 NLDM 查表原始碼；(2) 建立以 ANN 取代查表的深度學習模型與程式架構。
> 每個 `##` 段落 = 一頁投影片；「🎤 講稿」是報告時可以直接照唸的口白。

---

## 頁 0：一句話總結研究目標

用 **ANN（人工神經網路）取代 STA 工具裡的 NLDM 查表法**，讓標準單元延遲預測更準、更省特徵化成本。

🎤 講稿：
> 「這週我做了兩件事：一是把開源 STA 工具 OpenSTA 建置起來、找到它做延遲查表的原始碼位置；二是把要替換查表的 ANN 模型與整套訓練程式建好。這兩塊接起來就是我論文的核心。」

---

# 第一部分：OpenSTA 環境

## 頁 1：什麼是 STA / OpenSTA？為什麼需要它

- **STA（Static Timing Analysis，靜態時序分析）**：不跑模擬，靠計算每條路徑延遲來驗證晶片時序是否滿足時脈要求。
- **OpenSTA**：開源的 STA 引擎（也是 OpenROAD 的時序核心）。
- **關鍵**：STA 算每個 gate 的延遲時，用的就是 **NLDM 查表法** —— 這正是我要用 ANN 取代的對象。所以要先把它建起來、看懂它的查表程式碼。

🎤 講稿：
> 「NLDM 查表不是抽象概念，它實際跑在 STA 工具裡。我選 OpenSTA 是因為它開源、可以看到、也可以改原始碼，是驗證我方法最真實的平台。」

---

## 頁 2：環境建置成果

在 Windows 筆電上，透過 **WSL2 + Ubuntu 24.04** 從原始碼編譯成功：

| 項目 | 內容 |
|---|---|
| 作業系統 | WSL2 Ubuntu 24.04（Windows 無法原生編譯 EDA 工具） |
| 相依套件 | cmake / gcc13 / swig / bison / flex / tcl；**CUDD 3.0.0**（必要，需自行編譯） |
| 建置產物 | `OpenSTA 3.1.0` 執行檔（`build/sta`） |
| 驗證 | 成功跑通內建範例，輸出完整時序路徑報告 |

🎤 講稿：
> 「EDA 工具都在 Linux 上，所以我用 WSL2 裝 Ubuntu，從原始碼編譯。過程中比較卡的是 CUDD 這個相依函式庫要另外自己編。最後成功產生 sta 執行檔並跑通範例。」

---

## 頁 3：跑通證明 —— 實際時序報告

`sta` 對一個範例電路做時序分析，輸出的關鍵路徑：

```
Startpoint: r2 (rising edge-triggered flip-flop clocked by clk)
Endpoint:   r3 (rising edge-triggered flip-flop clocked by clk)

  Delay    Time   Description
---------------------------------------------------
   0.23    0.23 v r2/Q (DFF_X1)      ← 這些延遲值就是 NLDM 查表算出來的
   0.08    0.31 v u1/Z (BUF_X1)
   0.10    0.41 v u2/ZN (AND2_X1)
           0.41   data arrival time
           9.43   slack (MET)         ← 時序滿足
```

🎤 講稿：
> 「這是實際輸出。大家看 `r2/Q 0.23`、`u1/Z 0.08` 這些每個 gate 的延遲數字 —— **它們就是 NLDM 查表 + 雙線性插值算出來的**，也就是我要用 ANN 取代的目標值。」

---

## 頁 4：定位查表原始碼（研究的切入點）

查表邏輯在 `liberty/TableModel.cc`，呼叫鏈：

```
GateTableModel::gateDelay()     ← ★ 未來接 ANN 的最佳位置
   收 (in_slew, load_cap) → 回傳 (gate_delay, drvr_slew)
        │
        └─► findValue() → TableModel::findValue()
                 └─► Table::findValueOrder2()   ← 雙線性插值的數學在這
```

- `GateTableModel::gateDelay` 的介面正好是 `(slew, load) → (delay, slew)`，**跟我 ANN 的輸入輸出一模一樣** → 是最乾淨的替換邊界。

🎤 講稿：
> 「我追進原始碼，找到查表的核心函式。重點是 `gateDelay` 這個函式，它的輸入輸出剛好等於我 ANN 的輸入輸出，代表未來要把查表換成 ANN 推論，就是改這裡。」

---

# 第二部分：ANN 延遲模型

## 頁 5：程式架構總覽

一條 **「.lib → 資料 → ANN → 對比 LUT」** 的 pipeline，`src/` 五支檔案：

```
 .lib (NLDM 7×7 查表)
   │  inspect_lib.py   探索：看懂 .lib 裡的表長怎樣
   ▼
 data_loader.py        把 7×7 表攤平成 (slew,load)→value，存 parquet
   ▼
 models/mlp_nldm.py    ★ ANN 架構（MLPNldm / MLPNldmMultiCell）
   ▼
 tutorial_train.py     合成資料跑通流程（驗證 pipeline）
 train_real.py         真實 ASAP7 資料訓練 + 對比傳統 LUT（主實驗）
```

🎤 講稿：
> 「程式分五支，是一條完整流水線：先把 .lib 的查表資料攤平成訓練樣本，定義 ANN 模型，再用真實資料訓練並和傳統查表比精度。」

---

## 頁 6：資料轉換（方法的關鍵一步）

一張 7×7 NLDM 表 → 49 筆訓練樣本：

```python
# data_loader.py：把 (index_1 × index_2 → values) 攤平成三元組
for i, s in enumerate(idx1):          # idx1 = input slew
    for j, ell in enumerate(idx2):    # idx2 = output load
        yield Sample(slew=s, load=ell, value=vals[i, j])   # → delay
```

🎤 講稿：
> 「這步是把『離散查表』重新理解成『一個連續函數上的取樣點』。有了 (slew, load, delay) 三元組，就能拿去訓練 ANN。」

---

## 頁 7：ANN 架構 — 基礎 MLP

```python
class MLPNldm(nn.Module):
    def __init__(self, in_features=2, hidden_dims=(64,64,64), out_features=1):
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))   # 全連接層：線性組合
            layers.append(nn.ReLU())            # 非線性激活
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

架構圖：`(slew,load)[2] → 64 → 64 → 64 → delay[1]`，約 8,600 參數。

🎤 講稿：
> 「最基礎的模型就是多層感知器。Linear 做線性組合、ReLU 提供非線性，堆疊起來能逼近任意連續函數 —— 這就是它比『線性插值』更準的根本原因。」

---

## 頁 8：★ 進階架構 — cell embedding（亮點）

```python
class MLPNldmMultiCell(nn.Module):
    def __init__(self, num_cells, embedding_dim=16, ...):
        self.cell_embed = nn.Embedding(num_cells, embedding_dim)  # 每個 cell 一個可學向量
        # MLP 輸入 = cell 向量 + (slew, load)
    def forward(self, x, cell_id):
        emb = self.cell_embed(cell_id)      # 查出 cell 向量
        h = torch.cat([emb, x], dim=-1)     # 拼接
        return self.mlp(h)
```

- **一個模型服務所有 cell**：cell 身份編成 16 維向量，訓練時自動學到「哪些 cell 相似」。
- 對比：傳統要存幾萬個查表數字；ANN 用一個模型（~36k 參數）學到的是**連續函數**。

🎤 講稿：
> 「這是最有價值的設計：與其每個 cell 各訓一個模型，我用 cell embedding，把 cell 身份編成向量、和模型一起學。相似的 cell 會學到相近的向量，一個模型就涵蓋全部單元，也更容易泛化。這是 DL-NLDM 論文的標準做法。」

---

## 頁 9：訓練流程與對照組

深度學習核心五步（`tutorial_train.py`）：

```python
optimizer.zero_grad()          # 清梯度
pred = model(x_batch)          # 前向
loss = loss_fn(pred, y_batch)  # MSE 損失
loss.backward()               # 反向：算梯度
optimizer.step()              # Adam 更新權重
```

- 前處理：輸入正規化（用訓練集統計量）、輸出取 log（delay 動態範圍大）。
- **對照組**：`train_real.py` 用 `scipy` 雙線性插值忠實重現傳統 NLDM，用 MAE / MaxE / R² 直接和 ANN 比。

🎤 講稿：
> 「訓練就是標準的五步迴圈。我特別做了輸入正規化和輸出取 log 兩個前處理。最後和傳統雙線性插值做同一組測試資料的精度對比 —— 這就是論文第一個實驗結果。」

---

## 頁 10：兩部分如何銜接（下一步）

```
Python 端 (ANN)                          Linux 端 (OpenSTA)
  訓練 ANN ──► 產生「ANN 版 .lib」 ──────►  未改動的 OpenSTA 讀入
                                              │
                                        跑同一個電路，比對
                                        「ANN vs 原始查表」的時序差異
```

- **短期（選項 2，不改 C++）**：ANN 產生 .lib → 餵 OpenSTA → 端到端驗證誤差。
- **長期（選項 1）**：直接在 `GateTableModel::gateDelay` 內嵌 ANN 推論。

🎤 講稿：
> 「下一步先用不改 C++ 的方式：讓 ANN 產生一份新的 .lib 餵回 OpenSTA，做端到端驗證。長期再把 ANN 直接嵌進剛剛找到的 gateDelay 函式裡。」

---

## 頁 11：本週小結

- ✅ OpenSTA 3.1.0 從原始碼編譯成功、跑通時序分析
- ✅ 定位 NLDM 查表原始碼與未來 ANN 接入點
- ✅ 建立完整 ANN 訓練 pipeline（資料抽取 / MLP / cell embedding / LUT 對照）
- ⏭ 下一步：ANN 產生 .lib → OpenSTA 端到端精度驗證

🎤 講稿：
> 「總結這週：環境和原始碼定位完成，ANN 模型和訓練程式也建好，下一步就是把兩邊接起來做端到端驗證。」
