# PowerPoint 製作素材包：用 ANN 取代 NLDM 查表（含 OpenSTA 背景）

---

## 【給 Claude 的任務指示】

請根據**本文件的全部內容**，幫我製作一份**研究進度簡報（PowerPoint）**，給指導教授與實驗室的雙週報。

**輸出方式（擇一，預設 A）**：
- **A. 逐頁講稿版**：先輸出完整的「投影片逐頁內容」（每頁：標題 / 條列重點 / 視覺元素說明 / 講者備忘稿），我會拿去填進 PowerPoint。
- **B. 可執行程式版**：若我要求，請用 **`python-pptx`** 產生一份 `.pptx` 的完整 Python 程式碼，套用乾淨的學術風格（標題頁 + 內容頁 + 表格 + 圖表），我可直接執行產檔。

**要求**：
1. 語言：**繁體中文**；用字精簡、學術、專業。
2. 重點分配：**ANN 方法 + `train.py` + 實驗流程與結果**約佔 6–7 成；「資料來源」「OpenSTA」各 1–2 頁背景即可。
3. 本文件第 11 節已給出**逐頁藍圖（共約 18 頁）**——請以它為骨架，把每頁的 bullet 潤飾成最終投影片文字，並補上流暢的講者備忘稿。
4. 程式碼片段請保留（第 9、10 節與附錄 A 有完整程式碼），放進對應投影片並在旁附一句說明。
5. 數據請用第 8 節的**真實實驗數字**；結果頁請據此描述一張折線圖（第 8.4 節有圖表規格）。
6. 最後產生一頁「Q&A 預備」講者小抄（取自第 10 節）。

---

# 1. 一頁摘要（給 Claude 建立整體理解）

- **主題**：用人工神經網路（ANN）取代靜態時序分析（STA）中的 NLDM 查表法。
- **已完成**：(1) 從原始碼建置開源 STA 工具 OpenSTA 並定位查表程式碼；(2) 建立完整 ANN 訓練/評估 pipeline；(3) 在 Nangate45 全庫上跑出「ANN vs 傳統查表」的真實精度對比。
- **核心結論**：**特徵化資料越稀疏，ANN 相對傳統雙線性插值的優勢越明顯**（holdout 50% 時 ANN 的平均相對誤差已反超傳統法，R² 全程領先）。
- **意義**：ANN 有機會用**更少的 SPICE 特徵化成本**達到同等或更佳的延遲建模精度。

---

# 2. 研究背景：NLDM 是什麼

- **NLDM（Non-Linear Delay Model）**：STA 工具用來預測標準單元延遲的模型，寫在 Liberty（`.lib`）檔中。
- 本質是一張**查表（LUT）**：`delay = f(input_slew, output_load)`，用一張 **7×7 表格**做**雙線性插值**。
- 每個 timing arc 有 4 種表：
  - `cell_rise` / `cell_fall`：輸出上升 / 下降的**延遲**
  - `rise_transition` / `fall_transition`：輸出上升 / 下降的**轉態時間（slew）**
- 兩個輸入軸：`index_1 = 輸入 slew`、`index_2 = 輸出負載電容`。

---

# 3. 研究動機：傳統查表的限制

- 只在特徵化格點附近準確，**格點之間用線性硬插**，離格點越遠越不準。
- 表格越大越慢、越小越不準（精度 vs 儲存/速度的取捨）。
- 每個 **PVT（Process-Voltage-Temperature）角點**都要重跑 SPICE，**特徵化成本極高**。
- 無法自然泛化到訓練範圍以外的操作點。

---

# 4. 為什麼用 ANN

- 用神經網路學一個**連續、可微、非線性**的函數 `f(slew, load) → delay`，取代離散查表。
- 天生**平滑、可泛化**；近年（2022–2025）DL-NLDM 等論文顯示能用**更少特徵化點**達到同等/更好精度。
- **本專案定位**：目前沒有現成的完整開源端到端 DL-NLDM 實作，因此從零建立「**可重現的基準**：liberty → MLP → 對比 LUT」。

---

# 5. 資料來源（背景，1 頁）

- **開源元件庫**：Nangate 45nm Open Cell Library（`NangateOpenCellLibrary`）。
- **涵蓋**：**125 個標準 cell**（INV / BUF / AND / OR / NAND / NOR / AOI / OAI / XOR / MUX / DFF …，含 `_X1/_X2/_X4` 驅動強度變體）。
- **資料規模**：
  - 獨立 LUT 表：**3,540 張**
  - 每張表：**7×7 = 49 點**
  - 總資料點：**173,460 筆**
  - 四種表格數量大致均衡（各約 43k 點）。

---

# 6. OpenSTA 環境（背景，1–2 頁）

- **STA**：不跑模擬，靠計算路徑延遲驗證晶片時序；**NLDM 查表就實際跑在 STA 引擎裡**。
- **OpenSTA**：開源 STA 引擎（也是 OpenROAD 時序核心），選它因為**開源、可讀可改**。
- **建置成果**：
  - Windows 筆電上以 **WSL2 + Ubuntu 24.04** 從原始碼編譯成功（`OpenSTA 3.1.0`）。
  - 自行編譯相依函式庫 **CUDD 3.0.0**。
  - 跑通內建範例，輸出完整時序路徑報告（每個 gate 的延遲值即 NLDM 查表結果）。
- **定位查表原始碼**：`liberty/TableModel.cc` 的 `GateTableModel::gateDelay(in_slew, load_cap) → (gate_delay, drvr_slew)`；其輸入輸出正好等於 ANN 的輸入輸出，是**未來把查表換成 ANN 推論的最佳接入點**。

---

# 7. ★ ANN 方法與模型架構（重點）

## 7.1 核心想法
把 NLDM 的「查表 + 雙線性插值」換成一個小型 MLP：`MLP(slew, load) → delay`，介面與查表完全相同，可無縫替換。

## 7.2 模型架構 `MLPNldm`
- 輸入 2 維：`(log_slew, log_load)`
- 隱藏層：`(64, 64)` + ReLU
- 輸出 1 維：delay 或 transition
- 參數量：約 4~5 千

（完整程式碼見附錄 A-1）

## 7.3 為什麼「每張表一個獨立 MLP」
- `train.py` 對 3,540 張表**各訓一個** `MLPNldm`。
- 每張 7×7 表就是**一個乾淨的二維函數**；用 per-table 小模型**隔離變因**，純粹回答「表示單一延遲曲面時，ANN 能否贏插值」。
- （進階版 `MLPNldmMultiCell` + cell embedding 讓一個模型跨所有 cell，是**下一階段**實驗。）

---

# 8. ★ 實驗 A：設計、流程與真實結果（重點）

## 8.1 科學問題
> 「同樣只給一部分特徵化點，ANN 能不能比傳統雙線性插值更準地還原其餘的點？」

## 8.2 方法：Grid Holdout + 公平比較
- 把每張 7×7 表**隨機遮掉一部分點當「考題」（test）**，ANN 與傳統插值都**只能用剩下的可見點**建模，再在被遮點上比誰準。
- **公平性**：兩種方法都沒看過被遮點。
- **真值來源**：`.lib` 表格本身即 SPICE 特徵化結果，被遮的已知點就是真值考題（不需另跑 SPICE）。
- LUT baseline 用 `scipy.LinearNDInterpolator`（只用可見點）忠實重現傳統做法。

## 8.3 三支程式流水線
```
data_loader.py          .lib → parquet（17 萬筆 / 3,540 張表）
   │
train.py  --holdout r   每張表遮 r 比例，可見點訓 MLP → ratio{r}.pt
   │
evaluate.py             被遮點上比 ANN vs LUT → ratio{r}.json（誤差指標）
   │
plot_expA.py            掃描 r → 論文主圖
```

## 8.4 真實結果（全 3,540 張表；holdout 20% / 30% / 50%）

**整體精度對比：**

| Holdout | 方法 | MAPE % | R² | MaxRE % |
|---|---|---|---|---|
| 20%（點多） | 傳統 LUT | **3.32** | 0.9932 | 986 |
| 20% | **ANN** | 3.90 | **0.9960** | 1420 |
| 30% | 傳統 LUT | **4.18** | 0.9899 | 1308 |
| 30% | **ANN** | 4.53 | **0.9943** | **560** |
| 50%（點少） | 傳統 LUT | 6.92 | 0.9747 | 8469 |
| 50% | **ANN** | **6.51** | **0.9883** | **3134** |

**建議圖表（給 Claude 畫成折線圖）**：
- 圖1「MAPE vs Holdout」：X = holdout(20,30,50)，Y = MAPE%；兩條線 ANN=[3.90, 4.53, 6.51]、LUT=[3.32, 4.18, 6.92]。
- 圖2「R² vs Holdout」：ANN=[0.9960, 0.9943, 0.9883]、LUT=[0.9932, 0.9899, 0.9747]。
- 標註重點：MAPE 兩線在 50% 交叉（ANN 反超）；R² 兩線差距隨 holdout 擴大。

## 8.5 分表格類型（以 50% holdout、cell_fall 為例）

| 方法 | MAPE % | R² | MaxRE % |
|---|---|---|---|
| 傳統 LUT | 10.44 | 0.9705 | 8469 |
| **ANN** | **6.05** | **0.9897** | **3134** |

- `cell_rise` / `cell_fall`（**延遲**表）：ANN 明顯較準。
- `rise_transition` / `fall_transition`（**轉態**表）：互有勝負，密網格時 LUT 略穩。

## 8.6 核心發現（報告主結論）
1. **資料越稀疏，ANN 優勢越明顯**：MAPE 在 50% holdout 反超 LUT（6.51 vs 6.92）。
2. **R² 全程領先且差距擴大**：+0.003（20%）→ +0.014（50%）。
3. **最壞情況更穩健**：50% 時 MaxRE 3134% 遠低於 LUT 8469%。
4. **延遲表尤其強**：cell_rise / cell_fall 上 ANN 全面領先。

---

# 9. 結論與下一步

**結論**：
- 已建立可重現的「ANN vs 傳統 NLDM 查表」基準 pipeline。
- 真實數據支持假設：**特徵化資料越少，ANN 越有價值** → 有望降低 SPICE 特徵化成本。

**下一步**：
1. 掃描 holdout 70% / 80%，補齊完整趨勢曲線。
2. 改用 `MLPNldmMultiCell` + cell embedding，一個模型涵蓋全部 cell。
3. **端到端驗證**：用 ANN 產生新的 `.lib` 餵回未改動的 OpenSTA，比對電路層級時序差異（不需改 C++）。
4. 長期：直接在 OpenSTA `GateTableModel::gateDelay` 內嵌 ANN 推論。

---

# 10. Q&A 預備（講者小抄）

| 可能提問 | 建議回答 |
|---|---|
| 沒有 SPICE，真值哪來？ | `.lib` 表格即 SPICE 特徵化結果，被遮的已知點當真值考題 |
| 為什麼用 grid holdout，不用隨機切？ | 要模擬「特徵化點變少」的真實情境，且每張表都要留考題 |
| ANN 和 LUT 比公平嗎？ | 公平，兩者都只用可見點建模，被遮點皆未見 |
| 為什麼每張表一個模型？ | 本實驗隔離「單一延遲曲面 ANN vs 插值」；跨 cell 共享是下一階段 |
| 為什麼輸入取 log？ | LUT index 為幾何級數，取 log 後輸入分布均勻，利於訓練 |
| 20% holdout 時 ANN 為何略輸？ | 網格仍密、線性插值本就強；ANN 價值在稀疏情境，趨勢已顯示 50% 反超 |
| ANN 比查表慢嗎？ | 單次推論極快；且可泛化、省特徵化成本，整體效益是重點 |
| 下一步？ | 掃描更高 holdout、cell embedding 跨 cell、ANN 產 .lib 餵回 OpenSTA |

---

# 11. ★ 逐頁投影片藍圖（共 18 頁，請以此為骨架展開）

> 格式說明：**【版面】**=建議排版；**【內容】**=投影片上的條列文字（可直接用）；**【視覺】**=要放的圖/表/程式碼；**【講稿】**=講者口白。

### 投影片 1 — 封面
- 【版面】標題頁
- 【內容】主標「以人工神經網路取代 NLDM 查表法之延遲建模」／副標「STA 環境建置與 ANN 基準實驗　雙週進度報告」／姓名・實驗室・日期
- 【講稿】「今天報告兩件事：OpenSTA 環境建置，以及用 ANN 取代查表的初步實驗結果。」

### 投影片 2 — 大綱
- 【內容】1 研究動機　2 資料與環境　3 ANN 方法與 train.py　4 實驗設計　5 結果　6 結論與下一步
- 【講稿】「先講動機與背景，重點會放在 ANN 方法與實驗結果。」

### 投影片 3 — 研究動機：NLDM 查表是什麼
- 【內容】第 2 節條列
- 【視覺】一張 7×7 表格示意（列=slew、行=load、格=delay），或用附錄 B 的真實 7×7 數值
- 【講稿】「NLDM 就是一張以 slew 和 load 為軸的 7×7 延遲查表。」

### 投影片 4 — 傳統查表的限制
- 【內容】第 3 節四點
- 【視覺】示意圖：格點間線性插值 vs 真實曲線的落差
- 【講稿】「查表在格點間只能線性硬插，且每個 PVT 角點都要重跑昂貴的 SPICE。」

### 投影片 5 — 為什麼用 ANN
- 【內容】第 4 節條列 + 本專案定位
- 【講稿】「改用 ANN 學一個連續函數，天生平滑、可泛化。」

### 投影片 6 — 資料來源：Nangate 45nm
- 【內容】第 5 節數字（125 cells / 3,540 表 / 173,460 點）
- 【視覺】cell 家族範例（附錄 B）
- 【講稿】「訓練資料來自 Nangate 45nm 開源標準庫。」

### 投影片 7 — OpenSTA 環境建置
- 【內容】第 6 節建置成果條列
- 【視覺】真實時序報告輸出（附錄 C）
- 【講稿】「我把 OpenSTA 從原始碼編譯起來並跑通時序分析。」

### 投影片 8 — 定位查表原始碼
- 【內容】呼叫鏈 `gateDelay → findValue → Table::findValueOrder2`；強調 gateDelay 是 ANN 接入點
- 【視覺】附錄 C 的 gateDelay 程式碼片段
- 【講稿】「查表核心在 gateDelay，其輸入輸出剛好等於 ANN，未來可無縫替換。」

### 投影片 9 — ANN 方法總覽
- 【內容】第 7.1 核心想法 + 「介面與查表相同」
- 【視覺】示意：`(slew, load) → [MLP] → delay`
- 【講稿】「把查表換成一個小 MLP，介面完全一樣。」

### 投影片 10 — 模型架構 MLPNldm
- 【內容】2→64→64→1、ReLU、約 5k 參數；「Linear 做線性組合、ReLU 提供非線性，可逼近任意曲面」
- 【視覺】附錄 A-1 程式碼
- 【講稿】「多層堆疊由通用逼近定理可逼近任意連續延遲曲面，這是它勝過線性插值的根本原因。」

### 投影片 11 — 為什麼每張表一個 MLP
- 【內容】第 7.3 條列
- 【講稿】「先隔離變因，證明單張表 ANN 能贏；跨 cell 共享留待下一步。」

### 投影片 12 — 實驗核心：Grid Holdout
- 【內容】第 8.2；強調公平比較與真值來源
- 【視覺】遮點示意圖（可見點 vs 被遮考題）
- 【講稿】「把每張表遮一部分當考題，兩種方法都只用剩下的點。」

### 投影片 13 — train.py 核心程式碼
- 【視覺】附錄 A-2（split_by_grid_holdout）+ A-3（train_one_table 精簡）
- 【內容】固定 seed 可重現；per-table 標準化；full-batch；early stopping
- 【講稿】「每張表獨立遮點、獨立訓練，固定 seed 讓評估能重現同一組切分。」

### 投影片 14 — train.py 四個關鍵設計決策
- 【視覺】表格（第 5.2 節那張：log 輸入 / per-table 標準化 / full-batch / early stopping）
- 【講稿】逐項一句話解釋。

### 投影片 15 — 實驗流程總覽
- 【視覺】第 8.3 節流水線圖
- 【內容】train → evaluate → plot；掃描多個 holdout 比例
- 【講稿】「一條可重現的流水線：抽資料 → 遮點訓練 → 公平評估 → 掃描畫圖。」

### 投影片 16 — 結果（一）：整體精度
- 【視覺】第 8.4 節表格 + 圖1(MAPE) 與 圖2(R²) 折線圖
- 【內容】核心發現前 3 點
- 【講稿】「資料越稀疏，ANN 優勢越大：50% 時 MAPE 反超、R² 全程領先。」

### 投影片 17 — 結果（二）：分表格類型
- 【視覺】第 8.5 節 cell_fall 表格
- 【內容】延遲表 ANN 強、轉態表互有勝負
- 【講稿】「在延遲表上 ANN 尤其明顯，MAPE 幾乎腰斬。」

### 投影片 18 — 結論與下一步
- 【內容】第 9 節
- 【講稿】「已建立可重現基準，結果支持假設；下一步補齊曲線、做跨 cell 模型與端到端驗證。」

（＋可選）投影片 19 — Q&A 預備（第 10 節表格）

---

# 附錄 A：完整程式碼

## A-1. 模型 `MLPNldm`（src/models/mlp_nldm.py）
```python
import torch.nn as nn

class MLPNldm(nn.Module):
    """取代 7x7 LUT 的最簡基準 MLP。輸入 (slew, load)，輸出 delay/transition。"""
    def __init__(self, in_features=2, hidden_dims=(64, 64),
                 out_features=1, dropout=0.0, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))   # 全連接層：線性組合
            layers.append(activation())         # 非線性激活 (ReLU)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))  # 輸出層
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

## A-2. Grid Holdout 切分（src/train.py）
```python
ARC_COLS = ["cell", "output_pin", "related_pin", "when", "table_type"]

def split_by_grid_holdout(df, holdout_ratio, seed=42):
    """對每張 LUT 表獨立隨機遮蓋 holdout_ratio 比例的格點。"""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby(ARC_COLS, sort=False):   # 每張表獨立切
        n = len(g)
        n_test = int(round(n * holdout_ratio))
        if n_test == 0 or n_test == n:
            continue
        idx = g.index.to_numpy(copy=True)
        rng.shuffle(idx)                            # 固定 seed → 可重現
        is_test[idx[:n_test]] = True
    return df[~is_test], df[is_test]                # train, test
```

## A-3. 單張表訓練（src/train.py，精簡）
```python
def train_one_table(df_arc, hidden_dims=(64, 64), n_epochs=200,
                    patience=25, lr=3e-3, weight_decay=1e-4):
    X = df_arc[["log_slew", "log_load"]].to_numpy(np.float32)  # log 尺度輸入
    y = df_arc["value"].to_numpy(np.float32).reshape(-1, 1)

    # per-table 標準化（每張表 delay 尺度差很大）
    x_mean, x_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = float(y.mean()), float(y.std()) + 1e-8
    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std
    X_t, y_t = torch.from_numpy(Xn), torch.from_numpy(yn)

    model = MLPNldm(2, hidden_dims, 1)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best, best_state, no_improve = float("inf"), None, 0
    for epoch in range(n_epochs):        # full-batch：點少，一次全丟
        optim.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optim.step()
        cur = loss.item()
        if cur < best - 1e-6:            # early stopping
            best, best_state, no_improve = cur, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, stats, best
```

## A-4. LUT baseline（src/evaluate.py，精簡）
```python
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def lut_predict_arc(train_arc, test_arc):
    """傳統雙線性插值 baseline：只用可見(train)點建插值器，預測被遮(test)點。"""
    pts  = train_arc[["slew", "load"]].to_numpy()
    vals = train_arc["value"].to_numpy()
    f_lin  = LinearNDInterpolator(pts, vals)
    f_near = NearestNDInterpolator(pts, vals)   # 凸包外的點用最近鄰補
    q = test_arc[["slew", "load"]].to_numpy()
    p = f_lin(q)
    mask = np.isnan(p)
    if mask.any():
        p[mask] = f_near(q[mask])
    return p
```

## A-5. 訓練設定摘要
- 每張表：`MLPNldm(2 → 64 → 64 → 1)`，ReLU
- 優化器：Adam，`lr=3e-3`，`weight_decay=1e-4`
- 損失：MSE；full-batch；early stopping（patience=25）
- 輸入：`log_slew, log_load`；per-table 標準化
- 效能：`torch.set_num_threads(2)`（3,540 個微型模型，鎖執行緒反而更快）

---

# 附錄 B：真實資料範例（可放進投影片）

**cell 家族範例（前 24 個）**：
`AND2_X1, AND2_X2, AND2_X4, AND3_X1, AND3_X2, AND3_X4, AND4_X1, AND4_X2, AND4_X4, AOI211_X1, AOI211_X2, AOI211_X4, AOI21_X1, AOI21_X2, AOI21_X4, AOI221_X1, AOI221_X2, AOI221_X4, AOI222_X1, AOI222_X2, AOI222_X4, AOI22_X1, AOI22_X2, AOI22_X4`

**一張真實 7×7 cell_rise 表（nangate45，示意 NLDM 長相）**：
- `index_1`（輸入 slew）：0.00117, 0.00472, 0.01719, 0.04098, 0.07806, 0.13008, 0.19854
- `index_2`（輸出 load）：0.365, 1.893, 3.786, 7.572, 15.144, 30.289, 60.577
- `values`（delay，第一列）：0.0229, 0.0279, 0.0332, 0.0428, 0.0613, 0.0977, 0.1705 …（共 7×7）

---

# 附錄 C：OpenSTA 素材（可放進背景頁）

**真實時序報告輸出（節錄）**：
```
Startpoint: r2 (rising edge-triggered flip-flop clocked by clk)
Endpoint:   r3 (rising edge-triggered flip-flop clocked by clk)
  Delay    Time   Description
   0.23    0.23 v r2/Q (DFF_X1)      ← NLDM 查表算出的 gate 延遲
   0.08    0.31 v u1/Z (BUF_X1)
   0.10    0.41 v u2/ZN (AND2_X1)
           9.43   slack (MET)
```

**查表核心程式碼（liberty/TableModel.cc，ANN 接入點）**：
```cpp
GateTableModel::gateDelay(const Pvt *pvt, float in_slew, float load_cap,
                          float &gate_delay, float &drvr_slew) const {
    gate_delay = findValue(pvt, delay_models_->model(), in_slew, load_cap, 0.0);
    drvr_slew  = findValue(pvt, slew_models_->model(),  in_slew, load_cap, 0.0);
    // ↑ 未來這兩行可改成一次 ANN 前向推論
}
```

---

# 附錄 D：專案程式結構
```
nldm_ann/
├── src/
│   ├── inspect_lib.py      探索 .lib 結構
│   ├── data_loader.py      .lib → parquet（攤平 7×7 表為三元組）
│   ├── models/mlp_nldm.py  ANN 模型（MLPNldm / MLPNldmMultiCell）
│   ├── train.py            ★ 實驗 A：per-table MLP + grid holdout
│   └── evaluate.py         ★ 實驗 A：ANN vs LUT 精度比對
├── experiments/expA/
│   └── plot_expA.py        多 ratio 論文級圖表
├── data/processed/         parquet 訓練資料
└── checkpoints/            訓練好的模型
```

---

# 附錄 E：關鍵名詞對照
- **NLDM**：Non-Linear Delay Model，Liberty 檔中的查表式延遲模型。
- **STA**：Static Timing Analysis，靜態時序分析。
- **slew / transition**：訊號上升/下降斜率（過渡時間）。
- **load**：輸出負載電容。
- **Grid holdout**：把網格點遮掉一部分當測試集的驗證方法。
- **MAPE**：平均絕對百分比誤差（越小越好）。
- **R²**：決定係數（越接近 1 越好）。
- **MaxRE**：最大相對誤差（衡量最壞情況）。
