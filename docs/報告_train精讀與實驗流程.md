# 報告重點：ANN 訓練（train.py）與實驗 A 流程精讀

> 報告主軸 = 這份。目標：講清楚「我怎麼用 ANN 取代 NLDM 查表，並公平地和傳統插值比精度」。
> 附帶輕帶：用的開源元件庫（第 0 節）。

---

## 0. 資料來源（報告只需一頁輕帶）

- **開源庫**：Nangate 45nm Open Cell Library（`NangateOpenCellLibrary`），隨 OpenSTA 附帶的標準元件庫。
- **規模**：125 個標準 cell（AND / OR / AOI / OAI / INV / BUF / XOR / DFF …，含 `_X1/_X2/_X4` 不同驅動強度）。
- **資料**：每個 timing arc 有 4 種 NLDM 表（`cell_rise` / `cell_fall` / `rise_transition` / `fall_transition`），
  每張都是 **7×7 = 49 點** 的 `(input_slew, output_load) → 值`。
- **總計**：**3,540 張表 × 49 點 = 173,460 筆資料**。

🎤 一句話：「訓練資料來自 Nangate 45nm 開源庫，125 個 cell、3,540 張 7×7 查表，共 17 萬筆延遲/轉態點。」

---

## 1. 實驗架構總覽（先講這張，聽眾才懂後面）

**科學問題**：
> 「同樣只給一部分特徵化點，ANN 能不能比傳統雙線性插值更準地還原其餘的點？」
> （＝ 能不能用更少的 SPICE 特徵化成本，達到同等甚至更好的精度）

**方法：Grid Holdout（格點遮蓋）**

```
一張 7×7 表 (49 點)
   │  隨機遮掉 holdout_ratio 比例的點當「考題」(test)
   ▼
 ┌─────────────┬─────────────────────────┐
 │ 可見點 train │ 被遮點 holdout / test    │
 └─────────────┴─────────────────────────┘
       │                      │
       │ 兩種方法都「只能用可見點」 │
       ▼                      ▼
   ① ANN 訓練            ② 傳統雙線性插值
       │                      │
       └──── 在被遮點上比誰準 ───┘   ← MAE / MAPE / MaxRE / R²
```

**三支程式的分工**：

| 程式 | 做什麼 | 產出 |
|---|---|---|
| `train.py` | 對每張表用**可見點**訓一個小 MLP | `ratioX.pt`（3,540 個模型權重） |
| `evaluate.py` | 在**被遮點**上比 ANN vs LUT | `ratioX.json`（各種誤差指標） |
| `plot_expA.py` | 多個 holdout 比例畫成曲線 | 論文主圖 PNG |

**公平性設計（報告一定要強調）**：ANN 和 LUT **都只看得到可見點**，被遮點兩邊都沒看過 → 這才是公平比較。

🎤 一句話：「我把每張 7×7 表遮掉一部分當考題，ANN 和傳統插值都只用剩下的點，再比誰把考題答得準。」

---

## 2. train.py 逐段精讀（報告核心）

### 2-1. 為什麼「每張表一個獨立 MLP」？

`train.py` 對 3,540 張表**各訓一個** `MLPNldm`（2→64→64→1，約 4~5 千參數）。

- 每張 7×7 表就是**一個乾淨的二維函數** `f(slew, load) → delay`。
- 目的是最純粹地回答：「表示**單一張延遲曲面**時，ANN 能不能贏線性插值？」
- 所以用「per-table 小模型」隔離變因，不摻入跨 cell 的複雜度。
  （跨 cell 共享 + embedding 是**之後**另一個實驗，用 `MLPNldmMultiCell`。）

### 2-2. 核心方法：`split_by_grid_holdout`（train.py:61）

```python
for _, g in df.groupby(ARC_COLS, sort=False):   # 每張表獨立處理
    n = len(g)
    n_test = int(round(n * holdout_ratio))       # 例：0.7 → 49*0.7 ≈ 34 點當考題
    idx = g.index.to_numpy(copy=True)
    rng.shuffle(idx)                             # 固定 seed=42，可重現
    is_test[idx[:n_test]] = True                 # 前 n_test 個標記為 holdout
```

**重點**：
- **每張表獨立切**（不是全域切）→ 保證每張表都有 train 也有 test，且比例一致。
- **固定 seed** → `evaluate.py` 能用同樣 seed 重現「哪些點被遮」，兩邊對得上。

### 2-3. 單張表訓練：`train_one_table`（train.py:94）

```python
X = df_arc[["log_slew", "log_load"]].to_numpy()   # 用 log 尺度（index 是幾何級數）
y = df_arc["value"].to_numpy().reshape(-1, 1)

# per-table 標準化：每張表 delay 範圍差很大，各自標準化才好收斂
Xn = (X - x_mean) / x_std
yn = (y - y_mean) / y_std

model = MLPNldm(in_features=2, hidden_dims=(64,64), out_features=1)
optim = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

for epoch in range(n_epochs):        # full-batch（點很少，一次全丟）
    optim.zero_grad()
    loss = MSE(model(X_t), y_t)
    loss.backward()
    optim.step()
    # early stopping：loss 連續 patience 次沒進步就停
```

**四個要懂的設計決策**：

| 決策 | 為什麼 |
|---|---|
| 輸入用 **log_slew / log_load** | LUT 的 index 是幾何級數（越大間距越大），取 log 讓點在輸入空間分布均勻，MLP 好學 |
| **per-table 標準化**（x 和 y 都標準化） | 每張表 delay 尺度差很大（幾 ps ~ 幾百 ps），各自標準化才不會有的表主導、有的學不動 |
| **full-batch**（不用 DataLoader） | 每張表只有 ~49 點，一次全部丟進去最快，省掉 DataLoader 的 Python overhead |
| **early stopping**（patience=25） | 點少容易過擬合；loss 不再下降就提早停，省時間也較穩 |

> 小細節（可不講）：early stopping 監看的是**訓練 loss**。因為 train.py 只負責「把可見點擬合好」，
> 真正的類推能力（holdout 點準不準）留給 evaluate.py 去測 —— 職責分離很乾淨。

### 2-4. 訓所有表：`train_all_tables`（train.py:171）

```python
for keys, g in tqdm(groups):          # 3,540 張表逐一
    if len(g) < min_points:           # 點太少(<6)沒訓練意義，略過
        skipped += 1; continue
    model, stats, loss = train_one_table(g, ...)
    ckpts[keys] = {"state_dict":..., "stats":..., "n_train":..., "train_loss":...}
```

- `min_points` 保護：太少點的表跳過。
- 每張表存下：**權重 + 標準化統計量 + 訓練點數 + 最終 loss**。

### 2-5. 存檔格式（train.py:274）

```python
torch.save({
    "config": {holdout, seed, hidden_dims, lr, ...},   # 重現實驗用
    "arcs":  [arc_key, ...],                            # 表的順序（固定）
    "checkpoints": {arc_key: {state_dict, stats, ...}}, # 每張表的模型
}, out_path)
```

`config` 存了 seed 和 holdout → `evaluate.py` 才能**精準重現同一組切分**。

---

## 3. 為什麼這樣設計（報告可能被問，先準備好）

| 問題 | 回答 |
|---|---|
| 為什麼用 grid holdout，不用隨機切？ | 要模擬「特徵化點變少」的真實情境，而且每張表都要留考題，格點層次的遮蓋最貼切 |
| 為什麼每張表一個模型，不用一個大模型？ | 這個實驗只想隔離「單一延遲曲面 ANN vs 插值」；跨 cell 共享是下一個實驗 |
| ANN 和 LUT 比公平嗎？ | 公平：兩者都只用可見點建模，被遮點都沒看過 |
| 沒有 SPICE，真值哪來？ | .lib 的表格本身就是 SPICE 特徵化結果，被遮的點就是已知真值當考題 |
| 為什麼掃多個 holdout 比例？ | 高 holdout（點少）才是 ANN 的主場；掃描能畫出「資料越少 ANN 優勢越大」的趨勢 |

---

## 4. 完整實驗流程（一頁收尾）

```
data_loader.py         .lib → parquet（17 萬筆，3,540 張表）
   │
train.py  --holdout r  每張表遮 r 比例，可見點訓 MLP → ratio{r}.pt
   │
evaluate.py            被遮點上比 ANN vs LUT → ratio{r}.json（MAE/MAPE/R²）
   │
plot_expA.py           掃描 r ∈ {0.2,0.4,0.6,0.8} → 論文主圖
```

🎤 收尾：「整個實驗就是一條可重現的流水線：抽資料 → 遮點訓練 → 公平評估 → 掃描比例畫圖。核心結論是看 ANN 在資料越稀疏時，能不能贏過傳統查表插值。」
