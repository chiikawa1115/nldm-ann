# PLAN v4：GPU 主機操作手冊（1070 主機 git clone 後照做）

日期：2026-10-18。適用 branch：`plan-v4`。
取代舊版 `GPU_TRAINING_RUNBOOK.md`（那份是 8/15 global engine 的流程，PLAN v4 不使用）。

## 0. 為什麼要這樣做（先讀）

開發機是公司筆電，**任何資料檔都不從筆電拿出來**。GPU 主機只靠兩樣東西：

1. **git 上的程式碼與小檔案**：程式、E1–E4 manifests、L0 抽樣清單、資料的參考指紋。
2. **公開的原始資料**：OpenROAD-flow-scripts（ORFS）裡的 ASAP7 `.lib.gz`。

GPU 主機自己把 `.lib.gz` 轉成 parquet，再自己產生特徵快取。每一步都有 hash 檢查，
保證跟開發機算出來的資料**內容完全相同**：

| 東西 | 大小 | 從哪裡來 | 怎麼確認跟開發機一樣 |
|---|---|---|---|
| 程式、manifests、L0 清單 | < 1 MB | `git clone` | manifest 檔 hash（preflight 檢查） |
| ASAP7 原始 `.lib.gz`（60 個） | 約 250 MB | ORFS 公開 repo，固定 commit | 每個檔的 SHA-256 寫在 `data/asap7_sources.json` |
| `asap7_nldm.parquet` | 65 MB | GPU 主機自己抽取 | **內容指紋** `data/asap7_nldm.reference.json` |
| 特徵快取 `.npy` | 1.6 GB | GPU 主機自己產生 | 矩陣 hash 對照 `experiments/plan_v4/feature_cache_reference.json` |

**為什麼 parquet 比對「內容」而不是「檔案」**：parquet 檔案的 bytes 會因 pyarrow 版本、
壓縮參數而不同，即使資料一模一樣。所以參考指紋是把所有 row 排序、浮點數取到小數 9 位後
逐欄算 SHA-256。檔案 hash 相同直接通過；不同時改比內容，內容不同就停，並列出是哪幾欄不同。

**為什麼不用抽 table、也不用重做 split**：「哪 2,000 張 table 是 L0」「每份 E1–E4 怎麼切」
都已經寫成 manifests 進 git，兩台用的是同一份。GPU 主機只是照著讀。

## 1. 取得程式碼

```bash
git clone https://github.com/chiikawa1115/nldm-ann.git
cd nldm-ann
git checkout plan-v4
```

`.gitattributes` 已設定 manifests 與參考 JSON 不做換行轉換（Windows 的 CRLF 會改變 bytes，
導致 hash 對不上）。如果 preflight 回報 manifest hash mismatch，先確認是在 `.gitattributes`
加入之後 clone 的；必要時 `git rm --cached -r experiments/plan_v4 && git checkout -- experiments/plan_v4`。

## 2. 建 Python 環境（重點：PyTorch 的 CUDA 版本）

```bash
conda create -n nldm-v4 python=3.12 -y
conda activate nldm-v4
pip install -r requirements-plan-v4.txt
```

`requirements-plan-v4.txt` 鎖住了 numpy／pandas／pyarrow 版本，跟開發機相同，
這樣特徵快取的矩陣 hash 才能逐 byte 比對。

**PyTorch 另外裝**。GTX 1070 是 Pascal 架構（`sm_61`），較新的 PyTorch CUDA wheel
可能已經不含 Pascal 的 kernel。依序嘗試，**以第 5 步 preflight 的結果為準**：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # 先試這個
# preflight 的 cuda 步驟失敗、訊息寫 "no kernels for sm_61" 時，改裝較舊的：
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

`nvidia-smi` 右上角的 "CUDA Version" 是驅動支援的上限，wheel 的 CUDA 版本不能高於它。

## 3. 重建 parquet（從公開 ORFS，不從筆電複製）

```bash
# 只抓需要的 commit；ORFS 很大，用 sparse checkout 只取 ASAP7 NLDM lib
git clone --filter=blob:none --no-checkout \
  https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git ../ORFS
cd ../ORFS
git sparse-checkout set flow/platforms/asap7/lib/NLDM
git checkout da37dce1cef69bb75efbc2dc18ed95a29651295b
cd ../nldm-ann

python scripts/extract_asap7_dataset.py \
  --manifest data/asap7_sources.json \
  --repo ../ORFS \
  --out data/processed/asap7_nldm.parquet
```

- 抽取前會逐檔比對 `.lib.gz` 的 SHA-256，任何一個不符就停。
- 可中斷：每個 lib 抽完會存成 `data/processed/asap7_nldm_parts/` 下的一個 part，重跑會跳過已完成的。
  開發機 8/15 抽 60 個檔約花 6 小時（筆電 CPU），桌機通常較快。
- 抽完會覆寫 `data/processed/asap7_nldm.metadata.json`，裡面的 `parquet_sha256` 變成
  **這台的檔案 hash**，跟開發機不同是正常的。這個檔案有進 git，**不要 commit 這個變動**
  （`git checkout -- data/processed/asap7_nldm.metadata.json` 還原即可，程式只認
  `data/asap7_nldm.reference.json`）。

確認內容一致：

```bash
python scripts/dataset_fingerprint.py check
```

要看到 `"status": "PASS"`。`"verified_by": "content_sha256"` 表示檔案 bytes 不同但內容相同，
這是預期結果。失敗訊息會列出不一致的欄位，**不一致就不要往下做**，回報給開發機這邊。

## 4. 產生特徵快取

```bash
python scripts/plan_v4_w2.py
```

- 讀 parquet → 建 69 個特徵 → 寫到 `data/cache/plan_v4/features_<資料hash>_<程式hash>.npy`。
- 同時會重建 manifests 並檢查分組沒有交叉。manifests 由程式決定，重建結果應與 git 上的
  逐 byte 相同；`git status` 若顯示 `experiments/plan_v4/manifests/` 有變動，**停下來回報**，
  不要 commit。（`manifest_index.json` 只有 provenance 欄位會不同，屬正常。）
- 約 10 分鐘、需約 6 GB RAM。
- 檔名裡的「程式 hash」是特徵程式碼的 SHA-256（換行已統一），所以只要程式碼相同，
  兩台的檔名就相同。

## 5. Preflight（正式訓練前必跑）

```bash
python scripts/plan_v4_preflight.py
```

依序檢查四項，任何一項失敗就停：

| 步驟 | 檢查內容 |
|---|---|
| dataset | parquet 與 `data/asap7_nldm.reference.json` 一致（檔案或內容） |
| manifests | 137 份 E1–E4 manifests 與 L0 清單的 hash |
| cache | 特徵矩陣 hash 與自己的 JSON 一致，**且與開發機的矩陣 hash 一致** |
| cuda | 不只看 `is_available()`：在 GPU 上做 matmul 與 5 步 MLP 訓練，結果要與 CPU 相符 |

結果寫到 `experiments/plan_v4/preflight_<主機名>.json`，包含 GPU 型號、`sm_xx`、
PyTorch 與 CUDA 版本。**這個檔案請 commit 並 push**，作為 provenance 的一部分。

cache 步驟若顯示 matrix 與開發機不同：通常是 numpy/pandas 版本不同造成浮點數最後一位不同。
先確認 `pip list` 與 `requirements-plan-v4.txt` 相同；仍不同就回報，不要開始訓練。

## 6. 結果怎麼回來

| 產出 | 進 git？ |
|---|---|
| `preflight_<主機>.json` | 是 |
| 每次 run 的 metrics（CSV/JSON 彙整） | 是 |
| checkpoint（`.pt`） | 否（`.gitignore` 已排除） |
| 特徵快取、parquet | 否 |

GPU 主機只 push 小檔案；開發機 `git pull` 後做分析與寫作。

## 7. 常見問題

| 狀況 | 原因／處理 |
|---|---|
| `source hash mismatch` | ORFS commit 不對，確認是 `da37dce`；或 `.lib.gz` 下載不完整 |
| `content mismatch in columns [...]` | 抽取程式或 liberty-parser 版本不同；比對 `pip list` |
| `no kernels for sm_61` | PyTorch wheel 不支援 Pascal，改裝 cu118 |
| `GPU matmul result differs from CPU` | 驅動或 wheel 有問題，不要訓練 |
| manifest hash mismatch | 換行被轉成 CRLF，見第 1 節 |
