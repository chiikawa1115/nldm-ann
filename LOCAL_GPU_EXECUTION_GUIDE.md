# NLDM-ANN 本機 GPU 執行指南與進度追蹤 (Local GPU Execution Guide)

> **適用主機**：本機 Windows (GTX 1070 8GB, CUDA 12.6, Driver 560.94)  
> **專案路徑**：`C:\Users\User\claude_project\nldm-ann`  
> **目前分支**：`plan-v4`  
> **參考文檔**：`docs/PLAN_v4_GPU_HOST_RUNBOOK.md`、`docs/PLAN_v4_HIERARCHICAL_MERGE_SIGNOFF.md`

---

## 📌 進度檢查清單 (Progress Checklist)

- [x] **Step 0: 程式碼準備與分支確認** (`git checkout plan-v4`)
- [x] **Step 1: 建立 Python 3.12 虛擬環境** (`nldm-v4` @ `F:\ana\envs\nldm-v4`)
- [x] **Step 2: 安裝依賴與 PyTorch** (`requirements-plan-v4.txt` + `torch 2.14.1+cu126`, GTX 1070 CUDA 運算測試成功)
- [x] **Step 3: 從公開 ORFS 下載並抽取 Parquet 資料集** (`asap7_nldm.parquet`, 12,073,208 rows 完成)
- [x] **Step 4: 驗證資料集內容指紋** (`dataset_fingerprint.py check` -> `status: PASS`, `verified_by: file_sha256`)
- [x] **Step 5: 產生特徵快取** (`scripts/plan_v4_w2.py` -> 69 個特徵 `.npy`, `matches_other_machine: true`)
- [x] **Step 6: 執行正式 Preflight 驗證** (`scripts/plan_v4_preflight.py` -> `status: PASS`, 產出 `preflight_DESKTOP-0P2BF8K.json`)
- [進行中] **Step 7: 全局泛化全量訓練 (40 Runs, 50 Epochs)** (`src/run_generalization.py` 正在背景全速執行)

---

## 🏃 當前執行中：全量泛化訓練 (40 Runs)

### 執行指令
```powershell
F:\ana\envs\nldm-v4\python.exe -u src/run_generalization.py `
  --parquet data/processed/asap7_nldm.parquet `
  --manifest-dir experiments/asap7_generalization_v1/manifests `
  --out-dir experiments/asap7_generalization_v1/runs `
  --device cuda --epochs 50 --patience 8 `
  --batch-rows 250000 --micro-batch 8192
```

### 隨時查看進度
```powershell
# 查看已經跑完幾個 run
(Get-ChildItem experiments\asap7_generalization_v1\runs\*\model.pt).Count
```

### 中斷與接關 (Resume)
* 若關機或中斷，直接在同目錄再次執行上述指令，程式會自動跳過已完成的 runs，從上次中斷處自動接續訓練。

## 🚀 重開後如何接續執行 (How to Resume)

若對話中斷或重新連線，請依序檢查目前完成到哪一步：

### 1. 啟動環境
```powershell
cd C:\Users\User\claude_project\nldm-ann
conda activate nldm-v4
```

### 2. 環境安裝步驟
```powershell
# 建立 conda 環境（只需一次）
conda create -n nldm-v4 python=3.12 -y
conda activate nldm-v4

# 安裝固定版本依賴
pip install -r requirements-plan-v4.txt

# 安裝 PyTorch（先試 cu126；若 preflight 報 sm_61 錯誤則改裝 cu118）
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 3. 下載 ORFS 資料與抽取 Parquet
```powershell
# 回到外層目錄 sparse clone ORFS 的 NLDM 庫
cd C:\Users\User\claude_project
git clone --filter=blob:none --no-checkout https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git ORFS
cd ORFS
git sparse-checkout set flow/platforms/asap7/lib/NLDM
git checkout da37dce1cef69bb75efbc2dc18ed95a29651295b

# 回到專案抽取資料
cd C:\Users\User\claude_project\nldm-ann
python scripts/extract_asap7_dataset.py --manifest data/asap7_sources.json --repo ../ORFS --out data/processed/asap7_nldm.parquet

# 驗證指紋 (需看到 status: PASS)
python scripts/dataset_fingerprint.py check
```

### 4. 產生特徵快取
```powershell
python scripts/plan_v4_w2.py
```
*(約需 10 分鐘，佔用約 6GB RAM，生成約 1.6GB 的 `.npy` 快取)*

### 5. Preflight 驗證
```powershell
python scripts/plan_v4_preflight.py
```
* 檢查四項：`dataset` -> `manifests` -> `cache` -> `cuda`
* 成功後會輸出 `experiments/plan_v4/preflight_<主機名>.json`

---

## ⚠️ 常見問題與注意事項
1. **GTX 1070 相容性**：GTX 1070 為 Pascal 架構（Compute Capability 6.1）。若 PyTorch cu126 無法在 GPU 運算，改裝 `cu118`。
2. **Parquet 指紋規則**：Windows 上檔案 SHA-256 可能因壓縮或 Arrow 版本稍有差異，以 `dataset_fingerprint.py check`（比對排序後的欄位內容 Hash）回傳 `PASS` 為準。
3. **Git 保護**：快取 `.npy` 與抽取出的 `.parquet` 已被 `.gitignore` 排除，嚴禁手動 commit 大檔案。
