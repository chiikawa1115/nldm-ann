# NLDM-ANN Session Handoff — 2026-08-15

## Session close record

- Session label: `xk7`
- Closed at: 2026-08-15 (Asia/Taipei)
- Repository-side GPU readiness: complete
- Formal training: not started
- Git commit/push: not performed
- Background process: none
- Working tree: intentionally dirty; preserve all existing modifications and untracked files
- Final automated verification: 24 tests passed; `compileall`, `git diff --check`, and
  dataset/manifest preflight passed
- CUDA verification: not run because this host has CPU-only PyTorch

### Exact resume point

The next session should not modify the research split design. First read:

1. `docs/GPU_READINESS_REPORT.md`
2. `docs/GPU_TRAINING_RUNBOOK.md`
3. `docs/RESEARCH_CONTRACT.md`
4. `docs/GOAL_DRIVEN_PLAN.md`

Then review the dirty worktree and prepare a user-approved commit/push. On the GPU host,
copy or rebuild `data/processed/asap7_nldm.parquet`, run the mandatory CUDA preflight, and
run one capped two-epoch smoke before starting the 40 formal runs. Never use capped smoke
metrics as thesis evidence.

## Material Passport

- Artifact type: experiment implementation handoff
- Project: `nldm-ann`
- Status: implementation in progress; long-running training not started
- Verification status: repository-side GPU readiness verified; actual CUDA run pending
- Workspace: `C:\Users\rudy.hsieh\codex_proj\nldm-ann`
- Intended use: resume work safely in a new Codex session

## 一句話目標

以不依賴 cell/corner ID embedding 的 ANN，使用 cell 結構與連續 PVT 特徵預測 ASAP7 NLDM delay/transition，評估 unseen cell、unseen corner，以及兩者同時 unseen 時的泛化能力。

## 已鎖定的研究設計

- 主資料集：ASAP7。
- 外部 technology validation：GF180MCU；IHP SG13G2 為選配。
- Nangate45 僅作 pilot，不作主要論文證據。
- 四組實驗：random-point、unseen-cell、unseen-corner、joint unseen cell + corner。
- Unseen-cell 以 `cell_base` 分組，同一 topology 的所有 Vt variants 必須在同一 partition。
- PVT 輸入使用 process proxy（SS=-1、TT=0、FF=+1）、voltage、temperature，不使用 corner ID。
- Cell 輸入使用 drive strength、pin capacitance、pin 數、Boolean/logic 結構、Vt/table type 等明確特徵，不使用 cell ID embedding。
- ASAP7 的 process/voltage/temperature 是耦合的，不宣稱單一 P/V/T 的獨立因果效果。
- Joint split 的 cross-quadrants 不納入主要 train/test。
- Seeds：11、23、37、53、71。
- 指標：MAE、RMSE、sMAPE、P90/P99 relative error、R²；同時報 micro-average、equal-cell macro-average、mean/SD/95% CI。
- 保留並報告 negative timing values。

## 已完成

- 研究契約與里程碑文件：
  - `docs/RESEARCH_CONTRACT.md`
  - `docs/GOAL_DRIVEN_PLAN.md`
  - `docs/M0_BASELINE_AUDIT.md`
- ASAP7 已擷取為 `data/processed/asap7_nldm.parquet`。
- 資料規模：12,073,208 points、246,392 tables、800 Vt-specific cells、200 `cell_base` topologies。
- 200 個 cell bases 均涵蓋 FF/SS/TT；資料必要欄位無缺漏，table ID 無重複。
- 14,112 個 `related_pin=CON` 指向 output pin 的無效 FA/HA arcs 已排除。
- 4,280 個 negative values 已保留。
- Dataset audit 與 metadata：
  - `data/processed/asap7_nldm.audit.json`
  - `data/processed/asap7_nldm.audit.coverage.csv`
  - `data/processed/asap7_nldm.metadata.json`
- Feature-conditioned model、split、metrics、baseline、aggregate 與 experiment 基礎模組已建立。
- 最後確認的測試結果：24 tests passed；CPU smoke、resume、evaluate、aggregate 均通過。
- Nangate45 Experiment A v2 已完成；per-table 64×64 MLP 在所有 sparsity 下均顯著差於 regular-grid bilinear LUT，僅作 negative pilot evidence。

## 精確停止點

此段原始停止點已完成並由上方 `Session close record` 取代。28 個 manifests 已生成及驗證，global streaming trainer/evaluator、resume、aggregation 與 GPU preflight 流程均已完成。

預期產生：

- 5 random-point manifests
- 5 unseen-cell manifests
- 15 joint manifests（5 seeds × 3 PVT folds）
- 3 unseen-corner LOPO manifests
- 合計 28 manifests，另有 index 與 eligible-cell catalog

目前沒有長時間訓練正在背景執行，可以安全關閉 session。

## 下一個 Session 的第一個動作

先執行：

```powershell
cd C:\Users\rudy.hsieh\codex_proj\nldm-ann
.\.venv\Scripts\python.exe scripts\build_generalization_manifests.py `
  --parquet data\processed\asap7_nldm.parquet `
  --dataset-metadata data\processed\asap7_nldm.metadata.json `
  --audit data\processed\asap7_nldm.audit.json `
  --out-dir experiments\asap7_generalization_v1\manifests
```

接著檢查：manifest 數量是否為 28、每個 seed 是否恰好 hold out 40/200 cell bases、hash、PVT folds、family/logic-class 分布與 deterministic reproducibility。之後補 compact manifest 與 assignment expansion 測試，再跑完整測試：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 後續實作順序

1. 驗證 generalization manifests。
2. 加入 deterministic split/manifest tests。
3. 完成可串流 12M rows 的 global feature-conditioned trainer/evaluator。
4. Normalization 僅能 fit training partition，避免 leakage。
5. 先做單一 seed/fold smoke run，再估算完整訓練時間。
6. 跑 ASAP7 全部 seeds/folds，輸出 per-run 與 aggregate metrics/CI。
7. 跑 ablations：no-cell、no-PVT、no-process-proxy、basic vs Boolean features、architecture variants。
8. 取得 pinned GF180 資料並進行外部驗證。
9. 加入 OpenSTA end-to-end timing validation。
10. 產生論文 figures/tables 與完整 reproducibility commands。

## 時間估計

以下為規劃估計，不是已量測 runtime；真正時間應在單一 smoke run 後更新。

- Manifest 生成、檢查與測試：約 30–90 分鐘。
- Scalable trainer/evaluator 完成與除錯：約 3–8 小時。
- 單一 seed/fold smoke run：約 10–60 分鐘，視 GPU、batch size、epochs 與 I/O 而定。
- ASAP7 主實驗全部 28 個 split runs：若單 GPU 串行，粗估 8–30 小時；有 checkpoint/early stopping 或平行硬體可縮短。
- Ablations：可能再增加 0.5–2 天運算時間。
- GF180 acquisition、schema adaptation 與 external validation：約 1–3 天工程與運算時間。
- OpenSTA 與論文圖表：約 1–3 天。

因此，做到「ASAP7 主結果」合理預估為 1–2 天；做到「ASAP7 + ablations + GF180 + OpenSTA + publication-ready artifacts」較合理是數天到約一週。這些時間不代表 session 必須持續開啟；每階段都應輸出 checkpoint 與結果檔，可分 session 接續。

## Resume Prompt

新 session 可直接貼：

> 請先完整閱讀 `docs/SESSION_HANDOFF_2026-08-15.md`、`docs/GPU_READINESS_REPORT.md` 與 `docs/GPU_TRAINING_RUNBOOK.md`。保留 dirty worktree；經使用者同意後才 commit/push。GPU 主機先跑 CUDA preflight 與 capped 2-epoch smoke，通過後才啟動 40 個正式 runs。

## 注意事項

- 工作樹目前有許多未 commit 修改與新增檔案，均視為使用者工作，不可 reset、checkout 或清除。
- `scripts/build_generalization_manifests.py` 是尚未驗證的新檔，不能把預期輸出寫成已完成結果。
- 不應因 random-point 表現良好而宣稱 unseen-cell/corner 泛化成立。
- Fully unseen cell/corner 沒有 characterization tables 時，傳統 LUT baseline 並無直接定義；論文需清楚說明比較邊界。

## 2026-08-15 GPU-readiness update

Repository-side GPU preparation is now complete. See `docs/GPU_READINESS_REPORT.md` and
`docs/GPU_TRAINING_RUNBOOK.md`. The 28 split contracts expand to 40 formal model runs.
Automated tests are at 24 passed; real ASAP7 capped CPU train/evaluate/aggregate and resume
smokes passed. CUDA remains unverified on this CPU-only host and is the mandatory next gate.
