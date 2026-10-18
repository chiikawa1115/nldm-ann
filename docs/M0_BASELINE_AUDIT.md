# M0 Baseline Audit

Status: complete (pilot only; superseded as the main research question)

## Findings

1. Legacy Experiment A randomly removes individual grid points. The remaining
   samples are generally not a rectangular NLDM grid.
2. `LinearNDInterpolator` performs triangulated scattered-data interpolation;
   it is not the regular-grid bilinear interpolation used by an NLDM table.
3. The legacy nearest-neighbor fallback mixes extrapolation into the reported
   interpolation metric without labeling it.
4. The legacy table key omits library, corner, timing type/sense, and template.
   Combined PVT datasets can therefore merge distinct Liberty tables.
5. MAPE and maximum relative error are unstable for timing values near zero.
   Existing results show this failure mode and must not be used alone for claims.
6. Legacy result files do not include the source dataset, immutable split
   manifest, checkpoint, environment provenance, or repeated seeds. They cannot
   independently reproduce the committed summaries.

## Implemented controls

- `regular_subgrid_v1` retains complete slew/load axis subgrids including domain
  boundaries and emits an immutable split manifest.
- The full Liberty timing-table identity includes library/file, PVT corner,
  cell/pins, timing metadata, condition, table type, and template.
- The reference baseline uses `RegularGridInterpolator` and returns NaN outside
  its interpolation domain. It rejects incomplete or duplicate grids.
- Metrics now include MAE, RMSE, sMAPE, P50/P90/P99 relative error, and R².
- Dataset and manifest SHA-256 values are stored with environment and git
  provenance. Evaluation rejects mismatches and legacy checkpoints.

## Verification evidence

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current result: 6 tests passed. Tests cover deterministic rectangular splits,
boundary retention, cross-corner isolation, exact bilinear reconstruction,
incomplete-grid rejection, no extrapolation fallback, stable zero handling, and
order-independent dataset hashing.

## Gate evidence completed

- Nangate45 source hash and canonical ps/fF conversion are recorded.
- Experiment A v2 completed 4 realizable grid ratios × 5 fixed seeds.
- The configuration, per-seed metrics, paired confidence intervals, summary,
  and generated figure are archived under `experiments/expA_v2`.
- Liberty `Timing_7_7` declares variable_1=`input_net_transition` and
  variable_2=`total_output_net_capacitance`; parser orientation was checked.

The pilot found that the per-table 64×64 MLP is significantly worse than the
bilinear LUT at every tested sparsity. This negative result motivated the main
feature-conditioned generalization RQ; it must not be reframed as ANN superiority.
