# Global ANN / Ridge paired accuracy comparison

Under identical input information and test data, does global ANN reduce prediction error relative to global Ridge, and by how much?

Completed planned runs: 10/10.

Target scope: mixed delay + transition, with the frozen 42 input features.
Ridge uses fit-only scaling, an unpenalized intercept and the frozen six-lambda grid; validation MSE selects lambda without refitting on validation.
Positive error delta (Ridge - ANN) means lower ANN error. R2 uses ANN - Ridge.

| Mode | Pairs | ANN macro MAE | Ridge macro MAE | Delta | 95% CI |
|---|---:|---:|---:|---:|---|
| random_point | 5/5 | 13.0563 | 67.8212 | 54.7649 | [54.0725, 55.4572] |
| unseen_cell | 5/5 | 22.334 | 77.1179 | 54.7839 | [49.2311, 60.3367] |

## Coverage and limitations

- Completed pairs with ANN provenance pending: 0. Pending comparisons are provisional, not fully verified reproduction.
- Missing, failed, or mismatched runs remain listed in comparison_status.json.
- Modes are summarized separately; repeated seeds have overlapping source data.
- Five-seed t intervals are descriptive and rely on assumptions; they do not establish cross-library generalization.
- Formal two-sided primary p-values and Holm correction are emitted only when both primary mode groups are complete and provenance-matched.
- Both modes require five pairs with positive finite paired variance; degenerate or incomplete families remain descriptive only.
- See paired_metrics.csv for each seed and summary_by_mode.csv for relative gains, SD, confidence intervals and eligible p-values. Relative-error deltas use percentage points.
- Review P90/P99 alongside mean error; a lower MAE alone does not establish a tail-error benefit.
- Checkpoint metadata and committed source hashes cannot independently attest absence of uncommitted ANN training edits.
- Figures updated by this aggregation: True; if false, existing images may be stale.
- Hardware differs; runtime is not a primary comparison or a model speedup claim.
