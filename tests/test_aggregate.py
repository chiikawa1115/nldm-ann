import json
from pathlib import Path

import pandas as pd
import pytest

from aggregate import METRICS, load_run_records, paired_differences, summarize_records, write_aggregate


def result_payload(seed):
    def metrics(offset):
        return {name: float(seed + offset + i) for i, name in enumerate(METRICS)}
    return {
        "holdout_ratio": 0.5,
        "overall": {"MLP": metrics(0), "LUT": metrics(2)},
        "per_table_type": {
            "cell_rise": {"n": 10, "MLP": metrics(1), "LUT": metrics(3)}
        },
    }


def test_load_and_summarize_repeated_seed_results():
    runtime = Path(__file__).parent / "runtime"
    records = []
    for seed in [11, 23, 37, 53, 71]:
        path = runtime / f"seed_{seed}.json"
        path.write_text(json.dumps(result_payload(seed)), encoding="utf-8")
        records.extend(load_run_records(path, seed))
    frame = pd.DataFrame(records)
    summary = summarize_records(frame)
    row = summary[
        (summary.scope == "overall") & (summary.method == "MLP") & (summary.metric == "RMSE")
    ].iloc[0]
    assert row.n_seeds == 5
    assert row.ci_low < row["mean"] < row.ci_high
    paired = paired_differences(frame)
    delta = paired[
        (paired.scope == "overall") & (paired.metric == "RMSE")
    ].iloc[0]
    assert delta["mean"] == pytest.approx(-2.0)

    write_aggregate(frame, runtime / "aggregate")
    for name in ["per_seed_metrics.csv", "summary_ci.csv",
                 "paired_difference_ci.csv", "aggregate_metadata.json"]:
        assert (runtime / "aggregate" / name).is_file()


def test_summary_rejects_single_seed_as_confidence_evidence():
    frame = pd.DataFrame([{
        "seed": 1, "holdout_ratio": 0.5, "scope": "overall",
        "method": "MLP", "metric": "RMSE", "value": 1.0,
    }])
    row = summarize_records(frame).iloc[0]
    assert row.n_seeds == 1
    assert pd.isna(row.ci_low) and pd.isna(row.ci_high)
