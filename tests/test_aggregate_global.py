import pandas as pd

from aggregate_global import summarize


def test_global_summary_reports_sample_sd_and_t_interval():
    records = pd.DataFrame({
        "mode": ["unseen_cell"] * 3,
        "scope": ["micro"] * 3,
        "MAE": [1.0, 2.0, 3.0], "RMSE": [2.0, 3.0, 4.0],
        "sMAPE_pct": [3.0, 4.0, 5.0], "P90RE_pct": [4.0, 5.0, 6.0],
        "P99RE_pct": [5.0, 6.0, 7.0], "R2": [0.7, 0.8, 0.9],
    })
    result = summarize(records, ["mode", "scope"])
    mae = result.query("metric == 'MAE'").iloc[0]
    assert mae["n_runs"] == 3
    assert mae["mean"] == 2.0
    assert mae["standard_deviation"] == 1.0
    assert mae["ci95_low"] < mae["mean"] < mae["ci95_high"]
