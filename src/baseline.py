"""Reference NLDM interpolation used by the paper experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def bilinear_predict(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """Predict from a complete rectangular NLDM grid without extrapolation."""
    slews = np.sort(train["slew"].unique())
    loads = np.sort(train["load"].unique())
    expected = pd.MultiIndex.from_product([slews, loads], names=["slew", "load"])
    values = train.set_index(["slew", "load"])["value"]
    if values.index.has_duplicates or len(values) != len(expected):
        raise ValueError("training points are not a unique complete rectangular grid")
    values = values.reindex(expected)
    if values.isna().any():
        raise ValueError("training grid contains missing points")
    interpolator = RegularGridInterpolator(
        (slews, loads), values.to_numpy().reshape(len(slews), len(loads)),
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    return interpolator(query[["slew", "load"]].to_numpy(dtype=float))
