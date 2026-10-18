"""Inference-available cell and PVT features for generalization experiments."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LOGIC_CLASSES = (
    "INV", "BUF", "AND", "NAND", "OR", "NOR", "XOR", "XNOR",
    "AO", "AOI", "OA", "OAI", "MUX", "DFF", "LATCH", "CLOCK", "OTHER",
)
VT_CLASSES = ("SLVT", "LVT", "RVT", "SRAM", "UNKNOWN")
TABLE_TYPES = ("cell_rise", "cell_fall", "rise_transition", "fall_transition")


def infer_logic_class(cell_name: str) -> str:
    token = re.split(r"[xX_]", cell_name, maxsplit=1)[0].upper()
    ordered = ("XNOR", "XOR", "NAND", "NOR", "AOI", "OAI", "AND", "INV",
               "BUF", "MUX", "DFF", "LATCH", "AO", "OA")
    for category in ordered:
        if token.startswith(category):
            return category
    if token.startswith(("CLK", "CK")):
        return "CLOCK"
    return "OTHER"


def infer_cell_base(cell_name: str) -> str:
    """Remove ASAP7 Vt suffix so all Vt variants share one topology group."""
    return re.sub(r"_ASAP7_75t_(?:L|R|SL|SRAM)$", "_ASAP7_75t", cell_name)


def boolean_function_features(expression: str) -> dict[str, float]:
    expression = str(expression or "")
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    return {
        "function_input_count": float(len(set(identifiers))),
        "function_not_count": float(expression.count("!") + expression.count("~")),
        "function_and_count": float(expression.count("&") + expression.count("*")),
        "function_or_count": float(expression.count("|") + expression.count("+")),
        "function_xor_count": float(expression.count("^")),
    }


def infer_drive_strength(cell_name: str) -> float:
    match = re.search(r"[xX](\d+(?:[pP]\d+)?)", cell_name)
    if match:
        return float(match.group(1).lower().replace("p", "."))
    fractional = re.search(r"[xX][pP](\d+)", cell_name)
    return float(f"0.{fractional.group(1)}") if fractional else float("nan")


def infer_vt_class(*labels: str) -> str:
    text = "_".join(labels).upper()
    for vt in ("SLVT", "LVT", "RVT", "SRAM"):
        if re.search(rf"(?:^|[_\W]){vt}(?:$|[_\W])", text):
            return vt
    return "UNKNOWN"


def process_code(corner: str) -> float:
    """Ordered process proxy; its -1/0/+1 assumption must be reported."""
    mapping = {"SS": -1.0, "TT": 0.0, "FF": 1.0}
    return mapping.get(str(corner).upper(), float("nan"))


def feature_columns() -> list[str]:
    continuous = [
        "log_slew", "log_load", "process_code", "nom_voltage_v",
        "nom_temperature_c", "drive_strength", "related_pin_cap_ff",
        "drive_strength_missing", "related_pin_cap_missing",
        "n_input_pins", "n_output_pins",
        "function_input_count", "function_not_count", "function_and_count",
        "function_or_count", "function_xor_count",
    ]
    one_hot = (
        [f"logic_{name}" for name in LOGIC_CLASSES]
        + [f"vt_{name}" for name in VT_CLASSES]
        + [f"table_{name}" for name in TABLE_TYPES]
    )
    return continuous + one_hot


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a fixed-vocabulary feature matrix with no cell/corner ID embedding."""
    required = {
        "log_slew", "log_load", "corner", "nom_voltage_v", "nom_temperature_c",
        "drive_strength", "related_pin_cap_ff", "n_input_pins", "n_output_pins",
        "logic_class", "vt_class", "table_type",
        "logic_function",
    }
    if missing := required - set(df.columns):
        raise ValueError(f"missing generalization features: {sorted(missing)}")
    out = pd.DataFrame(index=df.index)
    for name in ["log_slew", "log_load", "nom_voltage_v", "nom_temperature_c",
                 "drive_strength", "related_pin_cap_ff", "n_input_pins", "n_output_pins"]:
        out[name] = pd.to_numeric(df[name], errors="coerce")
    out["drive_strength_missing"] = out["drive_strength"].isna().astype(np.float32)
    out["related_pin_cap_missing"] = out["related_pin_cap_ff"].isna().astype(np.float32)
    out[["drive_strength", "related_pin_cap_ff"]] = out[
        ["drive_strength", "related_pin_cap_ff"]
    ].fillna(0.0)
    out["process_code"] = df["corner"].map(process_code)
    function_frame = pd.DataFrame(
        [boolean_function_features(value) for value in df["logic_function"]], index=df.index
    )
    for name in function_frame:
        out[name] = function_frame[name].astype(np.float32)
    for category in LOGIC_CLASSES:
        out[f"logic_{category}"] = (df["logic_class"] == category).astype(np.float32)
    for category in VT_CLASSES:
        out[f"vt_{category}"] = (df["vt_class"] == category).astype(np.float32)
    for category in TABLE_TYPES:
        out[f"table_{category}"] = (df["table_type"] == category).astype(np.float32)
    out = out[feature_columns()]
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"non-inference-ready features contain NaN: {bad}")
    return out.astype(np.float32)
