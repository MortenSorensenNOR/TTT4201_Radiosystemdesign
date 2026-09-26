"""Read ADS datasets into pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load(ds_path: str | Path) -> dict[str, pd.DataFrame]:
    """All varblocks of a .ds file, as flat DataFrames (sweep indices become columns)."""
    import keysight.ads.dataset as dataset

    ds = dataset.open(Path(ds_path))
    return {name: ds[name].to_dataframe().reset_index() for name in ds.varblock_names}


def find(blocks: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """The first varblock containing ``column``, e.g. a MeasEqn result."""
    for df in blocks.values():
        if column in df.columns:
            return df
    raise KeyError(f"no varblock has a column {column!r}")


def at_harmonic(df: pd.DataFrame, column: str, f0: pd.Series | float, k: int) -> pd.Series:
    """Values of ``column`` at frequency k*f0.

    ``f0`` is a scalar, or a Series aligned with ``df`` when the fundamental
    changes between sweep points.
    """
    k_here = (df["freq"] / f0).round()
    return df.loc[k_here == k, column]
