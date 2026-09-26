"""Smith charts of a solver result: f0, 2f0, 3f0 across frequency.

Left: ZLi, the loads the intrinsic drain current source sees (with the class J
continuum from the ADS utility in grey). Right: the external loads at the S1P
plane that produce them.

    python_utils/adsenv.sh python -m adsutil.plot_loads python_utils/results/continuum_noline.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classj import Continuum

Z0 = 50.0
# categorical slots 1-3 of the reference palette (the three that validate all-pairs)
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
NAMES = {1: "f0", 2: "2f0", 3: "3f0"}
GRID = "#d6d5d0"
TEXT = "#52514e"


def gamma(z):
    z = np.asarray(z, complex) / Z0
    return (z - 1) / (z + 1)


def smith_axes(ax, title: str) -> None:
    """Draw a recessive impedance Smith grid on ``ax``."""
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=TEXT, lw=1)
    for r in (0.2, 0.5, 1, 2, 5):
        c, rad = r / (1 + r), 1 / (1 + r)
        ax.plot(c + rad * np.cos(t), rad * np.sin(t), color=GRID, lw=0.7)
    x_line = np.linspace(0, 1e3, 4000)
    for x in (0.2, 0.5, 1, 2, 5):
        for s in (1, -1):
            g = gamma(Z0 * (x_line + 1j * s * x))
            ax.plot(g.real, g.imag, color=GRID, lw=0.7)
    ax.plot([-1, 1], [0, 0], color=GRID, lw=0.7)
    for r in (0.2, 0.5, 1, 2):
        ax.text(r / (1 + r) * 2 - 1 + 0.01, 0.015, f"{r * Z0:g}", fontsize=7, color=TEXT)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color="#0b0b0b", pad=12)


def _rim_path(g):
    """Connect points on the rim (pure reactances) along the circle, not by chords."""
    th = np.angle(g)
    parts = []
    for a, b in zip(th[:-1], th[1:]):
        d = (b - a + np.pi) % (2 * np.pi) - np.pi      # shorter way round
        parts.append(np.exp(1j * (a + np.linspace(0, d, 30))))
    return np.concatenate(parts)


def trace(ax, z, f_ghz, n: int, label_ends: bool = True, offsets=((6, 4), (6, 4))) -> None:
    g = gamma(z)
    color = COLORS[n]
    on_rim = np.all(np.abs(np.abs(g) - 1) < 1e-3)
    path = _rim_path(g) if on_rim and len(g) > 1 else g
    ax.plot(path.real, path.imag, color=color, lw=2, zorder=3)
    ax.scatter(g.real, g.imag, s=22, color=color, edgecolor="white", linewidth=1, zorder=4,
               label=NAMES[n])
    if label_ends:
        for i, off in zip((0, -1), offsets):
            ax.annotate(f"{NAMES[n]} {f_ghz[i]:.1f}", (g.real[i], g.imag[i]), xytext=off,
                        textcoords="offset points", fontsize=8, color=TEXT, zorder=5)


def plot(csv: Path, out: Path | None = None) -> Path:
    df = pd.read_csv(csv)
    for col in [c for c in df.columns if c.startswith("ZLi")]:
        df[col] = df[col].map(lambda s: complex(str(s).replace(" ", "")))
    f = df["f0_GHz"].to_numpy()

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.3), facecolor="white")
    smith_axes(left, "Intrinsic drain, ZLi")
    smith_axes(right, "External loads at the S1P plane")

    # class J continuum from the utility: f0 arc and the 2f0 reactance range, alpha = -1..1
    c = Continuum()
    a = np.linspace(-1, 1, 101)
    g1, g2 = gamma(c.R + 1j * a * c.X1_max), gamma(-1j * a * c.X2_max + 1e-9)
    left.plot(g1.real, g1.imag, color="#9b9a95", lw=1.2, ls="--", zorder=2, label="continuum (f0 / 2f0)")
    left.plot(g2.real, g2.imag, color="#9b9a95", lw=1.2, ls="--", zorder=2)

    trace(left, df["ZLi_f0"].to_numpy(), f, 1)
    trace(left, df["ZLi_2f0"].to_numpy(), f, 2, offsets=((-44, -12), (6, 4)))
    trace(left, df["ZLi_3f0"].to_numpy(), f, 3, offsets=((10, 6), (10, -10)))

    trace(right, df["RL"].to_numpy() + 1j * df["XL"].to_numpy(), f, 1)
    if "X2" in df:
        x2 = df["X2"].to_numpy()
        fixed = np.ptp(x2) < 1
        trace(right, 1j * np.where(np.abs(x2) > 1e5, -1e5, x2), f, 2, label_ends=not fixed)
        if fixed:
            right.annotate("2f0 held open", (1, 0), xytext=(-70, -16), textcoords="offset points",
                           fontsize=8, color=TEXT)
    if "X3" in df:
        trace(right, 1j * df["X3"].to_numpy(), f, 3)

    for ax in (left, right):
        ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.text(0.5, 0.02, f"{f[0]:.1f}-{f[-1]:.1f} GHz in {f[1] - f[0]:.1f} GHz steps; "
             "labels mark the band edges (GHz). Z0 = 50 ohm.", ha="center", fontsize=8, color=TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = out or csv.with_suffix(".png")
    fig.savefig(out, dpi=150)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(plot(Path(arg)))
