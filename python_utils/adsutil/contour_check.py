"""Step 3: does the clipping contour predict what the real CG2H40010F model does?

At one frequency: set the intrinsic fundamental to Z1i = R_L * z1, then sweep the 2f0
load at the drain pin around the Smith chart rim. For every point the predicted margin
(clipping.margin with the simulated Z1i, Z2i) is compared with the simulated drain
efficiency, output power and intrinsic waveforms.

The pin-to-intrinsic map at each harmonic is fitted as a bilinear (Moebius) map from
three runs (classj.fit_moebius); the smoke run showed Z1i hardly moves with drive.

    python_utils/adsenv.sh python -m adsutil.contour_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import PACKAGE_DIR
from .classj import fit_moebius, invert_moebius
from .clipping import margin
from .loadbench import Bench, gamma

RESULTS = PACKAGE_DIR.parent / "results"
F0 = 3.0e9
R_L = 27.9                      # s10 load line (Keysight class J utility, alpha = 1)
Z1_NORM = 0.9 + 0.5j
PIN_FIT = 32.0
N_PHASE = 36


def zl(g):
    return 50 * (1 + g) / (1 - g)


def fit_maps():
    """Pin -> intrinsic maps at f0 and 2f0 from five runs (shared base point)."""
    G1 = np.array([gamma(13 + 6j), gamma(18 + 6j), gamma(13 + 11j), gamma(13 + 6j), gamma(13 + 6j)])
    G2 = np.array([1.0, 1.0, 1.0, np.exp(1j * 2.0), np.exp(-1j * 2.0)])
    b = Bench()
    b.set_pin_sweep(20, PIN_FIT, 4)
    b.set_points([F0] * 5, {1: G1, 2: G2})
    r = b.run("cc_fit")
    r = r[r["Pin"] == PIN_FIT].sort_values("pt")
    z1i, z2i = r["Z1i"].to_numpy(), r["Z2i"].to_numpy()
    f1 = fit_moebius(zl(G1[[0, 1, 2]])[:, None], z1i[[0, 1, 2]][:, None])
    f2 = fit_moebius(zl(G2[[0, 3, 4]])[:, None], z2i[[0, 3, 4]][:, None])
    return f1, f2


def main():
    (a, b_, c), _ = fit_maps()
    z1_ext = invert_moebius(a, b_, c, R_L * Z1_NORM)[0]
    print(f"Z1i target {R_L * Z1_NORM:.2f} -> load at pin {z1_ext:.2f}")

    phases = np.linspace(-np.pi, np.pi, N_PHASE, endpoint=False)
    bench = Bench()
    bench.set_pin_sweep(20, 34, 2)
    bench.set_points([F0] * N_PHASE, {1: [gamma(z1_ext)] * N_PHASE, 2: np.exp(1j * phases)})
    r = bench.run("cc_sweep")
    r["phi2"] = phases[(r["pt"].round().astype(int) - 1).to_numpy()]
    r["margin"] = margin(r["Z1i"].to_numpy() / R_L, r["Z2i"].to_numpy() / R_L)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "contour_check_3GHz.csv"
    r.to_csv(out, index=False)
    print(out)
    pd.set_option("display.width", 250)
    cols = ["phi2", "Pin", "Z1i", "Z2i", "margin", "Pout_dBm", "DE", "vdsi_min", "idi_max"]
    print(r[r["Pin"] == 32][cols].round(3).to_string(index=False))


def plot(csv=RESULTS / "contour_check_3GHz.csv", pin_main: float = 32.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .clipping import contour, safe_region
    from .plot_loads import COLORS, GRID, TEXT, gamma as g50, smith_axes

    r = pd.read_csv(csv)
    for col in ("Z1i", "Z2i", "Z3i"):
        r[col] = r[col].map(lambda s: complex(str(s).replace(" ", "")))
    m = r[r["Pin"] == pin_main].sort_values("phi2")
    z1 = m["Z1i"].mean() / R_L

    fig, (ax_s, ax_de, ax_v) = plt.subplots(1, 3, figsize=(16, 5.4), facecolor="white",
                                            gridspec_kw={"width_ratios": [1.05, 1.2, 1]})
    # (a) Smith chart: predicted safe region for the simulated Z1i, and the simulated Z2i
    smith_axes(ax_s, f"2f0 at the intrinsic drain, Pin = {pin_main:g} dBm")
    gx = np.linspace(-1.08, 1.08, 241)
    G = gx[None, :] + 1j * gx[::-1, None]
    inside = np.abs(G) <= 1
    safe = safe_region(z1, (50 * (1 + G) / (1 - G)) / R_L, theta=np.linspace(0, 2 * np.pi, 361)) & inside
    ax_s.contourf(G.real, G.imag, safe, levels=[0.5, 1.5], colors=[COLORS[3]], alpha=0.18, zorder=1)
    gc = g50(R_L * contour(z1))
    gc = gc[np.abs(gc) <= 1.0]
    ax_s.plot(gc.real, gc.imag, ".", ms=1.2, color=COLORS[3], zorder=2)
    g2 = g50(m["Z2i"].to_numpy())
    sc = ax_s.scatter(g2.real, g2.imag, c=100 * m["DE"], cmap="Blues", vmin=40, vmax=70, s=46,
                      edgecolor="white", linewidth=1.2, zorder=4)
    g1 = g50(m["Z1i"].mean())
    ax_s.plot(g1.real, g1.imag, "x", color=COLORS[2], ms=9, mew=2, zorder=5)
    ax_s.annotate(f"Z1i = {m['Z1i'].mean():.1f} ohm", (g1.real, g1.imag), xytext=(8, 6),
                  textcoords="offset points", fontsize=8, color=TEXT)
    cb = fig.colorbar(sc, ax=ax_s, fraction=0.04, pad=0.02)
    cb.set_label("drain efficiency (%)", color=TEXT, fontsize=9)
    cb.outline.set_visible(False)
    ax_s.text(-1.05, -1.2, "shaded: no clipping predicted (margin >= 0)", fontsize=8, color=TEXT)

    # (b) efficiency vs external 2f0 phase, for several drive levels
    pins = [p for p in (28, 30, 32, 34) if p in set(r["Pin"])]
    shades = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"][-len(pins):]
    lo, hi = m.loc[m["margin"] >= 0, "phi2"].agg(["min", "max"])
    ax_de.axvspan(lo, hi, color=COLORS[3], alpha=0.15, lw=0, label="no clipping predicted")
    for p, col in zip(pins, shades):
        d = r[r["Pin"] == p].sort_values("phi2")
        ax_de.plot(d["phi2"], 100 * d["DE"], color=col, lw=2, marker="o", ms=4, label=f"Pin {p:g} dBm")
    ax_de.set_xlabel("2f0 load phase at the drain pin, phi2 (rad; 0 = open, ±pi = short)", color=TEXT)
    ax_de.set_ylabel("drain efficiency (%)", color=TEXT)
    ax_de.set_title("Efficiency vs 2f0 load", fontsize=11)
    ax_de.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")

    # (c) does the waveform really touch zero where the margin crosses zero?
    ax_v.axhline(0, color=GRID, lw=1)
    ax_v.axvline(0, color=GRID, lw=1)
    ax_v.scatter(m["margin"], m["vdsi_min"], s=40, color=COLORS[1], edgecolor="white", linewidth=1.2)
    ax_v.set_xlim(-1.0, 0.2)
    ax_v.set_xlabel("predicted margin, min v(t) (ideal current, Vdc = 1)", color=TEXT)
    ax_v.set_ylabel("simulated min vdsi(t) (V)", color=TEXT)
    ax_v.set_title(f"Predicted vs simulated, Pin = {pin_main:g} dBm", fontsize=11)
    n_off = (m["margin"] < -1.0).sum()
    if n_off:
        ax_v.text(0.02, 0.02, f"{n_off} points with margin < -1 not shown", transform=ax_v.transAxes,
                  fontsize=8, color=TEXT)

    for ax in (ax_de, ax_v):
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=TEXT, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    out = csv.with_suffix(".png")
    fig.savefig(out, dpi=140)
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["plot"]:
        print(plot())
    else:
        main()
        print(plot())
