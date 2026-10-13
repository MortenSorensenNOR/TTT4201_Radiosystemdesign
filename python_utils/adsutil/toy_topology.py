"""Step 4a: why a line + low-pass filter works (the paper's Fig. 7-9, with our transistor).

The output network is idealized as
    drain pin -- series line (Zc, theta at 3 GHz) -- ideal low-pass box
where the box presents a real R_T across the fundamental band and a pure reactance at
the harmonics: a short (low-pass starting with a shunt C) or an open (starting with a
series L). The line rotates both, clockwise, by different amounts at f0 and 2f0.

Each design is passed through the fitted pin -> intrinsic maps (drain_map.py) and scored:
    worst Re(Z1i)/R_L over 2.4-3.6 GHz, among designs with clipping margin >= 0 everywhere.

    python_utils/adsenv.sh python -m adsutil.toy_topology
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import PACKAGE_DIR
from .clipping import margin

RESULTS = PACKAGE_DIR.parent / "results"
R_L = 27.9
F_REF = 3.0e9
TH = np.linspace(0, 2 * np.pi, 181)    # time samples for the margin; enough for a scan


def load_maps():
    """Per fundamental frequency: (a, b, c) for f0 (any drive, 32 dBm fit), 2f0 and 3f0
    (operating-point fits). Returns freqs and a dict n -> (a, b, c) arrays."""
    m1 = pd.read_csv(RESULTS / "drain_map.csv")
    mh = pd.read_csv(RESULTS / "drain_map_harm.csv")
    for df in (m1, mh):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(complex)
    m1 = m1[(m1["n"] == 1) & (m1["Pin"] == 32.0)].sort_values("f")
    mh = mh.sort_values("f0")
    f = mh["f0"].to_numpy().real
    assert np.allclose(m1["f"].to_numpy().real, f)
    maps = {1: tuple(m1[k].to_numpy() for k in "abc")}
    maps |= {n: tuple(mh[f"{k}{n}"].to_numpy() for k in "abc") for n in (2, 3)}
    return f, maps


def apply(m, Z):
    a, b, c = m
    return (a * Z + b) / (c * Z + 1)


def line(Z, Zc, theta_ref, f):
    t = np.tan(theta_ref * f / F_REF)
    return Zc * (Z + 1j * Zc * t) / (Zc + 1j * Z * t)


def scan(term: str, Zc_grid, th_grid_deg, RT_grid, m_min: float = 0.0):
    f, maps = load_maps()
    Zh = 1e-3 if term == "short" else 1e6                # box at the harmonics
    Zc, th, RT = np.meshgrid(Zc_grid, np.deg2rad(th_grid_deg), RT_grid, indexing="ij")
    Zc, th, RT = (x[..., None] for x in (Zc, th, RT))     # last axis: frequency
    z1i = apply(maps[1], line(RT + 0j, Zc, th, f))
    z2i = apply(maps[2], line(Zh + 0j, Zc, th, 2 * f))
    mg = margin(z1i / R_L, z2i / R_L, TH)
    re1 = z1i.real / R_L
    ok = (mg >= m_min).all(axis=-1)
    worst = np.where(ok, re1.min(axis=-1), np.nan)
    # best R_T for each (Zc, theta)
    best_k = np.nanargmax(np.where(np.isnan(worst), -np.inf, worst), axis=-1)
    i, j = np.indices(best_k.shape)
    return dict(f=f, worst=worst[i, j, best_k], RT=np.asarray(RT_grid)[best_k],
                z1i=z1i[i, j, best_k], z2i=z2i[i, j, best_k], margin=mg[i, j, best_k],
                Zc=np.asarray(Zc_grid), th=np.asarray(th_grid_deg))


def main():
    Zc_grid = np.arange(10, 62.5, 2.5)
    th_grid = np.arange(0, 181, 3)
    RT_grid = np.arange(4, 30.5, 0.5)
    out = {}
    for term in ("short", "open"):
        s = scan(term, Zc_grid, th_grid, RT_grid)
        out[term] = s
        w = s["worst"]
        if np.all(np.isnan(w)):
            print(f"{term}: no design without clipping")
            continue
        i, j = np.unravel_index(np.nanargmax(w), w.shape)
        print(f"harmonics {term:5}: best worst-case Re(Z1i)/R_L = {w[i, j]:.3f} "
              f"(Re Z1i >= {w[i, j] * R_L:.1f} ohm) with Zc = {s['Zc'][i]:.1f} ohm, "
              f"theta = {s['th'][j]:.0f} deg at 3 GHz, R_T = {s['RT'][i, j]:.1f} ohm; "
              f"{np.isfinite(w).mean() * 100:.0f} % of (Zc, theta) have a clean design")
    np.savez(RESULTS / "toy_topology.npz", **{f"{t}_{k}": v for t, s in out.items() for k, v in s.items()})
    return out


def plot(out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .clipping import contour, safe_region
    from .plot_loads import COLORS, GRID, TEXT, gamma as g50, smith_axes

    if out is None:
        d = np.load(RESULTS / "toy_topology.npz")
        out = {t: {k.split("_", 1)[1]: d[k] for k in d.files if k.startswith(t + "_")} for t in ("short", "open")}

    fig = plt.figure(figsize=(16, 5.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    for col, term in enumerate(("short", "open")):
        s = out[term]
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(s["th"], s["Zc"], 100 * s["worst"], cmap="Blues", vmin=60, vmax=95, shading="nearest")
        if np.isfinite(s["worst"]).any():
            i, j = np.unravel_index(np.nanargmax(s["worst"]), s["worst"].shape)
            ax.plot(s["th"][j], s["Zc"][i], "x", color=COLORS[2], ms=10, mew=2)
        ax.set_title(f"low-pass box looks like {'a short' if term == 'short' else 'an open'} at 2f0",
                     fontsize=11)
        ax.set_xlabel("line length at 3 GHz (deg)", color=TEXT)
        ax.set_ylabel("line impedance Zc (ohm)", color=TEXT)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.02, 0.97, "white: clips somewhere in band", transform=ax.transAxes, va="top",
                fontsize=8, color=TEXT)
    cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.03, pad=0.02)
    cb.set_label("worst-case Re(Z1i) / R_L over 2.4-3.6 GHz (%)", color=TEXT, fontsize=9)
    cb.outline.set_visible(False)

    # Smith chart of the best open-terminated design
    s = out["open"]
    i, j = np.unravel_index(np.nanargmax(s["worst"]), s["worst"].shape)
    z1, z2, f = s["z1i"][i, j], s["z2i"][i, j], s["f"]
    ax = fig.add_subplot(gs[0, 2])
    smith_axes(ax, f"best open design: Zc = {s['Zc'][i]:.1f} ohm, {s['th'][j]:.0f} deg, R_T = {s['RT'][i, j]:.0f} ohm")
    gx = np.linspace(-1.05, 1.05, 211)
    G = gx[None, :] + 1j * gx[::-1, None]
    Zg = 50 * (1 + G) / (1 - G)
    for k, alpha in ((0, 0.25), (len(f) // 2, 0.25), (len(f) - 1, 0.25)):
        safe = safe_region(z1[k] / R_L, Zg / R_L, TH) & (np.abs(G) <= 1)
        ax.contour(G.real, G.imag, safe.astype(float), levels=[0.5], colors=[COLORS[3]], linewidths=1)
    g1, g2 = g50(z1), g50(z2)
    ax.plot(g1.real, g1.imag, "-o", color=COLORS[1], ms=4, lw=2, label="Z1i (f0)")
    ax.plot(g2.real, g2.imag, "-o", color=COLORS[2], ms=4, lw=2, label="Z2i (2f0)")
    for g, lab, off in ((g1, "f0", (6, 4)), (g2, "2f0", (6, -10))):
        ax.annotate(f"{lab} {f[0] / 1e9:.1f}", (g[0].real, g[0].imag), xytext=off,
                    textcoords="offset points", fontsize=8, color=TEXT)
        ax.annotate(f"{lab} {f[-1] / 1e9:.1f}", (g[-1].real, g[-1].imag), xytext=off,
                    textcoords="offset points", fontsize=8, color=TEXT)
    ax.plot([], [], color=COLORS[3], lw=1, label="clipping contours (2.4 / 3.0 / 3.6 GHz)")
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    p = RESULTS / "toy_topology.png"
    fig.savefig(p, dpi=140)
    return p


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["plot"]:
        main()
    print(plot())
