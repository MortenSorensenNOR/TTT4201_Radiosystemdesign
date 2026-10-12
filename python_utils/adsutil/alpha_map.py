"""Map alpha against frequency and the 2f0 load phase at the device drain pin.

Bench: s30_fund_targets (ideal DC feed, load box right at the drain pin), with
the fundamental load from its fitted polynomials (a0..b2) and the optimizer
removed. Harmonic loads are Gamma = exp(j*phi): phi = 0 open, +-pi short.

For each f0 and phi2 the HB run gives Z2i = ZLi(2f0) at the intrinsic drain, and
    alpha = -Im(Z2i) / X2max      (X2max = 33.13 ohm, ADS class J utility)
The same is done for 3f0 against phi3 (with phi2 = 0). Everything between the
drain pin and the 50 ohm port (tee, bias line, stepped lines) is "the output
network": its reflection S11(f) at the pin is a curve on these maps.

Pin -> intrinsic is bilinear in the load (to ~0.02 in alpha away from the
package resonance), so each frequency's map is also stored as a Moebius fit,
which takes any load, including |Gamma| < 1.

    python_utils/adsenv.sh python -m adsutil.alpha_map            # run the grids (~15 s)
    python_utils/adsenv.sh python -m adsutil.alpha_map plot       # plot, with the s40 network on top
"""

from __future__ import annotations

import sys
from typing import Sequence

import numpy as np
import pandas as pd

from . import NETLIST_DIR, PACKAGE_DIR, results, sim
from .classj import Continuum, fit_moebius
from .netlist import Netlist

CELL = "s30_fund_targets"
ANALYSIS = "HB1"
RESULTS = PACKAGE_DIR.parent / "results"
Z0 = 50.0
BAND = (2.4e9, 3.4e9)
# s30 has no input match (50 ohm port straight onto the gate), so its own Pin = 26 dBm
# underdrives the device (36-38.5 dBm out). 32 dBm available gives 41-42 dBm and
# 64-76 % DE across the band, i.e. the drain is saturated as in the real amplifier.
PIN = 32.0
CONT = Continuum()


def bench(Pin: float | None = None) -> Netlist:
    nl = Netlist.from_file(NETLIST_DIR / f"{CELL}.net")
    nl.remove_optim()
    nl.remove_sweeps()
    nl.remove_analysis_sweep(ANALYSIS)
    if Pin is not None:
        nl.set("Pin", Pin)
    return nl


def run_grid(f0s, phi2s, phi3s, Pin: float | None = None, name: str = "alpha_map",
             extra: dict[str, Sequence] | None = None) -> pd.DataFrame:
    """One HB point per (f0, phi2, phi3) triple (the three sequences have equal length).
    ``extra`` sweeps more netlist variables along with them, e.g. {"RL": ..., "XL": ...}
    for the fundamental load at the pin; their values become columns too.

    Returns f0, phi2, phi3, Z1i, Z2i, Z3i, Pout (W, at f0 into the load box), DE
    (Pout / 28 V supply power) and I3_I1 (|Idi| at 3f0 over f0); a point that did
    not converge is missing.
    """
    nl = bench(Pin)
    extra = {k: np.asarray(v) for k, v in (extra or {}).items()}
    n = nl.sweep_lists(ANALYSIS, {"rf_freq": f0s, "phi2": phi2s, "phi3": phi3s, **extra}, index="py_i")
    blocks = results.load(sim.run(nl, name=name))
    df = results.find(blocks, "ZLi")
    idi = results.find(blocks, "Idi")
    hb = results.find(blocks, "Vload")
    f0s, phi2s, phi3s = map(np.asarray, (f0s, phi2s, phi3s))
    rows = []
    for i, g in df.groupby(df["py_i"].round().astype(int)):
        f0 = f0s[i - 1]
        z = {k: results.at_harmonic(g, "ZLi", f0, k) for k in (1, 2, 3)}
        if any(v.empty for v in z.values()):
            continue
        h = hb[hb["py_i"].round() == i]
        v1, i1 = (results.at_harmonic(h, c, f0, 1).iloc[0] for c in ("Vload", "Iload.i"))
        pout = 0.5 * np.real(v1 * np.conj(i1))
        pdc = abs(28.0 * np.real(results.at_harmonic(h, "SRC3.i", f0, 0).iloc[0]))
        d = idi[idi["py_i"].round() == i]
        i_h = {k: abs(results.at_harmonic(d, "Idi", f0, k).iloc[0]) for k in (1, 3)}
        rows.append({"f0": f0, "phi2": phi2s[i - 1], "phi3": phi3s[i - 1],
                     **{k: v[i - 1] for k, v in extra.items()},
                     **{f"Z{k}i": v.iloc[0] for k, v in z.items()},
                     "Pout": pout, "DE": pout / pdc, "I3_I1": i_h[3] / i_h[1]})
    out = pd.DataFrame(rows)
    out = out[np.isfinite(out[["Pout", "DE"]]).all(axis=1) & out["Z2i"].map(np.isfinite)]
    out.attrs["requested"] = n
    return out


# -- csv with complex columns ----------------------------------------------------

def save(df: pd.DataFrame, name: str) -> None:
    d = df.copy()
    for c in [c for c in d.columns if d[c].dtype == complex]:
        d[c + "_re"], d[c + "_im"] = d[c].to_numpy().real, d[c].to_numpy().imag
        d = d.drop(columns=c)
    d.to_csv(RESULTS / f"{name}.csv", index=False)


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / f"{name}.csv")
    for c in [c[:-3] for c in d.columns if c.endswith("_re")]:
        d[c] = d.pop(c + "_re") + 1j * d.pop(c + "_im")
    return d


# -- bilinear map pin -> intrinsic ---------------------------------------------------

def gamma_to_z(g):
    g = np.asarray(g, complex)
    return Z0 * (1 + g) / (1 - g + 1e-15)


class HarmonicMap:
    """Per f0: Zni = (a*ZL + b)/(c*ZL + 1) for harmonic n, ZL the load at the drain pin.
    Fitted through three reactive probes (phi = -90, 0, +90 deg) of a grid."""

    def __init__(self, grid: pd.DataFrame, n: int):
        self.n = n
        phi, z = f"phi{n}", f"Z{n}i"
        f0s, coefs = [], []
        for f0, g in grid.groupby("f0"):
            idx = [int(np.argmin(np.abs(g[phi].to_numpy() - p))) for p in (-np.pi / 2, 0, np.pi / 2)]
            zl = gamma_to_z(np.exp(1j * g[phi].to_numpy()[idx]))
            a, b, c = fit_moebius(zl[:, None], g[z].to_numpy()[idx][:, None])
            f0s.append(f0)
            coefs.append((a[0], b[0], c[0]))
        self.f0 = np.array(f0s)
        self.coefs = np.array(coefs)

    def __call__(self, f0, gamma_load):
        """Intrinsic Zn for loads gamma_load (re 50 ohm, at n*f0) at fundamentals f0."""
        f0 = np.atleast_1d(f0)
        # coefficients vary smoothly with f0: interpolate re/im separately
        a, b, c = (np.interp(f0, self.f0, self.coefs[:, k].real) + 1j * np.interp(f0, self.f0, self.coefs[:, k].imag)
                   for k in range(3))
        zl = gamma_to_z(gamma_load)
        return (a * zl + b) / (c * zl + 1)


def network_s11(netlist: str = "s40_output_match", name: str = "s40_now") -> pd.DataFrame:
    """S11 at the drain pin of a linear output-network bench, optimizer removed."""
    nl = Netlist.from_file(NETLIST_DIR / f"{netlist}.net")
    nl.remove_optim()
    df = results.find(results.load(sim.run(nl, name=name)), "S[1,1]")
    return df[["freq", "S[1,1]"]].rename(columns={"S[1,1]": "S11"})


def at(s11: pd.DataFrame, f):
    s = s11["S11"].to_numpy()
    fr = s11["freq"].to_numpy()
    return np.interp(f, fr, s.real) + 1j * np.interp(f, fr, s.imag)


# -- plot ----------------------------------------------------------------------------

def plot(out_png=RESULTS / "alpha_map.png", s11: pd.DataFrame | None = None):
    """alpha vs phi2, drain efficiency vs phi2, drain efficiency vs phi3; optionally with a
    network's S11 at 2f0 / 3f0 drawn on top. Each panel holds the other harmonic fixed
    (phi3 = short, resp. phi2 = open) and the fundamental on the s30 polynomial."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    TEXT, MUTED, LINE = "#0b0b0b", "#52514e", "#0b0b0b"
    # diverging blue <-> red with a gray midpoint (alpha < 0 blue, alpha > 0 red)
    div = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a82b2b"])
    # sequential blue, light = low efficiency, dark = high
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])

    d2, d3 = load("alpha_map_phi2"), load("alpha_map_phi3")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)

    def grid_of(d, phi, val):
        p = d.pivot_table(index=phi, columns="f0", values=val)
        return p.columns.to_numpy() / 1e9, np.degrees(p.index.to_numpy()), p.to_numpy()

    def hatch_bad_alpha(ax, F, P, A):
        ax.pcolor(F, P, np.ma.masked_invalid(np.where(np.abs(A) > 1, 1.0, np.nan)),
                  hatch="///", alpha=0, shading="nearest")

    # 2f0: alpha
    d2 = d2.assign(alpha=CONT.alpha_from_x2(d2["Z2i"].to_numpy().imag))
    F, P, A = grid_of(d2, "phi2", "alpha")
    ax = axes[0]
    levels = [-1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5]
    cf = ax.pcolormesh(F, P, np.clip(A, -1.5, 1.5), cmap=div, norm=BoundaryNorm(levels, div.N), shading="nearest")
    hatch_bad_alpha(ax, F, P, A)
    cs = ax.contour(F, P, np.where(np.abs(A) < 1.2, A, np.nan), levels=[-0.5, 0, 0.25, 0.5], colors=MUTED, linewidths=0.8)
    ax.clabel(cs, fmt="α=%.2g", fontsize=8)
    fig.colorbar(cf, ax=ax, label="α = −Im(Z2i)/33.13", ticks=levels[1:-1])
    ax.set(ylabel="φ2 at drain pin (deg)   0 = open, ±180 = short",
           title="α from the 2f0 load  (hatched: |α| > 1)")

    # 2f0: efficiency
    _, _, DE2 = grid_of(d2, "phi2", "DE")
    ax = axes[1]
    cf = ax.pcolormesh(F, P, 100 * DE2, cmap=seq, shading="nearest", vmin=45, vmax=80)
    cs = ax.contour(F, P, 100 * DE2, levels=[60, 65, 70, 75], colors=MUTED, linewidths=0.8)
    ax.clabel(cs, fmt="%g %%", fontsize=8)
    hatch_bad_alpha(ax, F, P, A)
    fig.colorbar(cf, ax=ax, label="drain efficiency (%)")
    ax.set(ylabel="φ2 at drain pin (deg)", title="Drain efficiency vs 2f0 load  (3f0 shorted at pin)")

    # 3f0: efficiency
    F3, P3, DE3 = grid_of(d3, "phi3", "DE")
    ax = axes[2]
    cf = ax.pcolormesh(F3, P3, 100 * DE3, cmap=seq, shading="nearest", vmin=45, vmax=80)
    cs = ax.contour(F3, P3, 100 * DE3, levels=[65, 70, 75], colors=MUTED, linewidths=0.8)
    ax.clabel(cs, fmt="%g %%", fontsize=8)
    _, _, X3 = grid_of(d3.assign(X3=d3["Z3i"].to_numpy().imag, absZ3=np.abs(d3["Z3i"].to_numpy())), "phi3", "X3")
    _, _, Z3 = grid_of(d3.assign(absZ3=np.abs(d3["Z3i"].to_numpy())), "phi3", "absZ3")
    ax.contour(F3, P3, np.where(Z3 < 60, X3, np.nan), levels=[0], colors=LINE, linewidths=1.2, linestyles="dotted")
    fig.colorbar(cf, ax=ax, label="drain efficiency (%)")
    ax.set(ylabel="φ3 at drain pin (deg)",
           title="Drain efficiency vs 3f0 load  (2f0 open at pin; dotted: intrinsic short)")

    if s11 is not None:
        f = np.linspace(*BAND, 101)
        for ax, n in zip(axes, (2, 2, 3)):
            ph = np.degrees(np.angle(at(s11, n * f)))
            ph[np.r_[False, np.abs(np.diff(ph)) > 180]] = np.nan   # break the line at the wrap
            ax.plot(f / 1e9, ph, color="#eda100", lw=3.2, solid_capstyle="round")
            ax.plot(f / 1e9, ph, color=LINE, lw=1.4, label="current output_match (s40)")
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, frameon=False)
    for ax, n in zip(axes, (2, 2, 3)):
        ax.set_xlabel(f"f0 (GHz)   [load at {n}·f0]")
        ax.set_yticks(range(-180, 181, 45))
        ax.set_xlim(BAND[0] / 1e9 - 0.025, BAND[1] / 1e9 + 0.025)
        ax.set_ylim(-182.5, 182.5)
    fig.suptitle(f"CG2H40010F at the drain pin, Pin = {PIN:g} dBm (saturated), fundamental on the s30 polynomial",
                 color=TEXT)
    fig.savefig(out_png, dpi=120)
    return out_png


def main_grids(Pin: float = PIN):
    f0 = np.round(np.arange(BAND[0], BAND[1] + 1, 0.05e9))
    phis = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    F, P = (a.ravel() for a in np.meshgrid(f0, phis, indexing="ij"))
    for name, args in (("alpha_map_phi2", (F, P, np.full_like(F, np.pi))),
                       ("alpha_map_phi3", (F, np.zeros_like(F), P))):
        df = run_grid(*args, Pin=Pin, name=name)
        save(df, name)
        print(f"{name}: {len(df)}/{df.attrs['requested']} points converged")


if __name__ == "__main__":
    if sys.argv[1:2] == ["plot"]:
        s11 = network_s11()
        print(plot(s11=s11))
    else:
        main_grids()
