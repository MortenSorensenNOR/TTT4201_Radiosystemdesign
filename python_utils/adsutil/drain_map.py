"""The map from the load at the drain pin to the intrinsic load, per harmonic and frequency.

The parasitics between the package pin and the current source are (nearly) linear, so at
each harmonic frequency the intrinsic load is a bilinear (Moebius) function of the pin load:
    Zi = (a*Z + b) / (c*Z + 1).
Three pin loads per frequency fix a, b, c; a fourth checks the fit. All harmonics are
varied in the same runs, since each harmonic sees its own linear network. Fits are made at
small and large drive to show how much the map moves with drive (a nonlinear Cds would).

The lumped de-embedding network from docs/paracitics.png is compared against the fits.

    python_utils/adsenv.sh python -m adsutil.drain_map
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import PACKAGE_DIR
from .classj import fit_moebius
from .loadbench import Bench, gamma

RESULTS = PACKAGE_DIR.parent / "results"
FREQS = np.round(np.arange(2.4e9, 3.6e9 + 1, 0.1e9))
HARMONICS = (1, 2, 3)
# pin loads: three for the fit, one to check it (fundamental kept low-ohmic for sane drive)
LOADS = {
    1: [8 + 2j, 15 + 10j, 12 - 6j, 20 + 0j],
    2: [3 + 0j, 15 + 30j, 15 - 30j, 50 + 0j],
    3: [3 + 0j, 15 + 30j, 15 - 30j, 50 + 0j],
}
PINS = (20.0, 32.0)


def lumped(Z, f):
    """docs/paracitics.png run forwards: intrinsic -> shunt Cds -> Rd + Ld -> shunt Cbond
    -> 0.01 ohm + 0.56 nH -> 30 mil lead (W 100 mil, H 20 mil, er 9.6). Returns the load
    at the intrinsic plane for a load Z at the pin."""
    w = 2 * np.pi * f
    eeff, z0 = 7.63, 17.9                       # Hammerstad, W/H = 5
    th = w * np.sqrt(eeff) / 3e8 * 0.762e-3
    t = np.tan(th)
    Z = z0 * (Z + 1j * z0 * t) / (z0 + 1j * Z * t)
    Z = Z + 0.01 + 1j * w * 0.56e-9
    Z = 1 / (1 / Z + 1j * w * 0.382833e-12)
    Z = Z + 0.9 + 1j * w * 0.165218e-9
    return 1 / (1 / Z + 1j * w * 1.39778e-12)


def run() -> pd.DataFrame:
    n_l = len(LOADS[1])
    f0 = np.repeat(FREQS, n_l)
    loads = {n: gamma(np.tile(LOADS[n], len(FREQS))) for n in HARMONICS}
    b = Bench()
    b.set_pin_sweep(PINS[0], PINS[1], 4)
    b.set_points(f0, loads)
    r = b.run("drain_map")
    r = r[r["Pin"].isin(PINS)].copy()
    i = r["pt"].round().astype(int).to_numpy() - 1
    r["f0"] = f0[i]
    r["load_i"] = i % n_l
    return r


def fit(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pin, f), g in r.groupby(["Pin", "f0"]):
        g = g.sort_values("load_i")
        for n in HARMONICS:
            Z = np.array(LOADS[n], complex)
            Zi = g[f"Z{n}i"].to_numpy()
            a, b_, c = (x[0] for x in fit_moebius(Z[:3, None], Zi[:3, None]))
            pred = (a * Z[3] + b_) / (c * Z[3] + 1)
            rows.append(dict(Pin=pin, f=n * f, n=n, a=a, b=b_, c=c,
                             check_err=abs(pred - Zi[3]),
                             lumped_err=np.max(np.abs(lumped(Z, n * f) - Zi))))
    return pd.DataFrame(rows)


def run_harmonics(z1_target, pin: float = 32.0) -> pd.DataFrame:
    """Second pass: the fundamental is held at the pin load that gives Z1i = z1_target
    (from the first pass's fit at ``pin``); only the 2f0 and 3f0 loads change. At large
    drive the harmonic maps depend on the fundamental swing, so this is the map at the
    operating point."""
    m1 = pd.read_csv(RESULTS / "drain_map.csv")
    for col in ("a", "b", "c"):
        m1[col] = m1[col].map(complex)
    m1 = m1[(m1["n"] == 1) & (m1["Pin"] == pin)].set_index("f")
    a, b_, c = (m1.loc[FREQS, k].to_numpy() for k in ("a", "b", "c"))
    z1_pin = (b_ - z1_target) / (c * z1_target - a)

    n_l = len(LOADS[2])
    f0 = np.repeat(FREQS, n_l)
    loads = {1: gamma(np.repeat(z1_pin, n_l))}
    loads |= {n: gamma(np.tile(LOADS[n], len(FREQS))) for n in (2, 3)}
    bench = Bench()
    bench.set_pin_sweep(20, pin, 4)
    bench.set_points(f0, loads)
    r = bench.run("drain_map_harm")
    r = r[r["Pin"] == pin].copy()
    i = r["pt"].round().astype(int).to_numpy() - 1
    r["f0"] = f0[i]
    r["load_i"] = i % n_l
    return r, z1_pin


def main_harmonics(z1_target=27.9 * (0.9 + 0.5j), R_L: float = 27.9, name: str = "drain_map_harm"):
    """z1_target: intrinsic fundamental (scalar or one per frequency in FREQS)."""
    r, z1_pin = run_harmonics(np.broadcast_to(np.asarray(z1_target, complex), FREQS.shape))
    gm = lambda z: (z - R_L) / (z + R_L)
    rows = []
    for f, g in r.groupby("f0"):
        g = g.sort_values("load_i")
        row = dict(f0=f, Z1_pin=z1_pin[list(FREQS).index(f)], Z1i=g["Z1i"].mean(),
                   Z1i_spread=np.ptp(np.abs(g["Z1i"])))
        for n in (2, 3):
            Z = np.array(LOADS[n], complex)
            Zi = g[f"Z{n}i"].to_numpy()
            a, b_, c = (x[0] for x in fit_moebius(Z[:3, None], Zi[:3, None]))
            row |= {f"a{n}": a, f"b{n}": b_, f"c{n}": c,
                    f"check{n}_dG": abs(gm((a * Z[3] + b_) / (c * Z[3] + 1)) - gm(Zi[3]))}
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["f0", "Z1_pin", "Z1i", "Z1i_spread", "check2_dG", "check3_dG"]].round(3).to_string(index=False))


def main():
    r = run()
    m = fit(r)
    RESULTS.mkdir(exist_ok=True)
    m.to_csv(RESULTS / "drain_map.csv", index=False)
    pd.set_option("display.width", 200)
    piv = m.pivot_table(index=["n", "Pin"], values=["check_err", "lumped_err"], aggfunc="max")
    print("worst errors over the band (ohm):")
    print(piv.round(2).to_string())
    # how far apart are the small- and large-signal maps? compare on a test grid of pin loads
    test = np.array([2, 5 + 5j, 10 - 10j, 20 + 20j, 1j * 30, -1j * 30, 50], complex)
    lo, hi = (m[m["Pin"] == p].set_index(["n", "f"]) for p in PINS)
    d = []
    for k in lo.index:
        zl = (lo.loc[k, "a"] * test + lo.loc[k, "b"]) / (lo.loc[k, "c"] * test + 1)
        zh = (hi.loc[k, "a"] * test + hi.loc[k, "b"]) / (hi.loc[k, "c"] * test + 1)
        d.append(dict(n=k[0], f=k[1], drive_shift=np.max(np.abs(zh - zl))))
    print("\nlargest change of the map from 20 to 32 dBm drive (ohm, over test loads):")
    print(pd.DataFrame(d).groupby("n")["drive_shift"].max().round(2).to_string())


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["harmonics"]:
        main_harmonics()
    elif sys.argv[1:] == ["harmonics_39"]:
        # the 38.5-39 dBm load line (R_L about 39 ohm), slightly inductive like the ADS networks
        main_harmonics(39 + 5j, R_L=39.0, name="drain_map_harm_39")
    elif sys.argv[1:] == ["harmonics_toy"]:
        # along the fundamental trajectory of the toy design (toy_hb.csv, Pin 32)
        t = pd.read_csv(RESULTS / "toy_hb.csv")
        t = t[t["Pin"] == 32.0].sort_values("f0")
        main_harmonics(t["Z1i"].map(lambda s: complex(s.replace(" ", ""))).to_numpy(), name="drain_map_harm_cap")
    else:
        main()
