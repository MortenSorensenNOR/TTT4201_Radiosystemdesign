"""Find the external loads that give a class J load at the intrinsic drain.

For each fundamental f0 the solver adjusts, using the ZLi measurement in the
parasitic_tune testbench,
    RL + j*XL   (load at f0)     until ZLi(f0)     = z1
    X2          (load at 2*f0)   so that ZLi(2f0) is as close as possible to j*x2
    X3          (load at 3*f0)   so that ZLi(3f0) is as close as possible to j*x3 (0 = short)

All frequencies are solved together, one simulation per iteration.
  f0:        fits the bilinear (Moebius) map Zi = (a*ZL + b)/(c*ZL + 1) through
             the last three simulations and inverts it. The first three runs are
             probes around the start load. At small signal the map is exactly
             bilinear, so this lands on the answer right after the probes; at
             large signal it refines. If the inverted load has RL < 0 the target
             is unreachable with a passive load (reported as reachable=False).
  harmonics: the same bilinear fit with the load jX; picks the X whose
             intrinsic Zn is closest to the target point (j*xn; a short for xn = 0).
             An unreachable target gives the nearest reachable point.

    python_utils/adsenv.sh python -m adsutil.classj
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import PACKAGE_DIR
from .parasitic_tune import Testbench

OPEN_X = -1e6   # reactance used for "open"
Z0 = 50.0


@dataclass
class Targets:
    """Loads wanted at the intrinsic drain.

    z1: ZLi(f0).  x2: Im ZLi(2f0).  x3: Im ZLi(3f0).
    x2 or x3 = None leaves that harmonic's load alone (see solve(fixed_X=...)).
    Values may be arrays with one entry per frequency.
    Give them directly (e.g. read off the ADS class J utility), or use class_j().
    """
    z1: complex
    x2: float | None
    x3: float | None = 0.0

    @classmethod
    def class_j(cls, Ropt: float, alpha: float = 1.0, Vdc: float | None = None, Vknee: float = 0.0) -> "Targets":
        """Class J continuum targets, 3rd harmonic shorted.

        Textbook (Cripps, knee ignored): Z1 = Ropt*(1 + j*alpha), X2 = -alpha*(3*pi/8)*Ropt.
        Pass Vdc and Vknee to match the ADS class J utility instead. Its voltage waveform
        is (Vdc + (Vdc - Vknee)*sin t)*(1 + alpha*cos t), so the quadrature swing is
        alpha*Vdc, not alpha*(Vdc - Vknee), and X1 = alpha*Ropt*Vdc/(Vdc - Vknee).
        """
        x1_scale = Vdc / (Vdc - Vknee) if Vdc else 1.0
        return cls(z1=Ropt * (1 + 1j * alpha * x1_scale), x2=-alpha * 3 * np.pi / 8 * Ropt)

    @property
    def harmonics(self) -> dict[int, float]:
        """Harmonic number -> target intrinsic reactance, for the harmonics being solved."""
        return {n: x for n, x in ((2, self.x2), (3, self.x3)) if x is not None}


@dataclass
class Continuum:
    """The class J continuum as the ADS class J utility draws it, from its alpha = 1 point:
    Z1(alpha) = R + j*alpha*X1_max,  X2(alpha) = -alpha*X2_max  (3rd harmonic shorted)."""
    R: float = 27.879       # classj_loadline.dds, alpha = 1 (corrected drain voltage)
    X1_max: float = 32.525
    X2_max: float = 33.129

    def alpha_from_x2(self, x2):
        return -np.asarray(x2) / self.X2_max

    def targets(self, alpha, x3: float | None = 0.0) -> Targets:
        """Targets for a given alpha, with the 2f0 load left alone."""
        return Targets(z1=self.R + 1j * np.asarray(alpha) * self.X1_max, x2=None, x3=x3)


@dataclass
class Settings:
    tol_ohm: float = 0.1          # stop when all errors are below this
    max_iter: int = 20
    probe_ohm: float = 5.0        # size of the two probe steps around the start f0 load
    min_RL: float = 1.0           # keep the fundamental load passive
    input_power_dBm: float | None = None   # None = keep the schematic's value
    bias_line_mm: float | None = None      # TL1 length in drain_bias; None = schematic's value
    drain_line_deg: float | None = None    # series TLIN at the drain (E at 3.8 GHz); None = schematic's
    drain_line_Z: float | None = None
    netlist: str | None = None             # testbench netlist; None = netlists/parasitic_tune.net


# -- harmonic loads: reactance <-> reflection phase ------------------------------

def reactance_to_phase(X):
    """Phase of the reflection coefficient of j*X: 0 = open, pi = short."""
    return np.pi - 2 * np.arctan(np.asarray(X, float) / Z0)


def phase_to_reactance(phase):
    t = np.tan(np.asarray(phase) / 2)
    with np.errstate(divide="ignore"):
        return np.where(np.abs(t) < 1e-9, OPEN_X, Z0 / t)


# -- fundamental: bilinear map --------------------------------------------------

def fit_moebius(zl, zi):
    """Per frequency, a, b, c of Zi = (a*ZL + b)/(c*ZL + 1) from three (ZL, Zi) pairs.

    zl, zi have shape (3, n). Rearranged: a*ZL + b - c*ZL*Zi = Zi, linear in a, b, c.
    """
    A = np.stack([zl, np.ones_like(zl), -zl * zi], axis=-1).transpose(1, 0, 2)   # (n, 3, 3)
    a, b, c = np.linalg.solve(A, zi.T[..., None])[..., 0].T
    return a, b, c


def invert_moebius(a, b, c, z_target):
    """Load ZL that gives Zi = z_target."""
    return (b - z_target) / (c * z_target - a)


# -- one simulation of all frequencies ------------------------------------------

MEASURED = (1, 2, 3)


def simulate(freqs, RL, XL, X: dict[int, np.ndarray], settings: Settings, name: str):
    """Returns ZLi at f0 and a dict harmonic -> ZLi at n*f0 (n = 2, 3)."""
    tb = Testbench.load(settings.netlist) if settings.netlist else Testbench.load()
    if settings.input_power_dBm is not None:
        tb.set_input_power(settings.input_power_dBm)
    if settings.bias_line_mm is not None:
        tb.set_bias_line_length(settings.bias_line_mm)
    tb.set_drain_line(settings.drain_line_deg, settings.drain_line_Z)
    tb.sweep_frequencies(freqs, RL=RL, XL=XL, harmonic_reactances=X)
    zli = tb.simulate(name).intrinsic_load(harmonics=MEASURED)
    return zli["Z1"].to_numpy(), {n: zli[f"Z{n}"].to_numpy() for n in MEASURED[1:]}


# -- solver --------------------------------------------------------------------

def best_on_fitted_curve(a, b, c, target):
    """For a fitted harmonic map Zi = (a*jX + b)/(c*jX + 1): the reactance X whose Zi
    is closest to ``target``, and that Zi. Searches a dense grid of load phases."""
    phases = np.linspace(0, 2 * np.pi, 720, endpoint=False) + np.pi / 720   # avoid an exact open
    jX = 1j * phase_to_reactance(phases)[None, :]
    zi = (a[:, None] * jX + b[:, None]) / (c[:, None] * jX + 1)
    k = np.argmin(np.abs(zi - np.asarray(target)[:, None]), axis=1)
    rows = np.arange(len(k))
    return jX[0, k].imag, zi[rows, k]


def solve(freqs, targets: Targets, settings: Settings = Settings(), start_Z1: complex = 10 + 10j,
          start_X: dict[int, float] | None = None, fixed_X: dict[int, float] | None = None,
          log=print) -> pd.DataFrame:
    """start_X: starting reactance per solved harmonic (default 0 = short).
    fixed_X: reactance for harmonics that are not solved (default: the netlist's value).

    Every quantity uses the same idea: the first three runs probe around the start
    point, then each iteration fits the bilinear map through the last three runs.
      f0:        invert the fit to hit z1 exactly.
      harmonics: pick the reactance whose intrinsic Zn lands closest to j*xn
                 (for xn = 0, a short). If that is not reachable, the result is the
                 nearest reachable point and dist_n says how far off it is.
    A quantity that has settled is frozen, so its fit never sees repeated points.
    """
    freqs = np.asarray(freqs, float)
    nf = len(freqs)
    goals = {n: np.broadcast_to(np.asarray(x, complex) * 1j, (nf,)) for n, x in targets.harmonics.items()}
    z1_target = np.broadcast_to(np.asarray(targets.z1, complex), (nf,))
    start_X = start_X or {}

    probes_Z1 = [0, settings.probe_ohm, 1j * settings.probe_ohm]
    probes_X = [0.0, 50.0, -50.0]
    z1 = np.full(nf, start_Z1, complex)
    X = {n: np.full(nf, start_X.get(n, 0.0)) for n in goals}
    hist = {n: ([], []) for n in [1, *goals]}          # (loads, intrinsic Z) per quantity
    frozen = {n: np.zeros(nf, bool) for n in [1, *goals]}
    predicted = {n: np.full(nf, np.nan + 0j) for n in goals}
    reachable = np.ones(nf, bool)

    for it in range(settings.max_iter + 1):
        if it < 3:
            z1 = np.full(nf, start_Z1 + probes_Z1[it], complex)
            X = {n: np.full(nf, start_X.get(n, 0.0) + probes_X[it]) for n in goals}
        RL, XL = np.maximum(z1.real, settings.min_RL), z1.imag
        loads = {n: np.full(nf, x, float) for n, x in (fixed_X or {}).items()} | X
        z1i, zni = simulate(freqs, RL, XL, loads, settings, name=f"classj_it{it:02d}")

        # settled: f0 on its target, harmonics where the fit predicted them
        settled = {1: np.abs(z1i - z1_target) < settings.tol_ohm}
        settled |= {n: np.abs(zni[n] - predicted[n]) < settings.tol_ohm for n in goals}
        done = np.all(list(settled.values()), axis=0)
        log(f"iter {it:2d}: max |Z1 err| = {np.abs(z1i - z1_target).max():7.3f}, "
            + ", ".join(f"settled@{n}f0 {settled[n].sum()}/{nf}" for n in goals))
        if it >= 2 and (done.all() or it == settings.max_iter):
            break

        hist[1][0].append(RL + 1j * XL); hist[1][1].append(z1i)
        for n in goals:
            hist[n][0].append(1j * X[n]); hist[n][1].append(zni[n])
        if it < 2:
            continue

        for n in [1, *goals]:
            frozen[n] |= settled[n]
            todo = ~frozen[n]
            if not todo.any():
                continue
            zl3 = np.array(hist[n][0][-3:])[:, todo]
            zi3 = np.array(hist[n][1][-3:])[:, todo]
            a, b, c = fit_moebius(zl3, zi3)
            if n == 1:
                z_new = invert_moebius(a, b, c, z1_target[todo])
                reachable[todo] = z_new.real >= settings.min_RL
                z1 = z1.copy()
                z1[todo] = z_new
            else:
                x_new, z_pred = best_on_fitted_curve(a, b, c, goals[n][todo])
                X[n] = X[n].copy()
                X[n][todo] = x_new
                predicted[n][todo] = z_pred

    out = pd.DataFrame({"f0_GHz": freqs / 1e9, "RL": RL, "XL": XL})
    for n in sorted(loads):
        out[f"X{n}"] = loads[n]
    out["ZLi_f0"] = z1i
    for n in MEASURED[1:]:
        out[f"ZLi_{n}f0"] = zni[n]
    out["err_Z1"] = np.abs(z1i - z1_target)
    for n in goals:
        out[f"dist_{n}f0"] = np.abs(zni[n] - goals[n])   # distance from the target point
    out["reachable"] = reachable
    out["converged"] = done
    return out


def solve_continuum(freqs, continuum: Continuum = Continuum(), X2: float = OPEN_X, x3: float | None = 0.0,
                    settings: Settings = Settings(), rounds: int = 3, log=print) -> pd.DataFrame:
    """Continuum mode: keep the 2f0 load fixed (default open), read alpha from the 2f0
    reactance that gives at the intrinsic drain, and solve the fundamental (and 3f0)
    for that alpha. Repeats a few rounds since the loads interact slightly."""
    alpha = np.zeros(len(freqs))
    for r in range(rounds):
        res = solve(freqs, continuum.targets(alpha, x3), settings, fixed_X={2: X2}, log=log)
        new_alpha = continuum.alpha_from_x2(res["ZLi_2f0"].to_numpy().imag)
        change = np.abs(new_alpha - alpha).max()
        log(f"continuum round {r}: max alpha change {change:.4f}")
        alpha = new_alpha
        if change < 0.005:
            break
    res.insert(1, "alpha", alpha)
    res.insert(2, "in_continuum", np.abs(alpha) <= 1)
    return res


if __name__ == "__main__":
    FREQS = np.arange(2.4e9, 3.8e9 + 1, 0.1e9)
    # from the ADS class J utility (classj_loadline.dds), alpha = 1; 3rd harmonic shorted
    TARGETS = Targets(z1=27.879 + 32.525j, x2=-33.129, x3=0.0)

    print(f"targets: ZLi(f0) = {TARGETS.z1:.2f}, Im ZLi(2f0) = {TARGETS.x2:.2f}, Im ZLi(3f0) = {TARGETS.x3}")
    table = solve(FREQS, TARGETS)
    out = PACKAGE_DIR.parent / "results" / "classj_loads.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    print(f"\nwritten to {out}")
