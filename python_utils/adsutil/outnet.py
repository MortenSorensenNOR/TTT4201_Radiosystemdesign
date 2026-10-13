"""Step 4b: microstrip output network on the project FR-4, optimized against the clipping margin.

    drain pin -- TLa --T1-- TLb --T2-- TLc --T3-- TLout -- 50 ohm
                       |          |          |
                    stub S1    stub S2    bias: 0.2 mm line -> decoupling C -> Vdd
                    (open)     (open)

ADS computes the passive network's S-parameters (many candidates per run, one per sweep
point). Python turns the impedance at the drain pin at f0, 2f0 into the intrinsic loads
with the fitted maps (drain_map.py), and scores each candidate over 2.4-3.6 GHz:

    maximize   min over f of  Re(Z1i)/R_L * eta_net        (eta_net = |S21|^2 / (1 - |S11|^2))
    subject to margin(Z1i, Z2i) >= M_MIN and Re(Z1i) >= RE_MIN at every f (as penalties)

    python_utils/adsenv.sh python -m adsutil.outnet            # optimize
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import PACKAGE_DIR, results, sim
from .clipping import margin
from .netlist import Netlist

RESULTS = PACKAGE_DIR.parent / "results"
R_L = 27.9
M_MIN = 0.05
RE_MIN = 21.0
FREQS = np.round(np.arange(2.4e9, 3.6e9 + 1, 0.1e9))
TH = np.linspace(0, 2 * np.pi, 181)

G2_MIN = 0.85      # the low-pass must reflect 2f0 at the pin
N_STUBS = 3
W50 = 2.8          # about 50 ohm on 1.52 mm FR-4
WALL = "Wall1=1.0E+30 mm Wall2=1.0E+30 mm Mod=1"

# name: (low, high) in mm. Section k: series line (W{k}, L{k}) then an open stub (Ws{k}, Ls{k}).
PARAMS = {}
for k in range(1, N_STUBS + 1):
    PARAMS |= {f"W{k}": (0.3, 12), f"L{k}": (0.5, 25), f"Ws{k}": (0.5, 6), f"Ls{k}": (0.5, 20)}
PARAMS |= {f"W{N_STUBS + 1}": (0.3, 12), f"L{N_STUBS + 1}": (0.5, 25)}

# Drain bias. "drain_L": chip inductor on a short 0.5 mm stub at a tee right at the drain, as in
# the paper's Fig. 10: Coilcraft 0603DC-15N from its de-embedded .s2p (valid 1 MHz-8 GHz, held
# constant above; the bare part resonates near 6 GHz, board pads will pull that down somewhat).
# "line_out" (v1): 0.2 mm line of length Lbias at a tee at the 50 ohm end.
BIAS = "drain_L"
INDUCTOR_S2P = PACKAGE_DIR.parent.parent / "data" / "coilcraft_0603dc" / "0603DC-15N.s2p"
if BIAS == "line_out":
    PARAMS["Lbias"] = (5, 20)


def _template() -> str:
    n = N_STUBS
    rows = ["Port:P1  n0 0 Num=1 Z=50 Ohm Noise=no"]
    first = "n0"
    if BIAS == "drain_L":
        first = "n0x"
        rows += [f"MTEE_ADS:Tdc  n0 n0x nb0 Subst=\"MSub1\" W1=W1 mm W2=W1 mm W3=0.5 mm",
                 f"MLIN2:TLdc  nb0 nb1 Subst=\"MSub1\" W=0.5 mm L=1 mm {WALL}",
                 "#uselib \"ckt\" , \"S2P\"",
                 f'S2P:Lind  nb1 nbc 0 File="{INDUCTOR_S2P}" Type="touchstone" InterpMode="linear" '
                 'InterpDom="ri" ExtrapMode="constant" Temp=27.0 CheckPassivity=0',
                 "C:Cdec  nbc 0 C=10 nF"]
    for k in range(1, n + 1):
        rows += [f"MLIN2:TL{k}  {first if k == 1 else f'n{k - 1}'} a{k} Subst=\"MSub1\" W=W{k} mm L=L{k} mm {WALL}",
                 f"MTEE_ADS:T{k}  a{k} n{k} s{k} Subst=\"MSub1\" W1=W{k} mm W2=W{k + 1} mm W3=Ws{k} mm",
                 f"MLOC2:S{k}  s{k} Subst=\"MSub1\" W=Ws{k} mm L=Ls{k} mm {WALL}"]
    if BIAS == "line_out":
        rows += [f"MLIN2:TL{n + 1}  n{n} a{n + 1} Subst=\"MSub1\" W=W{n + 1} mm L=L{n + 1} mm {WALL}",
                 f"MTEE_ADS:Tbias  a{n + 1} nf nbias Subst=\"MSub1\" W1=W{n + 1} mm W2={W50} mm W3=0.2 mm",
                 f"MLIN2:TLbias  nbias nbc Subst=\"MSub1\" W=0.2 mm L=Lbias mm {WALL}",
                 "C:Cdec  nbc 0 C=10 nF"]
    else:
        rows += [f"MLIN2:TL{n + 1}  n{n} nf Subst=\"MSub1\" W=W{n + 1} mm L=L{n + 1} mm {WALL}"]
    rows += [f"MLIN2:TLout  nf nout Subst=\"MSub1\" W={W50} mm L=5 mm {WALL}",
             "Port:P2  nout 0 Num=2 Z=50 Ohm Noise=no"]
    return "\n".join([
        'Options ResourceUsage=yes UseNutmegFormat=no EnableOptim=no TopDesignName="py:outnet"',
        "model MSub1 MSUB H=1.52 mm Er=4.6 Mur=1 Cond=5.96e7 Hu=1e+33 mm T=35 um TanD=0.02 Rough=0 mm "
        "DielectricLossModel=1 FreqForEpsrTanD=1.0 GHz LowFreqForTanD=1.0 kHz HighFreqForTanD=1.0 THz RoughnessModel=2",
        *rows,
        *(f"{k}={(lo + hi) / 2:g}" for k, (lo, hi) in PARAMS.items()),
        'S_Param:SP1 CalcS=yes CalcY=no CalcZ=no SweepVar="freq" SweepPlan="SP1_stim"',
        "SweepPlan: SP1_stim Start=2.4 GHz Stop=7.2 GHz Step=100 MHz",
    ]) + "\n"


TEMPLATE = _template()


def load_maps(harm_csv: str = "drain_map_harm_cap.csv"):
    m1 = pd.read_csv(RESULTS / "drain_map.csv")
    mh = pd.read_csv(RESULTS / harm_csv)
    for df in (m1, mh):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(complex)
    m1 = m1[(m1["n"] == 1) & (m1["Pin"] == 32.0)].sort_values("f")
    mh = mh.sort_values("f0")
    assert np.allclose(m1["f"].to_numpy().real, FREQS) and np.allclose(mh["f0"].to_numpy().real, FREQS)
    maps = {1: tuple(m1[k].to_numpy() for k in "abc")}
    maps[2] = tuple(mh[f"{k}2"].to_numpy() for k in "abc")
    return maps


MAPS = None


def simulate(X: np.ndarray, name: str = "outnet") -> dict:
    """X: (n_candidates, n_params). Returns S11, S21 arrays of shape (n, n_freq) and freqs."""
    nl = Netlist(TEMPLATE)
    lists = {k: X[:, i] for i, k in enumerate(PARAMS)}
    nl.sweep_lists("SP1", lists, index="pt")
    blocks = results.load(sim.run(nl, name=name))
    df = results.find(blocks, "S[1,1]")
    df = df.sort_values(["pt", "freq"])
    f = np.sort(df["freq"].unique())
    n = len(X)
    s11 = df["S[1,1]"].to_numpy().reshape(n, len(f))
    s21 = df["S[2,1]"].to_numpy().reshape(n, len(f))
    return dict(f=f, s11=s11, s21=s21)


def evaluate(sp: dict) -> dict:
    global MAPS
    MAPS = MAPS or load_maps()
    f = sp["f"]
    idx = lambda ff: np.array([np.argmin(np.abs(f - x)) for x in ff])
    i1, i2 = idx(FREQS), idx(2 * FREQS)
    zpin = 50 * (1 + sp["s11"]) / (1 - sp["s11"])
    ap = lambda m, Z: (m[0] * Z + m[1]) / (m[2] * Z + 1)
    z1i = ap(MAPS[1], zpin[:, i1])
    z2i = ap(MAPS[2], zpin[:, i2])
    s11, s21 = sp["s11"][:, i1], sp["s21"][:, i1]
    g2 = np.abs(sp["s11"][:, i2])                   # harmonic reflection at the pin
    eta = np.abs(s21) ** 2 / np.maximum(1 - np.abs(s11) ** 2, 1e-9)
    # the 2f0 maps were fitted with near-lossless loads; far from that they can return a
    # negative Re(Z2i), which would look like free margin. Never credit it.
    z2_safe = np.maximum(z2i.real, 0) + 1j * z2i.imag
    mg = margin(z1i / R_L, z2_safe / R_L, TH)
    score_f = z1i.real / R_L * eta
    pen = (10 * np.clip(M_MIN - mg, 0, None).sum(-1)
           + 20 * np.clip(RE_MIN - z1i.real, 0, None).sum(-1) / R_L
           + 10 * np.clip(G2_MIN - g2, 0, None).sum(-1))
    cost = -score_f.min(-1) + pen
    return dict(cost=cost, z1i=z1i, z2i=z2i, zpin1=zpin[:, i1], zpin2=zpin[:, i2], eta=eta, margin=mg,
                score=score_f.min(-1), g2=g2)


def optimize(maxiter: int = 200, popsize: int = 12, seed: int = 1):
    from scipy.optimize import differential_evolution

    bounds = list(PARAMS.values())
    log = RESULTS / "outnet_log.txt"
    log.write_text("")
    gen = [0]
    t0 = time.time()

    def cost_fn(xT):                    # vectorized: xT has shape (n_params, S)
        ev = evaluate(simulate(xT.T, name=f"outnet_g{gen[0]:03d}"))
        gen[0] += 1
        k = np.argmin(ev["cost"])
        with log.open("a") as fh:
            fh.write(f"gen {gen[0]:3d}  {time.time() - t0:6.0f}s  best cost {ev['cost'][k]:+.4f}  "
                     f"score {ev['score'][k]:.3f}  min margin {ev['margin'][k].min():+.3f}\n")
        return ev["cost"]

    res = differential_evolution(cost_fn, bounds, vectorized=True, updating="deferred", maxiter=maxiter,
                                 popsize=popsize, seed=seed, tol=1e-4, polish=False)
    best = dict(zip(PARAMS, map(float, res.x)))
    (RESULTS / "outnet_best.json").write_text(json.dumps(best, indent=1))
    return best


def report(best: dict):
    X = np.array([[best[k] for k in PARAMS]])
    ev = evaluate(simulate(X, name="outnet_best"))
    rows = pd.DataFrame({"f0_GHz": FREQS / 1e9, "Zpin_f0": ev["zpin1"][0], "Zpin_2f0": ev["zpin2"][0],
                         "G2pin": ev["g2"][0], "Z1i": ev["z1i"][0], "Z2i": ev["z2i"][0], "margin": ev["margin"][0],
                         "eta_net_dB": 10 * np.log10(ev["eta"][0])})
    pd.set_option("display.width", 220)
    print(json.dumps({k: round(v, 2) for k, v in best.items()}))
    print(rows.round(3).to_string(index=False))
    rows.to_csv(RESULTS / "outnet_best.csv", index=False)
    return rows



def hb_netlist(best: dict) -> Netlist:
    """The transistor bench of loadbench.py with this microstrip network as the drain load.
    Vdd comes in through the bias network (see BIAS); the output is DC-blocked into a 50 ohm port."""
    from .loadbench import TEMPLATE as BENCH
    drop = ("Short:DC_Feed_D", "Short:Ids", "Short:DC_Block2", "Short:Iload", "S_Port:LoadBox")
    bench = [l for l in BENCH.splitlines() if not l.startswith(drop)]
    skip = ("Options", "Port:", "S_Param", "SweepPlan")
    net = [l for l in _template().splitlines()
           if not l.startswith(skip) and not any(l.startswith(f"{k}=") for k in PARAMS)]
    net = [l.replace(" n0 ", " ndrain ").replace(" nout ", " nload0 ") for l in net]
    net += ["Short:Ids  nvd nbc Mode=0 SaveCurrent=yes",            # DC supply current
            "Short:DC_Block2  nload0 nload Mode=1",
            "Short:Iload  nload nbox Mode=0 SaveCurrent=yes",
            "Port:Term1  nbox 0 Num=2 Z=50 Ohm Noise=no"]
    net += [f"{k}={v:.6g}" for k, v in best.items()]
    return Netlist("\n".join(bench + net) + "\n")


def hb_verify(best: dict, name: str = "outnet_hb"):
    from .loadbench import Bench, summarize
    b = Bench(hb_netlist(best))
    b.set_pin_sweep(20, 34, 2)
    b.netlist.sweep_lists("HB1", {"rf_freq": list(FREQS)}, index="pt")
    r = summarize(results.load(sim.run(b.netlist, name=name)))
    r["f0_GHz"] = FREQS[r["pt"].round().astype(int).to_numpy() - 1] / 1e9
    r["margin"] = margin(r["Z1i"].to_numpy() / R_L, r["Z2i"].to_numpy() / R_L)
    r.to_csv(RESULTS / f"{name}.csv", index=False)
    pd.set_option("display.width", 220)
    for p in (32.0, 34.0):
        print(f"--- Pin = {p:g} dBm (available from 50 ohm, no input match)")
        print(r[r["Pin"] == p][["f0_GHz", "Z1i", "Z2i", "Z3i", "margin", "Pout_dBm", "DE", "vdsi_min"]]
              .round(3).to_string(index=False))
    return r


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["speedtest"]:
        rng = np.random.default_rng(0)
        lo, hi = np.array(list(PARAMS.values())).T
        for n in (1, 60):
            t = time.time()
            ev = evaluate(simulate(lo + (hi - lo) * rng.random((n, len(PARAMS))), name=f"outnet_speed{n}"))
            print(f"{n} candidates: {time.time() - t:.1f} s, best score {ev['score'].max():.3f}")
    elif sys.argv[1:] == ["hb"]:
        hb_verify(json.loads((RESULTS / "outnet_best.json").read_text()))
    elif sys.argv[1:] == ["report"]:
        report(json.loads((RESULTS / "outnet_best.json").read_text()))
    else:
        best = optimize()
        report(best)
