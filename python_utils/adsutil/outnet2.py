"""Layout-aware output networks (step 4c): the v3 ADS network electrically was
    short line -> 2f0 trap stub -> one big shunt C (a 12 mm patch) -> line -> small shunt C -> line,
which does not lay out (patches wider than long, stubs running along wide lines). Two buildable
versions of the same idea, both with the user's real drain bias (drain_bias from the
s30_output_match export, 0603DC-10N) and the 12 pF Johanson DC block:

  B1 "cap":  the shunt Cs are Johanson CapJT (continuous C) to ground through a pad and 3 vias
  B2 "stub": the shunt Cs are symmetric double open stubs on a cross junction (no vias)

        RF_drain --Tee(bias)-- TLa --Tee(trap stub)-- TLb --[shunt 1]-- TLc --[shunt 2]-- TLd -- 12 pF -- out

Layout rules (penalties): series lines and stubs at least as long as wide, junction width
ratio <= 4, room for the trap stub next to the first shunt. Scored like outnet.py, on a
0.05 GHz grid (maps interpolated) so a resonance cannot hide between points.

    python_utils/adsenv.sh python -m adsutil.outnet2 cap|stub        # optimize + HB check
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd

from . import PACKAGE_DIR, results, sim
from .clipping import margin
from .netlist import Netlist

RESULTS = PACKAGE_DIR.parent / "results"
WORKSPACE = PACKAGE_DIR.parent.parent
EXPORT = PACKAGE_DIR.parent / "netlists" / "s30_output_match.net"
DRAIN_L = "0603DC-10N"
# load line: 39 ohm for 38.5-39 dBm (was 27.9 ohm / Re(Z1i) 21-32 ohm for about 41 dBm)
R_L, M_MIN, RE_MIN, RE_MAX, G2_MIN = 39.0, 0.05, 32.0, 45.0, 0.85
HARM_MAP = "drain_map_harm_39.csv"
FREQS = np.round(np.arange(2.4e9, 3.6e9 + 1, 0.05e9))
TH = np.linspace(0, 2 * np.pi, 181)
WALL = "Wall1=1.0E+30 mm Wall2=1.0E+30 mm Mod=1"
PAD_W, PAD_L, N_VIA = 0.9, 0.6, 3      # capacitor branch: pad to the cap, pad to the vias
GAP = 0.5                               # minimum copper gap used in the spacing rule
W50 = 2.8                               # about 50 ohm on 1.52 mm FR-4
MAX_RATIO = 6                           # junction width ratio (MTEE/MCROS model validity, checked in EM)

COMMON = {"Wa": (1.0, 7), "La": (1.0, 4), "Wt": (0.4, 1.5), "Lt": (2.0, 10),
          "Wb": (0.5, 6), "Lb": (1.0, 10), "Wc": (0.5, 6), "Lc": (1.0, 12), "Wd": (0.5, 6), "Ld": (1.0, 10)}
PARAMS = {
    "cap": COMMON | {"C1": (0.3, 5.0), "C2": (0.1, 3.0)},
    "stub": COMMON | {"Ws1": (0.5, 5), "Ls1": (0.5, 10), "Ws2": (0.5, 5), "Ls2": (0.5, 10)},
}
# stepped variants: drain pad, trap stub, then high/low/high/low/high sections joined by MSTEPs
HI, LO = (0.4, 1.5), (2.0, 7.0)          # widths: high-Z sections, low-Z sections (<= 7 mm, below width resonance)
STEP = {"Wa": (1.0, 7), "La": (1.0, 4), "Wt": (0.4, 1.5), "Lt": (2.0, 10),
        "Wh1": HI, "Lh1": (1.0, 10), "Wl1": LO, "Ll1": (0.5, 8), "Wh2": HI, "Lh2": (0.5, 10),
        "Wl2": LO, "Ll2": (0.5, 8), "Wh3": HI, "Lh3": (0.5, 10)}
PARAMS |= {"step": STEP, "stepcap": STEP | {"C1": (0.2, 4.0), "C2": (0.2, 4.0)}}
PARAMS |= {"steppair0603": PARAMS["stepcap"], "steppair0402": PARAMS["stepcap"]}
# "paper": the paper's Fig. 10 on our FR-4: one line width throughout, bias tee after a short drain line,
# two open stubs on opposite sides as quarter-wave traps spread over the 2f0 band
PARAMS |= {"paper": {"W": (2.0, 6.0), "L0": (0.5, 3.0), "La": (1.0, 15.0), "Lb": (1.0, 15.0), "Lc": (1.0, 20.0),
                     "Ws1": (0.8, 2.5), "Ls1": (3.0, 12.0), "Ws2": (0.8, 2.5), "Ls2": (3.0, 12.0)}}
PAPER_X0 = {"W": 4.1, "L0": 0.8, "La": 7.5, "Lb": 4.5, "Lc": 17.0, "Ws1": 1.5, "Ls1": 7.5, "Ws2": 1.5, "Ls2": 6.0}
# paired shunt capacitors: one on each side of the line (cross junction), each with its own pads and vias
PKG = {"0603": dict(esl=0.3, pad_w=0.9, pad_l=0.6), "0402": dict(esl=0.2, pad_w=0.6, pad_l=0.4)}
STEP_SECTIONS = ["h1", "l1", "h2", "l2", "h3"]
CAP_AT = {"l1": "C1", "l2": "C2"}         # stepcap: shunt CapHQ at the end of these sections


def _defines() -> str:
    """CapJT and drain_bias subcircuits, copied from the exported user schematic."""
    text = EXPORT.read_text()
    blocks = []
    for name in ("CapJT", "drain_bias"):
        m = re.search(rf"define {name} \(.*?\nend {name}\n", text, re.S)
        blocks.append(m.group(0))
    out = "\n".join(blocks)
    return re.sub(r"0603DC-\w+\.s2p", f"{DRAIN_L}.s2p", out)


def _shunt(kind: str, k: int, node: str, w_in: str, w_out: str, n_out: str) -> list[str]:
    if kind == "cap":
        return [f'MTEE_ADS:Ts{k}  {node} {n_out} p{k} Subst="MSub1" W1={w_in} mm W2={w_out} mm W3={PAD_W} mm',
                f'MLIN2:Pc{k}  p{k} q{k} Subst="MSub1" W={PAD_W} mm L={PAD_L} mm {WALL}',
                f"CapJT:C{k}x  q{k} g{k} C=C{k} pF",
                f'MLIN2:Pg{k}  g{k} v{k} Subst="MSub1" W={PAD_W} mm L={PAD_L} mm {WALL}',
                *(f"VIA:V{k}_{i}  v{k} 0 D1=0.375 mm D2=0.375 mm H=1.52 mm T=0.00375 mm" for i in range(N_VIA))]
    return [f'MCROS:Xs{k}  {node} s{k}a {n_out} s{k}b Subst="MSub1" W1={w_in} mm W2=Ws{k} mm W3={w_out} mm W4=Ws{k} mm',
            f'MLOC2:S{k}a  s{k}a Subst="MSub1" W=Ws{k} mm L=Ls{k} mm {WALL}',
            f'MLOC2:S{k}b  s{k}b Subst="MSub1" W=Ws{k} mm L=Ls{k} mm {WALL}']


def _cap_branch(k: str, node: str, w_in: str, w_out: str, n_out: str) -> list[str]:
    return [f'MTEE_ADS:Tc{k}  {node} {n_out} p{k} Subst="MSub1" W1={w_in} mm W2={w_out} mm W3={PAD_W} mm',
            f'MLIN2:Pc{k}  p{k} q{k} Subst="MSub1" W={PAD_W} mm L={PAD_L} mm {WALL}',
            f"CapHQ:Cx{k}  q{k} g{k} C={CAP_AT[k]} pF",
            f'MLIN2:Pg{k}  g{k} v{k} Subst="MSub1" W={PAD_W} mm L={PAD_L} mm {WALL}',
            *(f"VIA:V{k}_{i}  v{k} 0 D1=0.375 mm D2=0.375 mm H=1.52 mm T=0.00375 mm" for i in range(N_VIA))]


def _cap_pair(k: str, node: str, w_in: str, w_out: str, n_out: str, pkg: str) -> list[str]:
    g = PKG[pkg]
    rows = [f'MCROS:Xc{k}  {node} pa{k} {n_out} pb{k} Subst="MSub1" W1={w_in} mm W2={g["pad_w"]} mm '
            f'W3={w_out} mm W4={g["pad_w"]} mm']
    for side in "ab":
        rows += [f'MLIN2:Pc{k}{side}  p{side}{k} q{k}{side} Subst="MSub1" W={g["pad_w"]} mm L={g["pad_l"]} mm {WALL}',
                 f"Short:Ic_{k}{side}  q{k}{side} r{k}{side} Mode=0 SaveCurrent=yes",   # current probe
                 f"CapHQ{pkg}:Cx{k}{side}  r{k}{side} g{k}{side} C={CAP_AT[k]} pF",
                 f'MLIN2:Pg{k}{side}  g{k}{side} v{k}{side} Subst="MSub1" W={g["pad_w"]} mm L={g["pad_l"]} mm {WALL}',
                 *(f"VIA:V{k}{side}_{i}  v{k}{side} 0 D1=0.375 mm D2=0.375 mm H=1.52 mm T=0.00375 mm"
                   for i in range(N_VIA))]
    return rows


def stepped(kind: str, drain: str, dc: str, out: str) -> list[str]:
    rows = [f'MTEE_ADS:Tb  na {drain} nbias Subst="MSub1" W1=Wa mm W2=Wa mm W3=0.5 mm',
            f"drain_bias:Xb  {dc} nbias",
            f'MLIN2:TLa  na n1 Subst="MSub1" W=Wa mm L=La mm {WALL}',
            f'MTEE_ADS:Tt  n1 t0 nt Subst="MSub1" W1=Wa mm W2=Wh1 mm W3=Wt mm',
            f'MLOC2:Strap  nt Subst="MSub1" W=Wt mm L=Lt mm {WALL}']
    node, w_prev = "t0", "Wh1"
    secs = STEP_SECTIONS + ["out50"]
    for i, sec in enumerate(STEP_SECTIONS):
        w, l = f"W{sec}", f"L{sec}"
        nxt_w = f"W{secs[i + 1]}" if secs[i + 1] != "out50" else str(W50)
        if w != w_prev:                                   # width change without a tee: MSTEP
            rows.append(f'MSTEP:St_{sec}  {node} s_{sec} Subst="MSub1" W1={w_prev} mm W2={w} mm')
            node = f"s_{sec}"
        rows.append(f'MLIN2:TL_{sec}  {node} e_{sec} Subst="MSub1" W={w} mm L={l} mm {WALL}')
        node, w_prev = f"e_{sec}", w
        if kind == "stepcap" and sec in CAP_AT:           # the tee also models the width change
            rows += _cap_branch(sec, node, w, nxt_w, f"c_{sec}")
            node, w_prev = f"c_{sec}", nxt_w
        elif kind.startswith("steppair") and sec in CAP_AT:  # cross junction, one cap each side
            rows += _cap_pair(sec, node, w, nxt_w, f"c_{sec}", kind[-4:])
            node, w_prev = f"c_{sec}", nxt_w
    if w_prev != str(W50):
        rows.append(f'MSTEP:St_out  {node} s_out Subst="MSub1" W1={w_prev} mm W2={W50} mm')
        node = "s_out"
    rows += [f'MLIN2:TLout  {node} n4 Subst="MSub1" W={W50} mm L=5 mm {WALL}',
             f"CapHQ:Cblk  n4 {out} C=12 pF"]
    return rows


def paper(drain: str, dc: str, out: str) -> list[str]:
    return [f'MLIN2:TL0  {drain} n0 Subst="MSub1" W=W mm L=L0 mm {WALL}',
            'MTEE_ADS:Tb  n0 na nbias Subst="MSub1" W1=W mm W2=W mm W3=0.5 mm',
            f"drain_bias:Xb  {dc} nbias",
            f'MLIN2:TLa  na n1 Subst="MSub1" W=W mm L=La mm {WALL}',
            'MTEE_ADS:T1  n1 n1b s1 Subst="MSub1" W1=W mm W2=W mm W3=Ws1 mm',
            f'MLOC2:S1  s1 Subst="MSub1" W=Ws1 mm L=Ls1 mm {WALL}',
            f'MLIN2:TLb  n1b n2 Subst="MSub1" W=W mm L=Lb mm {WALL}',
            'MTEE_ADS:T2  n2 n2b s2 Subst="MSub1" W1=W mm W2=W mm W3=Ws2 mm',
            f'MLOC2:S2  s2 Subst="MSub1" W=Ws2 mm L=Ls2 mm {WALL}',
            f'MLIN2:TLc  n2b n3 Subst="MSub1" W=W mm L=Lc mm {WALL}',
            f'MSTEP:Sto  n3 n4 Subst="MSub1" W1=W mm W2={W50} mm',
            f'MLIN2:TLout  n4 n5 Subst="MSub1" W={W50} mm L=5 mm {WALL}',
            f"CapHQ:Cblk  n5 {out} C=12 pF"]


def network(kind: str, drain: str, dc: str, out: str) -> list[str]:
    if kind == "paper":
        return paper(drain, dc, out)
    if kind.startswith("step"):
        return stepped(kind, drain, dc, out)
    return [
        f'MTEE_ADS:Tb  na {drain} nbias Subst="MSub1" W1=Wa mm W2=Wa mm W3=0.5 mm',
        f"drain_bias:Xb  {dc} nbias",
        f'MLIN2:TLa  na n1 Subst="MSub1" W=Wa mm L=La mm {WALL}',
        f'MTEE_ADS:Tt  n1 n1b nt Subst="MSub1" W1=Wa mm W2=Wb mm W3=Wt mm',
        f'MLOC2:Strap  nt Subst="MSub1" W=Wt mm L=Lt mm {WALL}',
        f'MLIN2:TLb  n1b n2 Subst="MSub1" W=Wb mm L=Lb mm {WALL}',
        *_shunt(kind, 1, "n2", "Wb", "Wc", "n2b"),
        f'MLIN2:TLc  n2b n3 Subst="MSub1" W=Wc mm L=Lc mm {WALL}',
        *_shunt(kind, 2, "n3", "Wc", "Wd", "n3b"),
        f'MLIN2:TLd  n3b n4 Subst="MSub1" W=Wd mm L=Ld mm {WALL}',
        f"CapJT:Cblk  n4 {out} C=12 pF",
    ]


CAPHQ = """define CapHQ ( P1  P2 )
parameters  C=1.0 pF
C:Cs  P1 n1 C=C
L:Ls  n1 n2 L=0.3 nH R=0
R:Rs  n2 P2 R=0.25 Ohm Noise=no
C:Cp1  P1 0 C=0.03 pF
C:Cp2  P2 0 C=0.03 pF
end CapHQ
"""  # 0603 high-Q per the Johanson RF datasheet: ESR ~0.2 ohm at 2 GHz, SRF ~7 GHz at 2 pF


def caphq(pkg: str) -> str:
    return CAPHQ.replace("define CapHQ", f"define CapHQ{pkg}").replace("end CapHQ", f"end CapHQ{pkg}") \
        .replace("L=0.3 nH", f"L={PKG[pkg]['esl']} nH")


def header() -> list[str]:
    return ['Options ResourceUsage=yes UseNutmegFormat=no EnableOptim=no TopDesignName="py:outnet2"',
            "model MSub1 MSUB H=1.52 mm Er=4.6 Mur=1 Cond=5.96e7 Hu=1e+33 mm T=35 um TanD=0.02 Rough=0 mm "
            "DielectricLossModel=1 FreqForEpsrTanD=1.0 GHz LowFreqForTanD=1.0 kHz HighFreqForTanD=1.0 THz RoughnessModel=2",
            _defines(), CAPHQ, caphq("0603"), caphq("0402")]


def sp_netlist(kind: str) -> Netlist:
    rows = header() + network(kind, "npin", "0", "nout") + [
        "Port:P1  npin 0 Num=1 Z=50 Ohm Noise=no",
        "Port:P2  nout 0 Num=2 Z=50 Ohm Noise=no",
        *(f"{k}={(lo + hi) / 2:g}" for k, (lo, hi) in PARAMS[kind].items()),
        'S_Param:SP1 CalcS=yes CalcY=no CalcZ=no SweepVar="freq" SweepPlan="SP1_stim"',
        "SweepPlan: SP1_stim Start=2.4 GHz Stop=7.2 GHz Step=50 MHz"]
    return Netlist("\n".join(rows) + "\n")


# -- maps on the fine grid ------------------------------------------------------------------

def load_maps():
    m1 = pd.read_csv(RESULTS / "drain_map.csv")
    mh = pd.read_csv(RESULTS / HARM_MAP)
    for df in (m1, mh):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(complex)
    m1 = m1[(m1["n"] == 1) & (m1["Pin"] == 32.0)].sort_values("f")
    mh = mh.sort_values("f0")
    f1, fh = m1["f"].to_numpy().real, mh["f0"].to_numpy().real
    interp = lambda f, x: np.interp(FREQS, f, x.real) + 1j * np.interp(FREQS, f, x.imag)
    return {1: tuple(interp(f1, m1[k].to_numpy()) for k in "abc"),
            2: tuple(interp(fh, mh[f"{k}2"].to_numpy()) for k in "abc")}


MAPS = None


def layout_penalty(kind: str, X: np.ndarray) -> np.ndarray:
    p = dict(zip(PARAMS[kind], X.T))
    pen = np.zeros(len(X))
    if kind == "paper":
        # stub 1 hangs down, stub 2 and the bias feed go up: keep stub 2 clear of the bias parts
        pen += np.clip(3.0 - (p["La"] + p["Lb"]), 0, None)
        pen += np.clip(p["W"] / np.minimum(p["Ws1"], p["Ws2"]) - MAX_RATIO, 0, None)
        return pen
    if kind.startswith("step"):
        # the trap stub hangs down beside the first high-Z section; keep it clear of the first wide one
        pen += np.clip(p["Wt"] / 2 + GAP - p["Lh1"], 0, None)
        w = np.stack([p["Wa"], p["Wh1"], p["Wt"]])
        pen += np.clip(w.max(0) / w.min(0) - MAX_RATIO, 0, None)
        if kind == "stepcap" or kind.startswith("steppair"):  # capacitor pads need room along the line
            pw = PKG[kind[-4:]]["pad_w"] if kind.startswith("steppair") else PAD_W
            pen += np.clip(pw + GAP - p["Lh2"], 0, None) + np.clip(pw + GAP - p["Lh3"], 0, None)
            if kind.startswith("steppair"):                # lower cap at l1 must clear the trap stub
                pen += np.clip((p["Wt"] + pw) / 2 + GAP - p["Lh1"] - p["Ll1"], 0, None)
        return pen
    # at least as long as wide; the first section is the drain pad and may be short and wide
    for w, l in (("Wb", "Lb"), ("Wc", "Lc"), ("Wd", "Ld"), ("Wt", "Lt")):
        pen += np.clip(p[w] - p[l], 0, None)
    junctions = [("Wa", "Wb", "Wt"), ("Wb", "Wc"), ("Wc", "Wd")]
    if kind == "stub":
        junctions[1] += ("Ws1",); junctions[2] += ("Ws2",)
        pen += np.clip(p["Ws1"] - p["Ls1"], 0, None) + np.clip(p["Ws2"] - p["Ls2"], 0, None)
        # the trap stub hangs down next to shunt 1's lower stub: keep them apart
        pen += np.clip((p["Wt"] + p["Ws1"]) / 2 + GAP - p["Lb"], 0, None)
        pen += np.clip((p["Ws1"] + p["Ws2"]) / 2 + GAP - p["Lc"], 0, None)
    else:
        pen += np.clip((p["Wt"] + PAD_W) / 2 + GAP - p["Lb"], 0, None)
        pen += np.clip(PAD_W + GAP - p["Lc"], 0, None)
    for j in junctions:
        w = np.stack([p[k] for k in j])
        pen += np.clip(w.max(0) / w.min(0) - MAX_RATIO, 0, None)
    return pen


def simulate(kind: str, X: np.ndarray, name: str) -> dict:
    nl = sp_netlist(kind)
    nl.sweep_lists("SP1", {k: X[:, i] for i, k in enumerate(PARAMS[kind])}, index="pt")
    df = results.find(results.load(sim.run(nl, name=name)), "S[1,1]").sort_values(["pt", "freq"])
    f = np.sort(df["freq"].unique())
    n = len(X)
    return dict(f=f, s11=df["S[1,1]"].to_numpy().reshape(n, len(f)), s21=df["S[2,1]"].to_numpy().reshape(n, len(f)))


def evaluate(sp: dict) -> dict:
    global MAPS
    MAPS = MAPS or load_maps()
    f = sp["f"]
    idx = lambda ff: np.array([np.argmin(np.abs(f - x)) for x in ff])
    i1, i2 = idx(FREQS), idx(2 * FREQS)
    zpin = 50 * (1 + sp["s11"]) / (1 - sp["s11"])
    ap = lambda m, Z: (m[0] * Z + m[1]) / (m[2] * Z + 1)
    z1i, z2i = ap(MAPS[1], zpin[:, i1]), ap(MAPS[2], zpin[:, i2])
    s11, s21 = sp["s11"][:, i1], sp["s21"][:, i1]
    g2 = np.abs(sp["s11"][:, i2])
    eta = np.abs(s21) ** 2 / np.maximum(1 - np.abs(s11) ** 2, 1e-9)
    mg = margin(z1i / R_L, (np.maximum(z2i.real, 0) + 1j * z2i.imag) / R_L, TH)
    score_f = z1i.real / R_L * eta
    pen = (10 * np.clip(M_MIN - mg, 0, None).sum(-1) + 20 * np.clip(RE_MIN - z1i.real, 0, None).sum(-1) / R_L
           + 10 * np.clip(G2_MIN - g2, 0, None).sum(-1) + 20 * np.clip(z1i.real - RE_MAX, 0, None).sum(-1) / R_L)
    return dict(cost=-score_f.min(-1) + pen, score=score_f.min(-1), z1i=z1i, z2i=z2i, margin=mg, eta=eta, g2=g2)


def optimize(kind: str, maxiter: int = 200, popsize: int = 10, seed: int = 2, x0: dict | None = None) -> dict:
    from scipy.optimize import differential_evolution
    log = RESULTS / f"outnet2_{kind}_log.txt"
    log.write_text("")
    gen, t0 = [0], time.time()

    def cost_fn(xT):
        X = xT.T
        ev = evaluate(simulate(kind, X, f"outnet2_{kind}_g{gen[0]:03d}"))
        cost = ev["cost"] + 5 * layout_penalty(kind, X)
        gen[0] += 1
        k = np.argmin(cost)
        with log.open("a") as fh:
            fh.write(f"gen {gen[0]:3d} {time.time() - t0:6.0f}s cost {cost[k]:+.4f} score {ev['score'][k]:.3f} "
                     f"margin {ev['margin'][k].min():+.3f} layout {layout_penalty(kind, X[k:k + 1])[0]:.2f}\n")
        return cost

    res = differential_evolution(cost_fn, list(PARAMS[kind].values()), vectorized=True, updating="deferred",
                                 maxiter=maxiter, popsize=popsize, seed=seed, tol=1e-4, polish=False,
                                 x0=None if x0 is None else [x0[k] for k in PARAMS[kind]])
    best = dict(zip(PARAMS[kind], map(float, res.x)))
    (RESULTS / f"outnet2_{kind}.json").write_text(json.dumps(best, indent=1))
    return best


def hb_verify(kind: str, best: dict, pins=(28, 30, 32)):
    from .loadbench import TEMPLATE as BENCH, Bench, summarize
    drop = ("Short:DC_Feed_D", "Short:Ids", "Short:DC_Block2", "Short:Iload", "S_Port:LoadBox", "Options")
    bench = [l for l in BENCH.splitlines() if not l.startswith(drop)]
    rows = header() + bench + network(kind, "ndrain", "nbc", "nload") + [
        "Short:Ids  nvd nbc Mode=0 SaveCurrent=yes",
        "Short:Iload  nload nbox Mode=0 SaveCurrent=yes",
        "Port:Term1  nbox 0 Num=2 Z=50 Ohm Noise=no",
        *(f"{k}={v:.6g}" for k, v in best.items())]
    b = Bench(Netlist("\n".join(rows) + "\n"))
    b.set_pin_sweep(20, max(pins), 2)
    b.netlist.sweep_lists("HB1", {"rf_freq": list(FREQS)}, index="pt")
    r = summarize(results.load(sim.run(b.netlist, name=f"outnet2_{kind}_hb")))
    r["f0_GHz"] = FREQS[r["pt"].round().astype(int).to_numpy() - 1] / 1e9
    r.to_csv(RESULTS / f"outnet2_{kind}_hb.csv", index=False)
    t = r[r["Pin"].isin(pins)].pivot_table(index="f0_GHz", columns="Pin", values=["Pout_dBm", "DE"])
    t["DE"] *= 100
    pd.set_option("display.width", 220)
    print(t.round(1).to_string())
    return r


if __name__ == "__main__":
    kind = sys.argv[1]
    if sys.argv[2:] == ["hb"]:
        hb_verify(kind, json.loads((RESULTS / f"outnet2_{kind}.json").read_text()))
    else:
        x0 = None
        if kind.startswith("steppair"):                    # start from the B4 solution, pairs at half value
            x0 = json.loads((RESULTS / "outnet2_stepcap.json").read_text())
            x0["C1"], x0["C2"] = max(x0["C1"] / 2, 0.2), max(x0["C2"] / 2, 0.2)
        if kind == "paper":
            x0 = PAPER_X0
        best = optimize(kind, x0=x0)
        print(json.dumps({k: round(v, 3) for k, v in best.items()}))
        hb_verify(kind, best)
