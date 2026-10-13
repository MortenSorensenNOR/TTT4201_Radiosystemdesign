"""Ideal harmonic load-pull bench for the CG2H40010F, written as a netlist (no schematic).

    gate:   PORT1 (50 ohm, Pin) -> DC block -> gate,  Vgs through an ideal feed
    drain:  Vds through an ideal feed -> DC block -> load box at the package drain pin

The load box is an S_Port whose reflection is chosen per harmonic: G[n] at n*f0 (n = 1..9),
referenced to 50 ohm. Harmonics without a setting are shorted.

Results per point (frequency index, Pin): Pout at the load, Pdc, drain efficiency, the
intrinsic load ZLi at each harmonic and the intrinsic waveforms vdsi(t), idi(t).

    python_utils/adsenv.sh python -m adsutil.loadbench        # smoke run at 3 GHz
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from . import results, sim
from .netlist import Netlist, fmt

ORDER = 9
Z0 = 50.0

TEMPLATE = f"""\
Options ResourceUsage=yes UseNutmegFormat=no EnableOptim=no TopDesignName="py:loadbench" DcopOutputNodeVoltages=yes DcopOutputPinCurrents=yes DcopOutputAllSweepPoints=no DcopOutputDcopType=0
#uselib "MA_RFP_ADS_v2p7" , "CG2H40010F_v2"
"CG2H40010F_v2":X1  ngate ndrain 0 Rth=7.83 Tbase=25 VDN=28 x1=1 x2=1
Port:PORT1  nsrc 0 Num=1 Z=50 Ohm P[1]=polar(dbmtow(Pin),0) Freq[1]=rf_freq Noise=no
Short:DC_Block1  nsrc ngate Mode=1
V_Source:SRC_G  nvg 0 Type="V_DC" Vdc=Vgs V SaveCurrent=1
Short:DC_Feed_G  nvg ngate Mode=-1
V_Source:SRC_D  nvd 0 Type="V_DC" Vdc=Vds V SaveCurrent=1
Short:DC_Feed_D  nvd nfeed Mode=-1
Short:Ids  nfeed ndrain Mode=0 SaveCurrent=yes
Short:DC_Block2  ndrain nload Mode=1
Short:Iload  nload nbox Mode=0 SaveCurrent=yes
S_Port:LoadBox  nbox 0 S[1,1]=SLoad Z[1]=50
aele Idi=X1.X1.idi;Vdsi=X1.X1.vdsi;ZLi=-Vdsi/Idi;
Vgs=-3
Vds=28
Pin=27
rf_freq=3e9
{chr(10).join(f"G{n}=-1" for n in range(1, ORDER + 1))}
GL=list(0,{",".join(f"G{n}" for n in range(1, ORDER + 1))})
frq=int(min(abs(freq)/(rf_freq)+1.5,length(GL)))
SLoad=GL[frq]
HB:HB1 MaxOrder=4 Freq[1]=rf_freq Hz Order[1]={ORDER} StatusLevel=2 HBSS_WSP=0 SweepVar="Pin" SweepPlan="HB1_stim" OutputPlan="HB1_Output"
SweepPlan: HB1_stim Start=20 Stop=34 Step=2
OutputPlan:HB1_Output Type="Output" UseNodeNestLevel=yes NodeNestLevel=2 UseEquationNestLevel=yes EquationNestLevel=2 UseSavedEquationNestLevel=yes SavedEquationNestLevel=2 UseCurrentNestLevel=yes CurrentNestLevel=999
"""


def gamma(z):
    z = np.asarray(z, complex)
    return (z - Z0) / (z + Z0)


@dataclass
class Bench:
    netlist: Netlist = field(default_factory=lambda: Netlist(TEMPLATE))

    def set_pin_sweep(self, start: float, stop: float, step: float) -> None:
        for i, s in enumerate(self.netlist._stmts):
            if s.startswith("SweepPlan: HB1_stim"):
                self.netlist._stmts[i] = f"SweepPlan: HB1_stim Start={start} Stop={stop} Step={step}"

    def set_points(self, f0: Sequence[float], loads: dict[int, Sequence[complex]]) -> int:
        """One point per entry: fundamental f0[i] and reflection loads[n][i] at n*f0.
        Harmonics missing from ``loads`` stay shorted."""
        lists = {"rf_freq": list(f0)} | {f"G{n}": list(g) for n, g in loads.items()}
        return self.netlist.sweep_lists("HB1", lists, index="pt")

    def run(self, name: str = "loadbench") -> pd.DataFrame:
        return summarize(results.load(sim.run(self.netlist, name=name)))


def _spectrum(blocks: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """Rows: one per point (pt, Pin). Columns: harmonic 0..ORDER of ``col``.
    The MeasEqn results sit in their own blocks without a harmindex, so the harmonic
    number comes from freq / f0 of each point."""
    df = results.find(blocks, col)
    keys = [c for c in ("pt", "Pin") if c in df.columns]
    d = df[keys + ["freq", col]].copy()
    f0 = d.groupby(keys)["freq"].transform(lambda f: f[f > 0].min())
    d["k"] = (d["freq"] / f0).round().astype(int)
    return d.set_index(keys + ["k"])[col].unstack("k")


def summarize(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    v = _spectrum(blocks, "nload")
    il = _spectrum(blocks, "Iload.i")
    ids = _spectrum(blocks, "Ids.i")
    vdsi = _spectrum(blocks, "Vdsi")
    idi = _spectrum(blocks, "Idi")
    zli = _spectrum(blocks, "ZLi")

    out = pd.DataFrame(index=v.index)
    out["Pout_W"] = 0.5 * np.real(v[1] * np.conj(il[1]))
    out["Pdc_W"] = np.real(ids[0]) * 28.0
    out["DE"] = out["Pout_W"] / out["Pdc_W"]
    out["Pout_dBm"] = 10 * np.log10(np.maximum(out["Pout_W"], 1e-12) * 1e3)
    for k in (1, 2, 3):
        out[f"Z{k}i"] = zli[k]
    # intrinsic time-domain waveforms (cos series of the HB phasors)
    t = np.linspace(0, 2 * np.pi, 361)
    ks = [k for k in vdsi.columns if k >= 0]
    def wave(spec):
        w = np.zeros((len(spec), len(t)))
        for k in ks:
            amp = spec[k].to_numpy()[:, None]
            w += np.real(amp * np.exp(1j * k * t)) if k > 0 else np.real(amp)
        return w
    vw, iw = wave(vdsi), wave(idi)
    out["vdsi_min"] = vw.min(axis=1)
    out["idi_max"] = iw.max(axis=1)
    out["idi_min"] = iw.min(axis=1)
    out.attrs["t"] = t
    out.attrs["vdsi"] = vw
    out.attrs["idi"] = iw
    return out.reset_index()


if __name__ == "__main__":
    b = Bench()
    # smoke run: 3 GHz, f0 load 13 + j6 ohm, 2f0 open, rest shorted
    b.set_points([3.0e9], {1: [gamma(13 + 6j)], 2: [1.0 + 0j]})
    res = b.run("loadbench_smoke")
    pd.set_option("display.width", 250)
    print(res.drop(columns=[]).round(3).to_string(index=False))
