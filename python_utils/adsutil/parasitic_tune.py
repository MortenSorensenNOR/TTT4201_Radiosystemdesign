"""The parasitic_tune testbench: CG2H40010F with ideal harmonic loads.

Variables in the schematic (VAR blocks):
    rf_freq         fundamental frequency                  (S1P_Calculations)
    RL, XL          fundamental load  Z1 = RL + j*XL       (Fund_Z)
    X2, X3, X4      harmonic loads    Zn = j*Xn            (Harm_SwpVars)
    TL1 (TLIN)      series line drain -> bias/load node, E at 3.8 GHz
Measurement:
    ZLi = -Vdsi/Idi, the load seen by the intrinsic drain current source (Meas1)

Typical use:
    tb = Testbench.load()
    tb.set_frequency(1.0e9)
    tb.set_fundamental_load(40, 35)
    tb.set_harmonic_reactance(2, -300)
    zli = tb.simulate().intrinsic_load()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from . import NETLIST_DIR, results, sim
from .netlist import Netlist

CELL = "parasitic_tune"
ANALYSIS = "HB1"
HARMONIC_VARS = {2: "X2", 3: "X3", 4: "X4"}


class Testbench:
    def __init__(self, netlist: Netlist):
        self.netlist = netlist.copy()
        # the schematic's own ParamSweeps would override what we set here
        self.netlist.remove_sweeps()
        self._freqs: list[float] | None = None

    @classmethod
    def load(cls, path: str | Path = NETLIST_DIR / f"{CELL}.net") -> "Testbench":
        return cls(Netlist.from_file(path))

    # -- single operating point ---------------------------------------------------

    def set_frequency(self, f0: float) -> None:
        """Fundamental (reference) frequency in Hz."""
        self.netlist.set("rf_freq", f0)

    def set_fundamental_load(self, RL: float, XL: float) -> None:
        """Load at f0, Z1 = RL + j*XL ohm."""
        self.netlist.set("RL", RL)
        self.netlist.set("XL", XL)

    def set_harmonic_reactance(self, n: int, X: float) -> None:
        """Purely reactive load at harmonic n (2..4), Zn = j*X ohm."""
        self.netlist.set(HARMONIC_VARS[n], X)

    def set_drain_line(self, E_deg: float | None = None, Z_ohm: float | None = None) -> None:
        """Series line between the drain and the bias/load node (TLIN TL1, E at F = 3.8 GHz)."""
        if E_deg is not None:
            self.netlist.set_instance_param("TL1", "E", E_deg)
        if Z_ohm is not None:
            self.netlist.set_instance_param("TL1", "Z", f"{Z_ohm} Ohm")

    def set_bias_line_length(self, mm: float) -> None:
        """Length of the high-impedance drain feed line (TL1 in drain_bias, 11 mm in the schematic)."""
        self.netlist.set_instance_param("TL1", "L", f"{mm} mm", subcircuit="drain_bias")

    def set_input_power(self, dBm: float) -> None:
        """Available input power of PORT1 (the schematic uses 0 dBm)."""
        self.netlist.set_instance_param("PORT1", "P[1]", f"polar(dbmtow({dBm}),0)")

    # -- several frequencies in one simulation -------------------------------------

    def sweep_frequencies(
        self,
        freqs: Sequence[float],
        RL: Sequence[float],
        XL: Sequence[float],
        harmonic_reactances: dict[int, Sequence[float]] | None = None,
        drain_line_deg: Sequence[float] | None = None,
        drain_line_Z: Sequence[float] | None = None,
    ) -> None:
        """One simulation over several points, each with its own settings.

        All sequences are per point and must have the same length. ``freqs``
        may repeat, e.g. to scan the drain line at each frequency.
        """
        lists = {"rf_freq": freqs, "RL": RL, "XL": XL}
        for n, X in (harmonic_reactances or {}).items():
            lists[HARMONIC_VARS[n]] = X
        if drain_line_deg is not None:
            self.set_drain_line(E_deg="drain_line_deg")
            lists["drain_line_deg"] = drain_line_deg
        if drain_line_Z is not None:
            self.netlist.set_instance_param("TL1", "Z", "drain_line_Z")
            lists["drain_line_Z"] = drain_line_Z
        self.netlist.sweep_lists(ANALYSIS, lists, index="freq_index")
        self._freqs = list(freqs)

    # -- run -------------------------------------------------------------------

    def simulate(self, name: str = CELL) -> "Result":
        return Result(sim.run(self.netlist, name=name), self._freqs)


class Result:
    def __init__(self, ds_path: Path, freqs: list[float] | None):
        self.ds_path = ds_path
        self.blocks = results.load(ds_path)
        self._freqs = freqs

    def intrinsic_load(self, harmonics: Sequence[int] = (1, 2)) -> pd.DataFrame:
        """ZLi at each harmonic: one row per fundamental, columns Z1, Z2, ..."""
        df = results.find(self.blocks, "ZLi")
        if self._freqs is None:
            f0 = self._single_f0(df)
            row = {"f0": f0} | {f"Z{k}": results.at_harmonic(df, "ZLi", f0, k).iloc[0] for k in harmonics}
            return pd.DataFrame([row])
        # the sweep index is 1-based; map each row to its fundamental
        f0 = df["freq_index"].round().astype(int).map(lambda i: self._freqs[i - 1])
        out = pd.DataFrame({"f0": self._freqs})
        for k in harmonics:
            out[f"Z{k}"] = results.at_harmonic(df, "ZLi", f0, k).to_numpy()
        return out

    @staticmethod
    def _single_f0(df: pd.DataFrame) -> float:
        # smallest non-zero frequency in the HB spectrum
        return float(df.loc[df["freq"] > 0, "freq"].min())
