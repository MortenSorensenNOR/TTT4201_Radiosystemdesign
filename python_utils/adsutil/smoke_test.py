"""Check that export, simulation and result reading work end to end.

    python_utils/adsenv.sh python -m adsutil.smoke_test
"""

import time

from .parasitic_tune import Testbench


def single_point():
    tb = Testbench.load()
    tb.set_frequency(1.0e9)
    tb.set_fundamental_load(RL=40, XL=35)
    tb.set_harmonic_reactance(2, -300)
    t = time.time()
    zli = tb.simulate("smoke_single").intrinsic_load()
    print(f"single point ({time.time() - t:.1f}s)\n{zli}\n")


def frequency_sweep():
    tb = Testbench.load()
    tb.sweep_frequencies(
        freqs=[0.9e9, 1.0e9, 1.1e9],
        RL=[40, 40, 40],
        XL=[35, 35, 35],
        harmonic_reactances={2: [-300, -300, -300]},
    )
    t = time.time()
    zli = tb.simulate("smoke_sweep").intrinsic_load()
    print(f"3-frequency sweep ({time.time() - t:.1f}s)\n{zli}\n")


if __name__ == "__main__":
    single_point()
    frequency_sweep()
