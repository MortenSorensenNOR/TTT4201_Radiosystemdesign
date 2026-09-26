# python_utils

Drive ADS simulations from Python (ADS 2027 bundled Python).

```bash
python_utils/adsenv.sh python -m adsutil.workspace parasitic_tune   # schematic -> netlists/parasitic_tune.net (ADS GUI must be closed)
python_utils/adsenv.sh python -m adsutil.smoke_test                 # end-to-end check
```

- `adsutil/parasitic_tune.py`: the testbench, with named setters (`set_frequency`, `set_fundamental_load`, `set_harmonic_reactance`, `sweep_frequencies`) and `intrinsic_load()` for ZLi per harmonic.
- `adsutil/netlist.py`: generic netlist editing (variables, sweeps).
- `adsutil/sim.py`: runs hpeesofsim in `runs/<timestamp>-<name>/` with its own design-kit config. It never writes to the workspace.
- `adsutil/results.py`: .ds to pandas.
- `adsutil/workspace.py`: netlist export on a temp copy of the workspace.

Re-export the netlist after changing the schematic. Don't open the real workspace from Python
(`keysight.ads.de.open_workspace`): on exit ADS rewrites the config files of the process's
working directory with the design-kit paths blanked, which breaks the kits in the GUI.
