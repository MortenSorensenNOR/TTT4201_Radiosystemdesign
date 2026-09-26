"""Helpers for driving ADS from Python without touching the open workspace.

netlist   - edit variables / sweeps in a netlist text
sim       - run hpeesofsim in an isolated run directory
results   - read .ds files into pandas
workspace - export netlists from schematics (on a copy of the workspace)
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE = PACKAGE_DIR.parent.parent
RUNS_DIR = PACKAGE_DIR.parent / "runs"
NETLIST_DIR = PACKAGE_DIR.parent / "netlists"
