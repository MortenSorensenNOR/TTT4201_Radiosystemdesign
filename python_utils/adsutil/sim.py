"""Run the ADS circuit simulator in an isolated run directory.

The workspace's own ADSlibconfig only exists while ADS has the workspace open,
so each run directory gets its own simulator config, built from lib.defs and
the design kits' eesof_lib.cfg. Nothing is written to the workspace.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import RUNS_DIR, WORKSPACE
from .netlist import Netlist


@dataclass
class Kit:
    name: str
    root: Path
    data_paths: list[Path] = field(default_factory=list)
    veriloga: list[Path] = field(default_factory=list)
    libraries: list[tuple[str, Path]] = field(default_factory=list)


def _read_cfg(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def design_kits(workspace: Path = WORKSPACE) -> list[Kit]:
    """Design kits INCLUDEd from the workspace lib.defs."""
    kits = []
    for line in (workspace / "lib.defs").read_text().splitlines():
        m = re.match(r"\s*INCLUDE\s+(\S+)", line)
        if not m or "$" in m.group(1):
            continue
        root = (workspace / m.group(1)).parent.resolve()
        for cfg_path in sorted(root.glob("*/eesof_lib.cfg")):
            cfg, base = _read_cfg(cfg_path), cfg_path.parent
            kit = Kit(cfg.get("DESIGN_KIT_NAME", root.name), root)
            for p in filter(None, cfg.get("INPUT_DATA_PATH", "").split(";")):
                if (d := (base / p).resolve()).is_dir():
                    kit.data_paths.append(d)
            if (v := cfg.get("VERILOGA_DIRECTORY")) and (d := (base / v).resolve()).is_dir():
                kit.veriloga.append(d)
            if lc := cfg.get("ADSLIBCONFIG_DIRECTORY"):
                lib_cfg = (base / lc / "ADSlibconfig").resolve()
                if lib_cfg.is_file():
                    for entry in lib_cfg.read_text().splitlines():
                        parts = entry.split()
                        if len(parts) == 2:
                            # first path component is the kit folder name; map it to the real root
                            rel = Path(*Path(parts[1].replace("\\", "/")).parts[1:])
                            kit.libraries.append((parts[0], root / rel))
            kits.append(kit)
    return kits


def write_sim_config(run_dir: Path, workspace: Path = WORKSPACE) -> None:
    kits = design_kits(workspace)
    run_dir = run_dir.resolve()
    (run_dir / "ADSlibconfig").write_text(
        "".join(f"{name} {path}\n" for k in kits for name, path in k.libraries)
    )
    data = ";".join(str(p) for k in kits for p in k.data_paths)
    vla = ";".join(str(p) for k in kits for p in k.veriloga)
    (run_dir / "hpeesofsim.cfg").write_text(
        "EESOF_MODEL_PATH=%DESIGN_KIT_MODEL_PATH:$ICCAP_MODEL_PATH:%ICCAP_MODEL_PATH:{%DESIGN_KIT_MODEL_PATH}\n"
        "DESIGN_KIT_MODEL_PATH=\n"
        f"DESIGN_KIT_VERILOGA_PATH={vla}\n"
        "SIM_FILE_PATH=$USER_SIM_FILE_PATH:%USER_SIM_FILE_PATH:$DE_SIM_FILE_PATH:%DE_SIM_FILE_PATH:"
        ".:..:../data:../networks:%DESIGN_KIT_SIM_FILE_PATH:{%DESIGN_KIT_SIM_FILE_PATH}\n"
        f"DESIGN_KIT_SIM_FILE_PATH={data}\n"
        f"DE_SIM_FILE_PATH={workspace.resolve() / 'data'}\n"
        f"DKIT_ADSLIBCONFIG_PATH={run_dir}\n"
    )


def run(netlist: Netlist | str, name: str = "sim", run_dir: Path | None = None) -> Path:
    """Simulate a netlist and return the path to the resulting .ds file."""
    from keysight.edatoolbox import ads

    text = netlist.text if isinstance(netlist, Netlist) else netlist
    if run_dir is None:
        run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_sim_config(run_dir)
    (run_dir / f"{name}.net").write_text(text)
    ads.CircuitSimulator().run_netlist(
        text, output_dir=str(run_dir), working_dir=str(run_dir), dataset_name=name
    )
    return run_dir / f"{name}.ds"
