"""Export netlists from schematics.

Opening a workspace from Python (automation mode) is not safe on the real one:
the design kits don't boot fully, and when the process exits ADS rewrites the
config files (hpeesofsim.cfg, de_sim.cfg) of whatever directory it exits in,
with the design-kit paths blanked. So the export copies the workspace to a temp
dir and runs in a child process whose working directory is that copy. The real
workspace's config files are checked afterwards and restored if they changed.

Run with the ADS GUI closed:
    python_utils/adsenv.sh python -m adsutil.workspace parasitic_tune
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from . import NETLIST_DIR, WORKSPACE

LIBRARY = "radiodesign_lib"
GUARDED_FILES = ["ADSlibconfig", "hpeesofsim.cfg", "de_sim.cfg", "lib.defs"]


def gui_running() -> bool:
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if b"hpeesofde" in cmdline.read_bytes().split(b"\0")[0]:
                return True
        except OSError:
            pass
    return False


def _snapshot(workspace: Path) -> dict[str, bytes | None]:
    return {f: (workspace / f).read_bytes() if (workspace / f).exists() else None for f in GUARDED_FILES}


def _restore(workspace: Path, before: dict[str, bytes | None]) -> list[str]:
    changed = []
    for f, content in before.items():
        path = workspace / f
        now = path.read_bytes() if path.exists() else None
        if now != content:
            changed.append(f)
            if content is None:
                path.unlink()
            else:
                path.write_bytes(content)
    return changed


def export_netlist(cell: str, library: str = LIBRARY, out: Path | None = None) -> Path:
    """Write the netlist of ``library:cell:schematic`` to netlists/<cell>.net."""
    if gui_running():
        raise RuntimeError("ADS is running; close it first (automation would fight over the workspace)")
    out = Path(out or NETLIST_DIR / f"{cell}.net").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    before = _snapshot(WORKSPACE)
    with tempfile.TemporaryDirectory(prefix="ads_export_") as tmp:
        copy = Path(tmp) / WORKSPACE.name
        shutil.copytree(
            WORKSPACE, copy, symlinks=True,
            ignore=shutil.ignore_patterns(".git", "python_utils", "simulation", "*.ds", "*_data"),
        )
        subprocess.run(
            [sys.executable, "-m", "adsutil.workspace", "--child", str(copy), library, cell, str(out)],
            cwd=copy, check=True,
        )
    changed = _restore(WORKSPACE, before)
    if changed:
        print(f"warning: export touched {changed} in the real workspace; restored them", file=sys.stderr)
    return out


def _export_in_child(copy: Path, library: str, cell: str, out: Path) -> None:
    import keysight.ads.de as de
    from keysight.ads.de import db_uu as db

    with warnings.catch_warnings():
        # design-kit palette AEL fails in automation; harmless for netlisting
        warnings.simplefilter("ignore", UserWarning)
        de.open_workspace(str(copy))
    try:
        out.write_text(db.open_design((library, cell, "schematic")).generate_netlist())
    finally:
        de.close_workspace()
        os.chdir(copy)  # ADS writes cfg files into the exit directory


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        _export_in_child(Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5]))
    else:
        for cell in sys.argv[1:] or ["parasitic_tune"]:
            print(export_netlist(cell))
